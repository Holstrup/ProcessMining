import os
import json
from pm4py.objects.log.exporter.xes import factory as xes_exporter
from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects.log.util.log import log as pmlog
from Natural_Language_Processing import NLP
import pandas as pd
from datetime import datetime
from DataPreprocessing import rm_code
import time


class Data_Processing_Transformation:
    def __init__(self, path_name, social_graph, kaggle):
        self.nlp_class = NLP()

        """ Parameters """
        self.path_name = path_name             # Slack filename
        self.social_graph = social_graph
        self.chunk_size = 10 ** 3
        self.last_date = datetime.strptime("2015-04-01T00:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ')


        """ Pipelines """
        if kaggle:
            self.log_file_name = "KaggleLogData.xes"
            log = self.kaggle_extract()
            xes_exporter.export_log(log, self.log_file_name)

        else:
            self.log_file_name = "SlackLogData.xes"
            self.extraction()
            log = self.transformation()
            xes_exporter.export_log(log, self.log_file_name)



    def kaggle_extract(self):
        log = pmlog.EventLog() # Log we want to return
        q_dict = {}            # Dict of (current) questions
        q_num = 0              # Key for question in q_dict


        break_loop = False
        for chunk in pd.read_csv(self.path_name, chunksize=self.chunk_size, usecols = ["id", "fromUser.displayName", "text", "sent"]):
            if break_loop:
                break
            
            for index, row in chunk.iterrows():

                date_string = row["sent"]
                datetime_object = datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%fZ')

                # Sometimes the diplayname is nan -> causes errors in disco if not caught
                if row["fromUser.displayName"] == "nan":
                    print("Nan in index", index)
                    continue

                # Terminate after late date, to trim dataset. 
                if datetime_object > self.last_date:
                    break_loop = True
                    break

                try:
                    text = rm_code(row['text'])
                    self.nlp_class.set_sentence(text)
                    classification = self.nlp_class.get_class()
                    name = row["fromUser.displayName"]

                    case_dict = {}
                    if not self.social_graph:
                        case_dict["org:resource"] = name
                        case_dict['concept:name'] = classification
                    else:
                        case_dict['concept:name']  = classification
                        case_dict["org:resource"] = name

                    case_dict["time:timestamp"] = datetime_object

                    
                    # Base case: When Question Dict is empty and we don't have a question
                    if len(q_dict.keys()) == 0 and classification != "Question":
                        continue
                    
                    # If Question -> Add to q_dict
                    if classification == "Question":
                        q_dict[q_num] = [case_dict]
                        q_num += 1
                        
                    # If Answer -> Add to earliest question and write that trace to the log
                    elif classification == "Answer":
                        case = q_dict[min(q_dict.keys())]
                        case.append(case_dict)
                        # Write to log
                        trace = pmlog.Trace()
                        for event in case:
                            ev = pmlog.Event(event)
                            trace.append(ev)
                        log.append(trace)

                        # Delete
                        del q_dict[min(q_dict.keys())]
                    
                    # Otherwise, append the message to all traces in the dictionary 
                    else:
                        delete_keys = []
                        for key in q_dict.keys():
                            q_dict[key].append(case_dict)
                
                            if len(q_dict[key]) >= 10:
                                delete_keys.append(key)
                        # Remove traces with more than 10 events -> Write to log 
                        for key in delete_keys:
                            trace = pmlog.Trace()
                            for event in q_dict[key]:
                                ev = pmlog.Event(event)
                                trace.append(ev)
                            log.append(trace)
                            del q_dict[key]

                except TypeError:
                    continue
            print("Current date: " + str(row["sent"]))
        return log



    def extraction(self):
        directory = os.listdir(self.path_name)
        directory.sort()
        self.data = []
        for filename in directory:
            if filename.endswith(".DS_Store"):
                continue
            else:
                with open(os.path.join(self.path_name, filename), 'r') as json_file:
                    self.data.extend(json.load(json_file))


    def transformation(self):
        ts_start = 0.0
        convo_id = 0
        log = pmlog.EventLog()
        self.nlp_class = NLP()

        for message in self.data:
            if 'subtype' in message.keys():
                continue
            else:
                if ts_start == 0:
                    ts_start = float(message['ts'])
                    trace = pmlog.Trace()
                    event = pmlog.Event()

                elif abs(ts_start - float(message['ts'])) > 5500.0: #5500.0: # Split roughly every 1,5 hours
                    log.append(trace)

                    #convo_id += 1
                    ts_start = float(message['ts'])
                    trace = pmlog.Trace()

                self.nlp_class.set_sentence(message['text'])

                try:
                    case_dict = {}

                    if not self.social_graph:
                        case_dict["org:resource"] = message['user_profile']['real_name']
                        case_dict['concept:name'] = str(self.nlp_class.get_class())
                    else:
                        case_dict['concept:name']  = message['user_profile']['real_name']
                        case_dict["org:resource"] = str(self.nlp_class.get_class())


                    case_dict["text"] = message['text']
                    real_time = time.strftime('%Y-%m-%dT%H:%M:%S',  time.gmtime(float(message['ts'])))
                    real_time = datetime.strptime(real_time, '%Y-%m-%dT%H:%M:%S')
                    case_dict["time:timestamp"] = real_time


                    event = pmlog.Event(case_dict)
                    trace.append(event)
                except KeyError:
                    print("Keyerror: ", message)
        return log



