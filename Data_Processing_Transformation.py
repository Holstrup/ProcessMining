import random
import os
import json
import xml.etree.ElementTree as ET
from pm4py.objects.log.exporter.xes import factory as xes_exporter
from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects.log.util.log import log as pmlog
from Natural_Language_Processing import NLP
import pandas as pd
from datetime import datetime

class Data_Processing_Transformation:
    def __init__(self, path_name, social_graph):
        self.path_name = path_name             # Slack filename
        self.log_file_name = "SlackDataLog.xes"
        self.social_graph = social_graph

        # Run Pipeline
        log = self.kaggle_extract()
        #xes_exporter.export_log(log, "GitterLogSocial.xes")


        #self.extraction()
        #log = self.transformation()
        #xes_exporter.export_log(log, self.log_file_name)



    def kaggle_extract(self):
        chunksize = 10 ** 3
        nlp_class = NLP()
        last_date = datetime.strptime("2015-02-01T00:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ')
        log = pmlog.EventLog()
        trace = pmlog.Trace()
        p_dict = {}
        class_dict =  {"Reject": "Answer",
                       "Statement": "Question",
                       "nAnswer" : "Answer",
                       "Accept" : "Answer",
                       "Emotion": "Other",
                       "Continuer": "Clarification",
                       "Clarify": "Clarification",
                       "ynQuestion": "Question",
                       "whQuestion": "Question",
                       "Other": "Other",
                       "Emphasis": "Clarification",
                       "System": "Other",
                       "Greet": "Other",
                       "yAnswer": "Answer",
                       "Bye": "Other"}

        break_loop = False
        for chunk in pd.read_csv(self.path_name, chunksize=chunksize, usecols = ["id", "fromUser.displayName", "text", "sent"]):
            if break_loop:
                break

            for index, row in chunk.iterrows():
                nlp_class.set_sentence(row['text'])
                date_string = row["sent"]
                datetime_object = datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%fZ')

                if row["fromUser.displayName"] == "nan":
                    print(index)

                if index == 0:
                    prev_date = datetime_object.date()
                    prev_event = str(nlp_class.get_class())
                    prev_prev_event = str(nlp_class.get_class())
                    continue

                elif datetime_object.date() > prev_date:
                    prev_date = datetime_object.date()
                    log.append(trace)
                    trace = pmlog.Trace()

                if datetime_object > last_date:
                    break_loop = True
                    break

                try:
                    """
                    case_dict = {}
                    #case_dict["org:resource"] = row["fromUser.displayName"]
                    case_dict["org:resource"] = str(nlp_class.get_class())
                    #case_dict['concept:name'] = str(nlp_class.get_class())
                    case_dict['concept:name'] = row["fromUser.displayName"]

                    case_dict["time:timestamp"] = date_string

                    event = pmlog.Event(case_dict)
                    trace.append(event)
                    """
                    event = class_dict[str(nlp_class.get_class())]

                    print(str(nlp_class.get_class()) + " : " + row["text"])

                    pattern = prev_prev_event + prev_event + event
                    if pattern in p_dict:
                        p_dict[pattern] += 1
                    else:
                        p_dict[pattern] = 1

                    prev_prev_event = prev_event
                    prev_event = event

                except TypeError:
                    continue

            print(row["sent"])
        log.append(trace)


        for key in p_dict.keys():
            if p_dict[key] > 900:
                print(key, p_dict[key])
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
        nlp_class = NLP()

        for message in self.data:
            if 'subtype' in message.keys():
                continue
            else:
                if ts_start == 0:
                    ts_start = float(message['ts'])
                    trace = pmlog.Trace()
                    event = pmlog.Event()

                elif abs(ts_start - float(message['ts'])) > 10000.0: #5500.0: # Split roughly every 1,5 hours
                    log.append(trace)

                    #convo_id += 1
                    ts_start = float(message['ts'])
                    trace = pmlog.Trace()

                nlp_class.set_sentence(message['text'])

                try:
                    case_dict = {}

                    if not self.social_graph:
                        case_dict["org:resource"] = message['user_profile']['real_name']
                        case_dict['concept:name'] = str(nlp_class.get_class())
                    else:
                        case_dict['concept:name']  = message['user_profile']['real_name']
                        case_dict["org:resource"] = str(nlp_class.get_class())


                    case_dict["text"] = message['text']
                    case_dict["time:timestamp"] = message['ts']


                    event = pmlog.Event(case_dict)
                    trace.append(event)
                except KeyError:
                    print(message)
        return log



