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
from DataPreprocessing import rm_code
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
#nltk.download('vader_lexicon')


class Data_Processing_Transformation:
    def __init__(self, path_name, social_graph):
        self.sid = SentimentIntensityAnalyzer()
        self.path_name = path_name             # Slack filename
        self.log_file_name = "SlackDataLog.xes"
        self.social_graph = social_graph

        # Run Pipeline
        log = self.kaggle_extract()
        xes_exporter.export_log(log, "GitterLog.xes")


        #self.extraction()
        #log = self.transformation()
        #xes_exporter.export_log(log, self.log_file_name)

    def sentiment(self, message):
        ss = self.sid.polarity_scores(message)
        if ss["compound"] == 0.0:
            return "neutral"
        elif ss["compound"] > 0.0:
            return "positive"
        else:
            return "negative"


    def kaggle_extract(self):
        chunksize = 10 ** 3
        nlp_class = NLP()
        last_date = datetime.strptime("2015-02-01T00:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ')
        log = pmlog.EventLog()
        trace = pmlog.Trace()
        q_num = 0
        q_dict = {}
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

                date_string = row["sent"]
                datetime_object = datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%fZ')

                # Sometimes the diplayname is nan -> causes errors in disco if not caught
                if row["fromUser.displayName"] == "nan":
                    print(index)
                    continue

                    # Terminate at some date
                if datetime_object > last_date:
                    break_loop = True
                    break

                try:
                    text = rm_code(row['text'])
                    nlp_class.set_sentence(text)
                    classification = class_dict[str(nlp_class.get_class())]
                    name = row["fromUser.displayName"]

                    case_dict = {}
                    case_dict["org:resource"] = name
                    case_dict['concept:name']= classification
                    case_dict["time:timestamp"] = datetime_object
                    case_dict["sentiment"] = self.sentiment(row["text"])

                    #case_dict['concept:name'] = name
                    #case_dict["org:resource"] = classification

                    if len(q_dict.keys()) == 0 and classification != "Question":
                        continue

                    if classification == "Question":
                        q_dict[q_num] = [case_dict]
                        q_num += 1
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

                    else:
                        delete_keys = []
                        for key in q_dict.keys():
                            q_dict[key].append(case_dict)

                            if len(q_dict[key]) >= 10:
                                delete_keys.append(key)

                        for key in delete_keys:
                            trace = pmlog.Trace()
                            for event in q_dict[key]:
                                ev = pmlog.Event(event)
                                trace.append(ev)
                            log.append(trace)
                            del q_dict[key]

                except TypeError:
                    continue
            print(row["sent"], len(q_dict.keys()))
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



