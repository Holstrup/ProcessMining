import random
import os
import json
import xml.etree.ElementTree as ET
from pm4py.objects.log.exporter.xes import factory as xes_exporter
from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects.log.util.log import log as pmlog
from Natural_Language_Processing import NLP

class Data_Processing_Transformation:
    def __init__(self, path_name):
        self.path_name = path_name             # Slack filename
        self.log_file_name = "SlackDataLog.xes"

        # Run Pipeline
        self.extraction()
        log = self.transformation()
        xes_exporter.export_log(log, self.log_file_name)




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
                    case_dict["org:resource"] = message['user_profile']['real_name']
                    case_dict["text"] = message['text']
                    case_dict['concept:name'] = str(nlp_class.get_class())
                    case_dict["time:timestamp"] = message['ts']


                    event = pmlog.Event(case_dict)
                    trace.append(event)
                except KeyError:
                    print(message)
        return log






    def classification(self, message):
        """
        This is where the magic happens. We need to take a message in
        and output a classification based on the message

        @:param message: Message String
        @:return classifiction: Q, A or C
        """

        return random.choice(self.message_types)


