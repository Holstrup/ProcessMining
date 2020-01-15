from pm4py.objects.log.util.log import log as pmlog
import os

import DataProcessing_02012020
import Message_Classifier
import Text_Preprocessing as TP
import operator

class Conversation:
    def __init__(self, open_time, event_dict, message_text, person, idf, classifier,dataprocessing):
        # {'Thread ID': Str, Posts: {abs_pos(int): {'User id': Str, 'Date': Str, 'Content': Str, 'Class': Str}}
        self.tf_idf = {} # Dict
        self.open_time = open_time # Datetime Object
        self.people = set({person})
        self.message_events = {1: event_dict}
        self.message_texts = [message_text]
        self.no_messages = 2


        # Compute td-idf
        self.compute_tfidf(idf)

        # Inistantiate data processor and  Classifier
        self.dataprocess=dataprocessing
        self.classification = classifier
        # load previous trained models from directory

    def similarity_score(self, tf_idf_message):
        score = 0
        for word in tf_idf_message.keys():
            if word in self.tf_idf:
                score += tf_idf_message[word] * self.tf_idf[word]
        return score

    def compute_tfidf(self, idf):
        tf = {"totalWords": 0}
        self.tf_idf = {}
        for message in self.message_texts:
            message_list = TP.preprocess_text(message)
            for word in message_list:
                if word in idf and word in tf:
                    tf[word] += 1
                    tf["totalWords"] += 1
                elif word in idf:
                    tf[word] = 1
                    tf["totalWords"] += 1
        for word in tf.keys():
            if word == "totalWords": continue
            else: self.tf_idf[word] = (tf[word] / tf["totalWords"]) * idf[word]


    def add_message(self, event_dict, message_text, person, idf):
        self.message_events[self.no_messages] = event_dict
        self.message_texts.append(message_text)
        self.people.add(person)

        # Recompute tf-idf
        #self.compute_tfidf(idf)
        self.no_messages += 1
        return None


    def add_to_trace(self):
        trace = pmlog.Trace()
        print(type(trace))
        event = {}



        dict = self.message_events
        print(dict)
        outer_dict = {"Thread ID":"1","Posts":dict}

        self.dataprocess.load_dict(outer_dict,False)


        dataframe=self.dataprocess.get_dataframe()

        predicted_class = self.classification.predict_class(dataframe)

        for message_no in self.message_events.keys():

            event['concept:name'] = predicted_class[int(message_no - 1)]
            event["time:timestamp"] = self.message_events[message_no]["Date"]
            event["org:resource"] = self.message_events[message_no]["User id"]
            event["keyword"] = max(self.tf_idf.items(), key=operator.itemgetter(1))[0]
            event["text"] = self.message_events[message_no]["Content"]

            event = pmlog.Event(event)

            trace.append(event)

        return trace


    def write_to_txt(self):
        filename = "conversations.txt"
        if os.path.exists(filename):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        convotext = open(filename, append_write)
        #for message in self.message_texts:
        #    convotext.write("- " + message + "\n")

        for message_no in self.message_events.keys():
            C = self.message_events[message_no]["Content"]
            U_id = self.message_events[message_no]["User id"]
            convotext.write("- " + str(U_id) + ": " + str(C) + "\n")

        convotext.write("Keywords: " + str(self.tf_idf.keys()) + "\n")
        convotext.write("-----\n")
        convotext.close()

    def is_person_in_conversation(self, person_id):
        return person_id in self.people
