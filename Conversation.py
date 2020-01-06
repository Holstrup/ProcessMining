from pm4py.objects.log.util.log import log as pmlog
import os
import Text_Preprocessing as TP


class Conversation:
    def __init__(self, open_time, event, message_text, person, idf):
        # {'Thread ID': Str, Posts: {abs_pos(int): {'User id': Str, 'Date': Str, 'Content': Str, 'Class': Str}}
        self.tf_idf = {} # Dict
        self.open_time = open_time # Datetime Object
        self.people = set({person})
        self.message_events = [event]
        self.message_texts = [message_text]

        # Compute td-idf
        self.compute_tfidf(idf)

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


    def add_message(self, event, message_text, person, idf):
        self.message_events.append(event)
        self.message_texts.append(message_text)
        self.people.add(person)

        # Recompute tf-idf
        #self.compute_tfidf(idf)
        return None


    def add_to_trace(self):
        trace = pmlog.Trace()
        for event in self.message_events:
            trace.append(event)
        return trace


    def write_to_txt(self):
        filename = "conversations.txt"
        if os.path.exists(filename):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        convotext = open(filename, append_write)
        for message in self.message_texts:
            convotext.write("- " + message + "\n")

        convotext.write("Keywords: " + str(self.tf_idf.keys()) + "\n")
        convotext.write("-----\n")
        convotext.close()

    def is_person_in_conversation(self, person_id):
        return person_id in self.people
