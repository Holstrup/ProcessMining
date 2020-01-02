from pm4py.objects.log.util.log import log as pmlog
import os

class Conversation:
    def __init__(self, text_body, open_time, event, message_text, person):
        self.text_body = text_body # Dict
        self.open_time = open_time # Datetime Object
        self.people = set({person})
        self.message_events = [event]
        self.message_texts = [message_text]


    def similarity_score(self, message):
        score = 0
        message_listed = message.split(" ")
        for word in set(message_listed):
            words_in_message = len(message_listed)
            term_freq = message_listed.count(word)
            if word in self.text_body.keys():
                score += (term_freq / words_in_message) * self.text_body[word]
        return score


    def add_message(self, text_body, event, message_text, person):
        for word in text_body.keys():
            if word not in self.text_body:
                self.text_body[word] = text_body[word]
        self.message_events.append(event)
        self.message_texts.append(message_text)
        self.people.add(person)
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

        convotext.write("Keywords: " + str(self.text_body.keys()) + "\n")
        convotext.write("-----\n")
        convotext.close()

    def is_person_in_conversation(self, person_id):
        return person_id in self.people
