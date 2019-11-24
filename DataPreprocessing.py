import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from datetime import datetime
from Natural_Language_Processing import NLP
from Conversation import Conversation
from pm4py.objects.log.util.log import log as pmlog
from pm4py.objects.log.exporter.xes import factory as xes_exporter

#nltk.download('stopwords')
def kaggle_extract():
    chunksize = 10 ** 3
    last_date = datetime.strptime("2015-02-01T00:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ')
    datetime_object = datetime.strptime("2015-01-01T00:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ')
    break_loop = False
    word_dict = {}
    for chunk in pd.read_csv("Kaggle/chatroom.csv", chunksize=chunksize, usecols=["text", "sent"]):

        # Terminate after late date, to trim dataset.
        if datetime_object > last_date:
            print(datetime_object)
            break_loop = True


        if break_loop:
            break_loop = True
            break

        for index, row in chunk.iterrows():
            try:
                text = (row["text"]).strip()
                text = text.lower()
                text = rm_mentions(text)
                text = rm_code(text)
                text = rm_punctuation(text)
                text = remove_stop_words(text)
                text = text.split(" ")

                for word in text:
                    if word not in word_dict:
                        word_dict[word] = 0
                    word_dict[word] += 1

            except AttributeError:
                continue
        date_string = row["sent"]
        datetime_object = datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%fZ')
    del word_dict['']
    print("No. Words: " + str(len(word_dict.keys())))
    return word_dict


def rm_mentions(data):
    return " ".join(filter(lambda x: x[0] != '@', data.split()))

def get_mentions(data):
    return list(filter(lambda x: x[0] == '@', data.split()))

def rm_code(data):
    return re.sub("(```.+?```)", "", data)

def rm_punctuation(data):
    return re.sub(r'[^\w\s]', '', data)


def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def mine_conversations(word_dict):
    nlp_class = NLP()
    chunksize = 10 ** 3
    last_date = datetime.strptime("2015-02-01T00:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ')
    datetime_object = datetime.strptime("2015-01-01T00:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ')
    break_loop = False
    open_conversations = []
    log = pmlog.EventLog()
    did_not_match = 0
    matches = 0
    for chunk in pd.read_csv("Kaggle/chatroom.csv", chunksize=chunksize, usecols = ["id", "fromUser.displayName", "text", "sent", "fromUser.username"]):
        # Terminate after late date, to trim dataset.
        if datetime_object > last_date:
            break_loop = True


        if break_loop:
            break_loop = True
            break

        for index, row in chunk.iterrows():
            try:
                # Start by getting the variables we need
                text = row["text"]
                if str(text) == "nan":
                    continue
                nlp_class.set_sentence(rm_code(text))
                classification = nlp_class.get_class()
                datetime_object = datetime.strptime(row["sent"], '%Y-%m-%dT%H:%M:%S.%fZ')

                case_dict = {}
                case_dict['concept:name'] = classification
                case_dict["org:resource"] = row["fromUser.displayName"]

                """ For Social Graph
                case_dict['concept:name'] = row["fromUser.displayName"]
                case_dict["org:resource"] = classification
                """

                case_dict["time:timestamp"] = datetime_object
                event = pmlog.Event(case_dict)


                # Now we find our text body
                text_body = {}
                for word in text.split(" "):
                    if word in word_dict:
                        text_body[word] = word_dict[word]

                else:
                    mention = get_mentions(text)
                    mention_added = False
                    if len(mention) > 0:
                        for conversation in open_conversations:
                            if conversation.is_person_in_conversation(mention[0][1:]):
                                conversation.add_message(text_body, event,
                                                         message_text=row["text"], person=row["fromUser.username"])
                                mention_added = True
                                break

                    # If question -> Make a new conversation
                    elif classification == "Question":
                        convo = Conversation(text_body=text_body, open_time=datetime_object,
                                             event=event, message_text=row["text"], person=row["id"])
                        open_conversations.append(convo)


                    elif not mention_added and classification != "Question":
                        # Find the best matching conversation
                        score = 0
                        best_matching_conversation = None
                        for conversation in open_conversations:
                            conversation_score = conversation.similarity_score(text)
                            if conversation_score > score:
                                score = conversation_score
                                best_matching_conversation = conversation

                        if best_matching_conversation != None:
                            matches += 1
                            best_matching_conversation.add_message(text_body, event,
                                                                   message_text=row["text"], person=row["fromUser.username"])
                        else:
                            did_not_match += 1

                    for conversation in open_conversations:
                        time_diff = (datetime_object - conversation.open_time).total_seconds() / 60.0
                        if time_diff > 15.0: #minutes
                            log.append(conversation.add_to_trace())
                            conversation.write_to_txt()
                            open_conversations.remove(conversation)
                print(len(open_conversations))
            except AttributeError:
                continue
    return log


word_dict = kaggle_extract()
log = mine_conversations(word_dict)
xes_exporter.export_log(log, "Mined_Conversations_w_mentions.xes")