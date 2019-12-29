import pandas as pd
from datetime import datetime
from NLP_Classification import NLP
from Conversation import Conversation
from pm4py.objects.log.util.log import log as pmlog
import Text_Preprocessing as TP


""" Main Functions """


def mine_conversations(tf_idf, csv_file_path, stop_datetime, chunksize, conversation_duration):
    nlp_class = NLP()
    datetime_object = datetime.strptime("2015-01-01T00:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ')
    break_loop = False
    open_conversations = []
    log = pmlog.EventLog()
    for chunk in pd.read_csv(csv_file_path, chunksize=chunksize, usecols = ["id", "fromUser.displayName", "text", "sent", "fromUser.username"]):
        # Terminate after late date, to trim dataset.
        if datetime_object > stop_datetime:
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
                nlp_class.set_sentence(TP.rm_code(text))
                classification = nlp_class.get_class()
                datetime_object = datetime.strptime(row["sent"], '%Y-%m-%dT%H:%M:%S.%fZ')


                # Creating an event
                event_dict = {}
                event_dict['concept:name'] = row["fromUser.displayName"]
                event_dict["org:resource"] = classification
                event_dict["time:timestamp"] = datetime_object
                event = pmlog.Event(event_dict)

                for conversation in open_conversations:
                    time_diff = (datetime_object - conversation.open_time).total_seconds() / 60.0
                    if time_diff > conversation_duration:
                        log.append(conversation.add_to_trace())
                        conversation.write_to_txt()
                        open_conversations.remove(conversation)


                # Now we find our text body
                text_body = {}
                for word in text.split(" "):
                    if word in tf_idf:
                        text_body[word] = tf_idf[word]



                mention = TP.get_mentions(text)
                mention_added = False
                if len(mention) > 0:
                    for conversation in open_conversations:
                        if conversation.is_person_in_conversation(mention[0][1:]):
                            conversation.add_message(text_body, event,
                                                message_text=row["text"], person=row["fromUser.username"])
                            mention_added = True
                            break

                elif not mention_added:
                    # Find the best matching conversation
                    score = 0
                    best_matching_conversation = None
                    for conversation in open_conversations:
                        conversation_score = conversation.similarity_score(text)
                        if conversation_score > score:
                            score = conversation_score
                            best_matching_conversation = conversation

                    if best_matching_conversation != None:
                        best_matching_conversation.add_message(text_body, event,
                                                            message_text=row["text"], person=row["fromUser.username"])

                    else:
                        convo = Conversation(text_body=text_body, open_time=datetime_object,
                                                event=event, message_text=row["text"], person=row["id"])
                        open_conversations.append(convo)

                print("Currently " + str(len(open_conversations)) + " open conversations")
            except AttributeError:
                continue
    return log
