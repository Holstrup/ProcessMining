import pandas as pd
from datetime import datetime
from NLP_Classification import NLP
from Conversation import Conversation
from pm4py.objects.log.util.log import log as pmlog
import Text_Preprocessing as TP


""" Main Functions """


def mine_conversations(idf, csv_file_path, stop_datetime, chunksize, conversation_duration):
    datetime_object = datetime.strptime("2015-01-01T00:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ')
    break_loop = False
    open_conversations = []
    log = pmlog.EventLog()
    score_stats = []
    for chunk in pd.read_csv(csv_file_path, chunksize=chunksize, usecols = ["id", "fromUser.displayName", "text", "sent", "fromUser.username"]):
        # Terminate after late date, to trim dataset.
        if datetime_object > stop_datetime:
            break_loop = True


        if break_loop:
            break_loop = True
            break

        for index, row in chunk.iterrows():
            #print("Currently " + str(len(open_conversations)) + " open conversations")
            try:
                # Start by getting the variables we need
                text = row["text"]

                if str(text) == "nan":
                    continue

                if len(text.split(" ")) <= 7: # Filter out messages with less than n wordsw
                    continue

                datetime_object = datetime.strptime(row["sent"], '%Y-%m-%dT%H:%M:%S.%fZ')


                event_dict = {}
                event_dict["User id"] = row["fromUser.username"]
                event_dict["Date"] = datetime_object
                event_dict["Content"] = text
                event_dict["Class"] = None

                #event_dict['concept:name'] = row["fromUser.displayName"]
                #event_dict["time:timestamp"] = datetime_object
                # event = pmlog.Event(event_dict)



                for conversation in open_conversations:
                    time_diff = (datetime_object - conversation.open_time).total_seconds() / 60.0
                    if time_diff > conversation_duration:
                        if len(conversation.message_texts) > 1:
                            log.append(conversation.add_to_trace())
                            conversation.write_to_txt()
                        open_conversations.remove(conversation)


                # Now we find our text body
                tf_idf_message = {}
                text_list = TP.preprocess_text(text)
                for word in set(text_list):
                    if word in idf:
                        tf_idf_message[word] = (text_list.count(word) / len(text_list)) * idf[word]


                mention = TP.get_mentions(text)
                if len(mention) > 0:
                    for conversation in open_conversations:
                        if conversation.is_person_in_conversation(mention[0][1:]):
                            conversation.add_message(event_dict, message_text=row["text"], person=row["fromUser.username"],
                                                     idf=idf)

                else:
                    # Find the best matching conversation
                    score = 0
                    best_matching_conversation = None
                    for conversation in open_conversations:
                        conversation_score = conversation.similarity_score(tf_idf_message)
                        if conversation_score > score:
                            score = conversation_score
                            best_matching_conversation = conversation

                    if best_matching_conversation != None and score > 0.05:
                        score_stats.append(score)
                        best_matching_conversation.add_message(event_dict, message_text=row["text"],
                                                               person=row["fromUser.username"],
                                                               idf=idf)

                    else:
                        convo = Conversation(open_time=datetime_object,
                                                event_dict=event_dict, message_text=row["text"], person=row["id"], idf=idf)
                        open_conversations.append(convo)

            except AttributeError as e:
                print(e)
                continue

        print(datetime_object)
    print(score_stats)
    print(sum(score_stats) / len(score_stats))
    print(min(score_stats))
    print(max(score_stats))
    return log
