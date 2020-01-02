from datetime import datetime
import pandas as pd
import Text_Preprocessing as TP

def Kaggle_TF(chunksize, stop_datetime, csv_file_path):
    datetime_object = datetime.strptime("2015-01-01T00:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ')
    break_loop = False
    word_dict = {}
    for chunk in pd.read_csv(csv_file_path, chunksize=chunksize, usecols=["text", "sent"]):

        # Terminate after late date, to trim dataset.
        if datetime_object > stop_datetime:
            print(datetime_object)
            break_loop = True


        if break_loop:
            break_loop = True
            break

        for index, row in chunk.iterrows():
            try:
                text = TP.preprocess_text(row["text"])

                for word in text:
                    if word not in word_dict:
                        word_dict[word] = 0
                    word_dict[word] += 1

            except AttributeError:
                continue

        date_string = row["sent"]
        datetime_object = datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%fZ')
    del word_dict['']
    return word_dict