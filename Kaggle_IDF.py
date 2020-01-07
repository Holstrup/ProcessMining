from datetime import datetime
import pandas as pd
import Text_Preprocessing as TP
import math

def Kaggle_IDF(chunksize, stop_datetime, csv_file_path, idf_threshold = 0.0):
    datetime_object = datetime.strptime("2015-01-01T00:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ')
    break_loop = False
    term_unique_occurences = {}
    idf = {}
    document_count = 0
    for chunk in pd.read_csv(csv_file_path, chunksize=chunksize, usecols=["text", "sent"]):

        # Terminate after late date, to trim dataset.
        if datetime_object > stop_datetime:
            print(datetime_object)
            break_loop = True


        if break_loop:
            break

        for index, row in chunk.iterrows():
            try:
                document_count += 1
                text = TP.preprocess_text(row["text"])

                for word in set(text):
                    if word not in term_unique_occurences:
                        term_unique_occurences[word] = 0
                    term_unique_occurences[word] += 1

            except AttributeError:
                continue

        date_string = row["sent"]
        datetime_object = datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%fZ')

    del term_unique_occurences['']

    for word in term_unique_occurences:
        idf_value = math.log2(document_count / term_unique_occurences[word])
        if idf_value > idf_threshold:
            idf[word] = idf_value

    return idf