import xml

import DataProcessing_02012020
import XML_Extraction
from os import listdir
from os.path import isfile, join

from Kaggle_Miner import mine_conversations
from Kaggle_TF import Kaggle_TF
from Kaggle_IDF import Kaggle_IDF
from datetime import datetime
from pm4py.objects.log.exporter.xes import factory as xes_exporter


if __name__ == '__main__':
     #nltk.download('stopwords')



    """ Hyper-parameters """
    # How long is a conversation active for? In minutes
    conversation_duration = 15.0  # Minutes

    # Pandas chunk size
    chunksize = 10 ** 3

    # Stop datetime
    stop_datetime = datetime.strptime("2015-02-01T00:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ')


    """ Runner Code """
    idf = Kaggle_IDF(chunksize, stop_datetime, "Data/Kaggle/chatroom.csv")

    #word_dict = Kaggle_TF(chunksize, stop_datetime, "Data/Kaggle/chatroom.csv")
    print("Done mining words")
    log = mine_conversations(idf, "Data/Kaggle/chatroom.csv", stop_datetime, chunksize, conversation_duration)
    print("Done mining conversations")
    xes_exporter.export_log(log, "Mined_Conversations_Kaggle.xes")
    
    
    '''data_process = DataProcessing_02012020.DataProcessing()

    all_files = [f for f in listdir('Ubuntu') if isfile(join('Ubuntu', f))]

    for file in all_files:
        try:
            file = XML_Extraction.Extract_Data_to_log('Ubuntu/' + file)
            dict = file.create_dict()

            data_process.load_dict(dict)
            data_process.append_dataframe()
        except xml.etree.ElementTree.ParseError:
            print(file)


    
    print(data_process.get_dataframe())'''
