# import Data_Processing_Transformation as DPT
import xml

import DataProcessing_02012020
import XML_Extraction
import Extraction
import Message_Classifier

import pandas as pd

import data_formatting
from Kaggle_Miner import mine_conversations
from Kaggle_TF import Kaggle_TF
from Kaggle_IDF import Kaggle_IDF
from datetime import datetime
from pm4py.objects.log.exporter.xes import factory as xes_exporter



if __name__ == '__main__':
    """
    # test_dataframe = get_test_dataframe('MSDialog')

    # save dataframes
    
    test_dataframe = get_test_dataframe('MSDialog')
    test_dataframe.to_pickle("test_dataframe")
    training_data = get_training_dataframe_xml("Ubuntu")
    training_data=get_training_dataframe_xml("NYC")
    training_data.to_pickle("Processed_Training_Data")'''

    # load dataframe from file
    test_dataframe = pd.read_pickle("Processed_MSDialog")
    training_dataframe = pd.read_pickle("Processed_Training_Data")

    # Inistantiate Classifier
    classification = Message_Classifier.MessageClassifier()

    # load training data to class to able to train and evaluate models' performance
    classification.load_data_for_training(training_dataframe)

    # save fitted models in directory
    # classification.save_models()

    # load previous trained models from directory
    classification.load_models()

    # Evaluate to pick best classifier
    classification.evaluate_models()

    #Predict data
    predicted class= classification.predict_class(test_dataframe,"MSDialog_rf)"""


    """ Hyper-parameters """
    # How long is a conversation active for? In minutes
    conversation_duration = 15.0  # Minutes

    # Pandas chunk size
    chunksize = 10 ** 3

    # Stop datetime
    stop_datetime = datetime.strptime("2020-03-30T00:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ')

    """ Runner Code """
    #idf = Kaggle_IDF(chunksize, stop_datetime, "Data/Kaggle/chatroom.csv")
    idf = Kaggle_IDF(chunksize, stop_datetime, "synthetic_data.csv")


    print("Done mining words")
    #log = mine_conversations(idf, "Data/Kaggle/chatroom.csv", stop_datetime, chunksize, conversation_duration)
    log = mine_conversations(idf, "synthetic_data.csv", stop_datetime, chunksize, conversation_duration)
    print("Done mining conversations")
    xes_exporter.export_log(log, "synthetic_data_processed.xes")

    '''# Inistantiate Classifier
    classification = Message_Classifier.MessageClassifier()

    training_data = data_formatting.format_and_msdialog('MSDialog/Intent/train_features.tsv')
    #print(training_data)
    test_data=data_formatting.format_and_msdialog('MSDialog/Intent/test_features.tsv')

    # load training data to class to able to train and evaluate models' performance
    classification.load_data_for_training(training_data)
    classification.load_data_for_test(test_data)

    # save fitted models in directory
    classification.save_models()'''

