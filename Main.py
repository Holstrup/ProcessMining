# import Data_Processing_Transformation as DPT
import xml

import DataProcessing_02012020
import XML_Extraction
import Extraction
import Message_Classifier
from os import listdir
from os.path import isfile, join
import pandas as pd


def get_training_dataframe_xml(folder_name):
    data_process = DataProcessing_02012020.DataProcessing(has_class=True)
    # load nyc data to dataframe

    all_files = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]

    for file in all_files:
        try:
            file = XML_Extraction.ExtractDataLog(folder_name + '/' + file)
            dict = file.create_dict()

            data_process.load_dict(dict)
            data_process.append_dataframe()
        except xml.etree.ElementTree.ParseError:
            print("Cannot import " + file)

    dataframe = data_process.get_clean_dataframe()

    return dataframe


def get_test_dataframe(file):
    data_process = DataProcessing_02012020.DataProcessing()

    data = Extraction.Extract(file).get_all_threads_list()
    total = len(data)
    current = 0
    for dict in data:
        current += 1
        data_process.load_dict(dict)
        try:
            data_process.append_dataframe()
        except KeyError:
            print(dict)
            continue
        print("{0} of {1}".format(current, total))
    dataframe = data_process.get_clean_dataframe()
    return dataframe


    # Stop datetime
    stop_datetime = datetime.strptime("2015-01-15T00:00:00.000Z", '%Y-%m-%dT%H:%M:%S.%fZ')

if __name__ == '__main__':
    # test_dataframe = get_test_dataframe('MSDialog')

    # save dataframes
    '''
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
    classification.predict_class(test_dataframe)
