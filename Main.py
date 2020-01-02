#import Data_Processing_Transformation as DPT
import xml

import DataProcessing_02012020
import XML_Extraction
from os import listdir
from os.path import isfile, join


if __name__ == '__main__':
    #DPT.Data_Processing_Transformation("Kaggle/chatroom.csv", social_graph=True, kaggle=True)
    data_process = DataProcessing_02012020.DataProcessing()

    all_files = [f for f in listdir('Ubuntu') if isfile(join('Ubuntu', f))]

    for file in all_files:
        try:
            file = XML_Extraction.Extract_Data_to_log('Ubuntu/' + file)
            dict = file.create_dict()

            data_process.load_dict(dict)
            data_process.append_dataframe()
        except xml.etree.ElementTree.ParseError:
            print(file)


    rowData = data_process.get_dataframe().loc['2985_3', :]
    print(rowData)