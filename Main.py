import Data_Processing_Transformation
from Natural_Language_Processing import  NLP

if __name__ == '__main__':

    #Data_Processing_Transformation.Data_Processing_Transformation("Test")
    nlp=NLP()
    nlp.set_sentence("Hello. How are you doing?")
    nlp.chance_of_question_sentence()

