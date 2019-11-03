import Data_Processing_Transformation as DPT
from Natural_Language_Processing import NLP


if __name__ == '__main__':
    DPT.Data_Processing_Transformation("Data/General")
    l=NLP()
    l.set_sentence("is it correctly understood?")
    print(l.get_class())
    l.set_sentence("thanks!")
    print(l.get_class())