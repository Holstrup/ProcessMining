import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

#nltk.download('stopwords')
def kaggle_extract():
    chunksize = 10 ** 3

    break_loop = False
    for chunk in pd.read_csv("Kaggle/chatroom.csv", chunksize=chunksize, usecols=["text"]):
        if break_loop:
            break

        for index, row in chunk.iterrows():
            try:
                text = (row["text"]).strip()
                text = text.lower()
                text = rm_mentions(text)
                text = rm_code(text)
                text = rm_punctuation(text)
                text = remove_stop_words(text)

            except AttributeError:
                continue


            break
    return text


def rm_mentions(data):
    return " ".join(filter(lambda x: x[0] != '@', data.split()))

def rm_code(data):
    return re.sub("(```.+?```)", "", data)

def rm_punctuation(data):
    return re.sub(r'[^\w\s]', '', data)


def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

kaggle_extract()
