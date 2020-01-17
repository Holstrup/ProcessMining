from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re

def rm_mentions(data):
    return " ".join(filter(lambda x: x[0] != '@', data.split()))

def get_mentions(data):
    return list(filter(lambda x: x[0] == '@', data.split()))

def rm_code(data):
    return re.sub("(```.+?```)", "", data)

def rm_punctuation(data):
    return re.sub(r'[^\w\s]', '', data)

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    custom_stop_words = ["hello", "hi", "hey"]
    stop_words.extend(custom_stop_words)
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def stemming(data):
    ps = SnowballStemmer("english")
    new_text = []
    for word in data:
        new_text.append(ps.stem(word))
    return new_text

def preprocess_text(text):
    text = text.strip()
    text = text.lower()
    text = rm_mentions(text)
    text = rm_code(text)
    text = rm_punctuation(text)
    # text = rm_urls(text)
    text = remove_stop_words(text)
    text = text.split(" ")
    text = stemming(text)
    return text
