import nltk

class NLP:

    def __init__(self):
        posts = nltk.corpus.nps_chat.xml_posts()[:10000]
        for post in posts:

            print (post.text)
    def set_sentence(self,text):
        self.sentence=text

    def tokenize(self):
        return nltk.word_tokenize(self.sentence)

    #def is_question(self):
