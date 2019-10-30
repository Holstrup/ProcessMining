import nltk

class NLP:

    def set_sentence(self,text):
        self.sentence=text

    def tokenize(self):
        return nltk.word_tokenize(self.sentence)

    def is_question(self):
        '''Here goes logic'''
        if "?" in self.tokenize():
            return True
        else:
            return False

    def get_tags(self):
        tokens=self.tokenize()

        print(nltk.pos_tag(tokens))
