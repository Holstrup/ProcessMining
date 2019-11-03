import nltk

class NLP:
    def __init__(self):
        self.train_classifier()

    def train_classifier(self):
        #trains a Naive bayes with known text from nltk data and classes like  ynQuestion, Statement,whQuestion
        #nltk.download('punkt')
        #nltk.download('nps_chat')


        posts = nltk.corpus.nps_chat.xml_posts()[:10000]

        featuresets = [(self.dialogue_act_features(post.text), post.get('class')) for post in posts]
        self.classifier = nltk.NaiveBayesClassifier.train(featuresets)



    def set_sentence(self,text):
        self.sentence=text

    def tokenize(self):
        return nltk.word_tokenize(self.sentence)

    def get_class(self):
        #uses trained classifier to classify text

        new_featureset=self.dialogue_act_features(self.sentence)
        return self.classifier.classify(new_featureset)

    def dialogue_act_features(self,post):
        features = {}
        for word in nltk.word_tokenize(post):
            features['contains({})'.format(word.lower())] = True
        return features



    def get_tags(self):
        #Position tagging: https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b
        tokens=self.tokenize()

        return nltk.pos_tag(tokens)
