import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class NLP:
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
        self.train_classifier()
        self.class_mappings =  {"Reject": "Answer",
                       "Statement": "Question",
                       "nAnswer" : "Answer",
                       "Accept" : "Answer",
                       "Emotion": "Other",
                       "Continuer": "Clarification",
                       "Clarify": "Clarification",
                       "ynQuestion": "Question",
                       "whQuestion": "Question",
                       "Other": "Other",
                       "Emphasis": "Clarification",
                       "System": "Other",
                       "Greet": "Other",
                       "yAnswer": "Answer",
                       "Bye": "Other"}

    def train_classifier(self):
        print(len(nltk.corpus.nps_chat.xml_posts()))
        posts = nltk.corpus.nps_chat.xml_posts()


        featuresets = [(self.dialogue_act_features(post.text), post.get('class')) for post in posts]
        print(featuresets)
        self.classifier = nltk.NaiveBayesClassifier.train(featuresets)



    def set_sentence(self,text):
        self.sentence=text

    def tokenize(self):
        return nltk.word_tokenize(self.sentence)

    def get_class(self):
        #uses trained classifier to classify text

        new_featureset=self.dialogue_act_features(self.sentence)
        classification = str(self.classifier.classify(new_featureset))
        return self.class_mappings[classification]

    def dialogue_act_features(self,post):
        features = {}
        for word in nltk.word_tokenize(post):
            features['contains({})'.format(word.lower())] = True
        return features



    def get_tags(self):
        #Position tagging: https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b
        tokens=self.tokenize()

        return nltk.pos_tag(tokens)

    def sentiment(self, message):
        ss = self.sid.polarity_scores(message)
        if ss["compound"] == 0.0:
            return "neutral"
        elif ss["compound"] > 0.0:
            return "positive"
        else:
            return "negative"
