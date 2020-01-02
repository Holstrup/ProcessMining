import nltk

class NLP:
    def __init__(self):
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
        posts = nltk.corpus.nps_chat.xml_posts()
        featuresets = [(self.dialogue_act_features(post.text), post.get('class')) for post in posts]
        self.classifier = nltk.NaiveBayesClassifier.train(featuresets)

    def set_sentence(self,text):
        self.sentence=text

    def tokenize(self):
        return nltk.word_tokenize(self.sentence)

    def get_class(self):
        new_featureset=self.dialogue_act_features(self.sentence)
        classification = str(self.classifier.classify(new_featureset))
        return self.class_mappings[classification]

    def dialogue_act_features(self,post):
        features = {}
        for word in nltk.word_tokenize(post):
            features['contains({})'.format(word.lower())] = True
        return features

    def get_tags(self):
        tokens=self.tokenize()
        return nltk.pos_tag(tokens)
