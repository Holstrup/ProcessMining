import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

from nltk.corpus import opinion_lexicon

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import XML_Extraction


class DataProcessing:
    has_class: bool

    def __init__(self, has_class=False):
        self.has_class = has_class
        if has_class:
            self.df = pd.DataFrame(columns=['IUS', 'DS', '?', 'D', 'What',
                                            'Where', 'When', 'Why', 'Who', 'How',
                                            'AP', 'NP', 'UL', 'ULU', 'ULSU', 'IS',
                                            'T', '!', 'FB', 'SS(Neu)', 'SS(Pos)', 'SS(Neg)', 'Lex(Pos)', 'Lex(Neg)',
                                            'Class'])
        else:
            self.df = pd.DataFrame(columns=['IUS', 'DS', '?', 'D', 'What',
                                            'Where', 'When', 'Why', 'Who', 'How',
                                            'AP', 'NP', 'UL', 'ULU', 'ULSU', 'IS',
                                            'T', '!', 'FB', 'SS(Neu)', 'SS(Pos)', 'SS(Neg)', 'Lex(Pos)', 'Lex(Neg)'])
        self.sentiment_analyser = SentimentIntensityAnalyzer()

    def load_dict(self, dict):

        """Dict has following structure:
        {'Thread ID': Str, Posts: {abs_pos(int): {'User id': Str, 'Date': Str, 'Content': Str, 'Class': Str}}
                    """
        self.datadict = dict

    def append_dataframe(self):

        thread_id = self.datadict["Thread ID"]
        initial_post = self.datadict["Posts"][1]['Content']
        initial_author = self.datadict["Posts"][1]['User ID']
        entire_thread = self.get_dialogue()
        num_posts = len(self.datadict["Posts"])

        for abs_pos in self.datadict["Posts"]:
            # instantiate row for insertion
            #If training data include attribute 'Class' in dict
            if self.has_class:
                row_dict = {'IUS': 0, 'DS': 0, '?': 0, 'D': 0, 'What': 0, 'Where': 0,
                            'When': 0, 'Why': 0, 'Who': 0, 'How': 0, 'AP': 0,
                            'NP': 0, 'UL': 0, 'ULU': 0, 'ULSU': 0, 'IS': 0, 'T': 0, '!': 0,
                            'FB': 0, 'SS(Neu)': 0, 'SS(Pos)': 0, 'SS(Neg)': 0, 'Lex(Pos)': 0, 'Lex(Neg)': 0, 'Class': 0}
            else:
                row_dict = {'IUS': 0, 'DS': 0, '?': 0, 'D': 0, 'What': 0, 'Where': 0,
                        'When': 0, 'Why': 0, 'Who': 0, 'How': 0, 'AP': 0,
                        'NP': 0, 'UL': 0, 'ULU': 0, 'ULSU': 0, 'IS': 0, 'T': 0, '!': 0,
                        'FB': 0, 'SS(Neu)': 0, 'SS(Pos)': 0, 'SS(Neg)': 0, 'Lex(Pos)': 0, 'Lex(Neg)': 0}

            abs_pos = abs_pos

            post_id = str(thread_id) + "_" + str(abs_pos)

            # Initial Utterance similarity
            content = self.datadict["Posts"][abs_pos]['Content']

            # Throws error when text is ***. look at line 142
            try:
                row_dict['IUS'] = round(self.get_cosine_sim(initial_post, content)[0, 1], 4)
            except IndexError:
                row_dict['IUS'] = 0

            # Dialogue sim
            row_dict['DS'] = round(self.get_cosine_sim(content, entire_thread)[0, 1], 4)

            # Question mark
            row_dict['?'] = '?' in content

            # Duplicate
            row_dict['D'] = 'same' in content.lower() or 'similar' in content.lower()

            # 5W1H
            row_dict['What'] = 'what' in content.lower()
            row_dict['Where'] = 'where' in content.lower()
            row_dict['When'] = 'when' in content.lower()
            row_dict['Why'] = 'why' in content.lower()
            row_dict['Who'] = 'who' in content.lower()
            row_dict['How'] = 'how' in content.lower()

            # AP
            row_dict['AP'] = abs_pos

            # NP
            row_dict['NP'] = abs_pos / num_posts

            # UL
            filtered_content = self.remove_stopwords(content.lower())
            row_dict['UL'] = len(filtered_content)

            # 'ULU'
            unique_filtered_content = set(filtered_content)
            row_dict['ULU'] = len(unique_filtered_content)

            # ULSU
            row_dict['ULSU'] = self.get_unique_count_stemming(unique_filtered_content)

            # IS
            row_dict['IS'] = initial_author == self.datadict["Posts"][abs_pos]['User ID']

            # T
            row_dict['T'] = "thank" in content.lower() or "thanks" in content.lower()

            # !
            row_dict['!'] = "!" in content

            # FB
            row_dict['FB'] = "did not" in content.lower() or "does not" in content.lower()

            # Sentiment scores
            neu_score, pos_score, neg_score = self.get_sentiment_analyzer_scores(content)

            row_dict['SS(Neu)'] = neu_score
            row_dict['SS(Pos)'] = pos_score
            row_dict['SS(Neg)'] = neg_score

            # Lacking Lex(Pos)', 'Lex(Neg)'
            row_dict['Lex(Pos)'] = self.count_words(content,positive=True)

            row_dict['Lex(Neg)'] = self.count_words(content)

            if self.has_class:
                # Class
                row_dict['Class'] = self.datadict["Posts"][abs_pos]['Class']

            # Create dataframe from dict
            new_row = pd.DataFrame(row_dict, index=[post_id])

            self.df = self.df.append(new_row, sort=False)

    def get_cosine_sim(self, *strs):
        vectors = [t for t in self.get_vectors(*strs)]

        return cosine_similarity(vectors)

    def get_vectors(self, *strs):
        text = [t for t in strs]

        vectorizer = CountVectorizer(text)
        try:

            vectorizer.fit(text)
        # Throws error when text = ['****','*****']
        except ValueError:
            text = ['Hello World']
            vectorizer = CountVectorizer(text)
            vectorizer.fit(text)

        return vectorizer.transform(text).toarray()

    def get_dialogue(self):
        dialogue = ""
        for abs_pos in self.datadict["Posts"]:
            dialogue = dialogue + ". " + self.datadict["Posts"][abs_pos]['Content']

        return dialogue

    def remove_stopwords(self, content):

        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(content)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]

        return filtered_sentence

    def get_unique_count_stemming(self, list_content):

        ps = PorterStemmer()
        stemmed = [ps.stem(x) for x in list_content]

        return len(set(stemmed))

    def get_sentiment_analyzer_scores(self, sentence):

        score = self.sentiment_analyser.polarity_scores(sentence)

        return score['neu'], score['pos'], score['neg']

    def get_clean_dataframe(self):
        df = self.df.dropna()

        return df

    def count_words(self, sentence,positive=False):
        if positive:
            lex=set(opinion_lexicon.positive())


        else:
            lex = set(opinion_lexicon.negative())

        numb_acc = 0
        for word in word_tokenize(sentence):
            numb_acc += word in lex
        return numb_acc




