from django.db import models
from dataServices.db import DbConn
from constance import config
from .singleton import *

from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Merge
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from gensim.models import Word2Vec
import json
import keras.backend as K
from keras.models import Model
import re
import nltk
import itertools
from keras.preprocessing.sequence import pad_sequences

from keras.backend import clear_session

import pickle
import pandas as pd
import math
from collections import Counter

WORD = re.compile(r'\w+')
class BugDupsOldModel(Singleton):
    clusterId = None
    ignoreSubsequent = True
    WORD = re.compile(r'\w+')

    def __init__(self, cluster_id=None, project=None, product=None, comp=None):

        self.stops = set(stopwords.words('english'))
        self.max_seq_length = 150
        self.text_cols = ['wrdEmbSet', 'dup_wrdEmbSet']
        self.embedding_dim = 150
        self.cut_off = 90

        n_hidden = 50
        gradient_clipping_norm = 1.25
        batch_size = 64
        n_epoch = 1 #25


        self.clusterId = cluster_id
        print(cluster_id)
#         self.projectData = self.load_data(collection, project, product, comp, True)
        #print(self.projectData)
        #print('0')

        clear_session()

        filename = config.W2V_MODEL_PREFIX + str(cluster_id) + '.bin'
        self.w2vmodel = Word2Vec.load(filename)


        f = config.VOCAB_MODEL_PREFIX + str(cluster_id) + '.json'
        self.vocabulary = json.load(open(f, 'r'))
        thefile = config.INV_VOCAB_MODEL_PREFIX + str(cluster_id) + '.json'
        with open (thefile, 'rb') as fp:
            self.inverse_vocabulary = pickle.load(fp)

        self.words = list(self.w2vmodel.wv.vocab)
        thefile = config.EMEDBED_MODEL_PREFIX + str(cluster_id) + '.json'
        print(thefile)
        with open (thefile, 'rb') as fp:
            self.embeddings = pickle.load(fp)
        #print(self.embeddings)
        #print(len(self.embeddings))

        left_input = Input(shape=(self.max_seq_length,), dtype='int32')
        #print(left_input)
        right_input = Input(shape=(self.max_seq_length,), dtype='int32')
        embedding_layer = Embedding(len(self.embeddings), self.embedding_dim, weights = [self.embeddings], input_length=self.max_seq_length, trainable=False)
        #print(embedding_layer)
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)
        shared_lstm = LSTM(n_hidden)
        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)
        malstm_distance = Merge(mode=lambda x: self.exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
        self.malstm = Model([left_input, right_input], [malstm_distance])

        optimizer = Adadelta(clipnorm=gradient_clipping_norm)
        self.malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        # load weights into new model
        filename = config.TXT_MODEL_PREFIX + str(cluster_id) + '.h5'
        self.malstm.load_weights(filename)

    def text_to_word_list(self, text):
        text = str(text)
        text = text.lower()
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", "  ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", "  ", text)
        text = re.sub(r"\+", "  ", text)
        text = re.sub(r"\-", "  ", text)
        text = re.sub(r"\=", "  ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r">", "  ", text)
        text = re.sub(r"<", "  ", text)
        text = re.sub(r"--", "  ", text)
        text = re.sub(r"`", "  ", text)
        #text = re.sub(r"...", "  ", text)
        text = re.sub(r"#", "  ", text)
        text = re.sub(r"@", "  ", text)
        text = re.sub(r"$", "  ", text)
        text = re.sub(r"%", "  ", text)
        text = re.sub(r"&", "  ", text)
        text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
        text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)
        #text = re.sub(r"*", "  ", text)
        #text = re.sub(r"[", "  ", text)
        #text = re.sub(r"]", "  ", text)
        #text = re.sub(r"(", "  ", text)
        #text = re.sub(r")", "  ", text)
        text = re.sub(r"_", "  ", text)
        text = re.sub(r"\\", "  ", text)
        text = re.sub(r"~", "  ", text)
        #print(text)
        text = re.sub(r"(\d+)(k)", r"", text)
        text = re.sub(r":", "  ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        #text = text.split()
        text_list = []
        for word in nltk.word_tokenize(text):
            case = self.get_word(word)
            if case:
                text_list.append(word)
        return text_list

    def get_word(self, word):
        chars = ["Object","?", "'s", ">", "<", ",", ":", "'", "''", "--", "`", "``", "...", "", "!", "#", '"', '$', '%', '&', '(', ')', '*', '+', '-', '.', '/', ';', '=', '@', '[', '\\', ']', '^', '_', '{', '}', '|', '~', '\t', '\n', '', 'test', 'case', 'id', 'short', 'description', 'and', 'on', 'if', 'the', 'you', 'of', 'is', 'which', 'what', 'this', 'why', 'during', 'at', 'are', 'to', 'in', 'with', 'for', 'cc', 'email', 'from', 'subject', 'a', 'that', 'yet', 'so', 'raise', 'or', 'then', 'there', 're', 'thanks', 'i', 'as', 'me', 'am', 'attaching', 'thread', 'file', 'along', 'files', 'was', 'it', 'n', 'do', 'does', 'well', 'com']

        if word not in chars and '*' not in word and '=' not in word and '++' not in word and '___' not in word and (not word.isdigit()):
            return True
        return False

    def exponent_neg_manhattan_distance(self, left, right):
            return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

    def build_test_data_text(self, sentence, w2vmodel, words, vocabulary, inverse_vocabulary, cluster_id, stops):
        q1=[0]
        a = -1
        s = 0
        q2n = []
        for word in self.text_to_word_list(sentence):
            # Check for unwanted words
            if word.lower() in stops and word.lower() not in words: #word2vec.vocab:
                continue
            if word not in vocabulary:
                s = 1
            else:
                q2n.append(vocabulary[word])
        q1[a] = q2n
        return q1

    def get_test_data(self, df, test_df, w2vmodel, words, vocabulary, inverse_vocabulary, cluster_id, stops):
        c = pd.DataFrame()
        new_feature_columns = ['DUPLICATE_OF', 'DUP_Headline', 'DUP_ENCL-Description', 'DUP_PROJECT', 'DUP_PRODUCT', 'DUP_COMPONENT', 'DUP_wrdEmbSet', 'is_duplicate']
        feature_columns_to_use = ['IDENTIFIER', 'Headline', 'ENCL-Description', 'PROJECT', 'PRODUCT', 'COMPONENT', 'wrdEmbSet']
        for i in range(0, len(feature_columns_to_use)):
            c[new_feature_columns[i]] = df[feature_columns_to_use[i]]
        for i in range(0, len(feature_columns_to_use) - 2):
            c[feature_columns_to_use[i + 1]] = test_df[feature_columns_to_use[i + 1]]
        sentence = test_df["Headline"] + " " + test_df["ENCL-Description"]
        #print(sentence)
        wrdEmbSet_int = self.build_test_data_text(sentence, w2vmodel, words, vocabulary, inverse_vocabulary, cluster_id, stops)
        wrdEmbSet = wrdEmbSet_int*df.shape[0]
        #print(len(wrdEmbSet_int[0]))
        c['wrdEmbSet'] = wrdEmbSet
        return c


class BugDupsOld(models.Model):
    Identifier = models.CharField(max_length=50)
    Headline = models.CharField(max_length=200)
    Probability = models.DecimalField(max_digits=5, decimal_places=2)
    Product = models.CharField(max_length=200)
    Project = models.CharField(max_length=200)
    Component = models.CharField(max_length=200)
    
    @staticmethod
    def load_data( collection,  project, product ):
        print(collection)
        cursor = collection.find({"PROJECT":project,"PRODUCT":product},{ 'IDENTIFIER':1, 'Headline':1, 'ENCL-Description':1, 'PRODUCT':1, 'COMPONENT':1, 'wrdEmbSet':1, "PROJECT":1 })
        df =  pd.DataFrame(list(cursor))
        print(df.columns)
#         df["SUBMITTED_DATE"] = pd.to_datetime(df['SUBMITTED_DATE'], unit='ms')
        return df

    @staticmethod
    def get_cluster_id(project):
        db = DbConn()
        collection = db.dbConn[config.ProjectCluster]
        cursor = collection.find({})
        clusters =  pd.DataFrame(list(cursor))
        project_clusters = []
        groups = clusters.groupby('Cluster')
        for name, group in groups:
            project_clusters.append(list(group['Project']))
        print(project_clusters)
        for i in range(0, len(project_clusters)):
            if(project in project_clusters[i]):
                cluster_id = i + 1
                return cluster_id
        return -1

    @staticmethod
    def test_model_text(malstm, embeddings, vocabulary, model, words, test_df, cluster, stops, max_seq_length, text_cols):
        #del model
        X_test = {'left': test_df.wrdEmbSet, 'right': test_df.DUP_wrdEmbSet}
        # Convert labels to their numpy representations
        # Zero padding
        for dataset, side in itertools.product([X_test], ['left', 'right']):
            dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)
        # Model variables
        #print("Loaded model from disk")
        # evaluate loaded model on test data
        #print(X_test['left'], X_test['right'])
        predictions = malstm.predict([X_test['left'], X_test['right']])
        return predictions

    @staticmethod
    def check_duplicates( bugData ):

        print("In check_duplicates ")
        dupList = []
        cluster_id = BugDups.get_cluster_id(bugData["PROJECT"])
        print("Cluster ID is")
        print(cluster_id)
        #print(bugData)
        if cluster_id < 0:
            return dupList
        bugDupInst = BugDupsOldModel.getInstance(cluster_id, bugData["PROJECT"], bugData["PRODUCT"], bugData["COMPONENT"])
        print("Got BugDupInst")
        #print(bugDupInst.clusterId)

        if bugDupInst and bugDupInst.clusterId != cluster_id:
            bugDupInst._forgetClassInstanceReferenceForTesting()
            bugDupInst = BugDupsOldModel.getInstance(cluster_id, bugData["PROJECT"], bugData["PRODUCT"], bugData["COMPONENT"])
            #bugDupInst = BugDups.getInstance()
        else:
            bugDupInst = BugDupsOldModel.getInstance()

        coll_name = config.BugDupsTrain_Prefix + str(cluster_id) + "_complete"
        print(coll_name)
        db = DbConn()
        collection = db.dbConn[coll_name]

        df = BugDups.load_data(collection, bugData["PROJECT"], bugData["PRODUCT"])
#         bugData = df.iloc[1]
        #print(df[df['IDENTIFIER']=='CSCtc56807'])
        print(df.shape)
#         df = df[(df['PRODUCT'] == bugData['PRODUCT']) & (df['PROJECT'] == bugData['PROJECT'])]
        #print(df.shape)
        #print(df[df['IDENTIFIER']=='CSCuq51291'])

        df1 = bugDupInst.get_test_data(df, bugData, bugDupInst.w2vmodel, bugDupInst.words, bugDupInst.vocabulary, bugDupInst.inverse_vocabulary, bugDupInst.clusterId, bugDupInst.stops)
#         print(df1)
        text_predictions = BugDups.test_model_text(bugDupInst.malstm, bugDupInst.embeddings, bugDupInst.vocabulary, bugDupInst.w2vmodel, bugDupInst.words, df1, bugDupInst.clusterId, bugDupInst.stops, bugDupInst.max_seq_length, bugDupInst.text_cols)
        #print(text_predictions)
        result = pd.DataFrame()
        result['DUPLICATE_OF'] = df1['DUPLICATE_OF']
        result['Headline'] = df1['DUP_Headline']
        result['Product'] = df1['DUP_PRODUCT']
        result['Project'] = df1['DUP_PROJECT']
        result['Component'] = df1['DUP_COMPONENT']

        p = []
        for i in text_predictions:
            p.append(i[0]*100)

        result['pred_text'] = p
        result = result.drop_duplicates(subset='DUPLICATE_OF', keep="last")
        print(result[result['DUPLICATE_OF']=='CSCzv37475'])
        print(result[result['DUPLICATE_OF']=='CSCzv36398'])
        result = result.sort_values(['pred_text'], ascending=[0])
        result = result[result['pred_text'] > bugDupInst.cut_off]
        if(result.shape[0] > 5):
            v = 5
        else:
            v = result.shape[0]
        print(result)
        if(v != 0):
            result = result.iloc[:v]
            for index, row in result.iterrows():
                temp = dict()
                temp["Identifier"] = row["DUPLICATE_OF"]
                temp["Probability"] = row["pred_text"]
                temp["Headline"] = row["Headline"]
                temp['Product'] = row['Product']
                temp['Project'] = row['Project']
                temp['Component'] = row['Component']
                dup = BugDupsOld(**temp)
                dupList.append(dup)

        return dupList

class ViewSetDuplicatesOld(models.Model):
    Identifier = models.CharField(max_length=50)
    Headline = models.CharField(max_length=200)
    Probability = models.DecimalField(decimal_places=2,max_digits=5)
    DuplicateBug = models.CharField(max_length=50)
    ActualDuplicate = models.CharField(max_length=50)

    @staticmethod
    def get_viewset_bug_dups(queryID, viewID=0, bu=None):
        db = DbConn()

        print("In BugDups::get_viewset_bug_dups")
        print(viewID)
        print(queryID)
        print(bu)
        tablePrefix = config.BugDupViewPrefix
        collName = ""
        if viewID == 0:
            collName = tablePrefix + bu + "_" + queryID + "_results"
        else:
            collName = tablePrefix + viewID + "_" + queryID + "_results"

        dups = []

        print(collName)
        resColl = db.dbConn[collName]
        temp = resColl.find().sort([("last_run_date",-1)]).limit(1)
        lastRun = temp[0]['last_run_date']
#         lastRun = lastRun.replace(hour=0, minute=0, second=0, microsecond=0)
        print(lastRun)
        query = []
        match_dict = {"$match":{"$and" :[{"last_run_date": {"$gte": lastRun}}]}}
        project_dict =  {"$project":{"_id":0, "Identifier": "$IDENTIFIER","Headline":"$Headline","DuplicateBug":"$DUPLICATE_LIST","Probability":"$PROBABILITIES","ActualDuplicate":"$actual_duplicate"}}
        query = [ match_dict, project_dict ]
        print(query)
        for res in resColl.aggregate(pipeline=query, allowDiskUse=True):
#             temp = {}
#             temp["Identifier"] = res["Identifier"]
#             temp["Identifier"] = res["Headline"]
#             temp["DuplicateBug"] = res["Headline"]
#
#
            b = ViewSetDuplicatesOld(**res)
            dups.append(b)

        return dups

class BugDupsModel(Singleton):
    clusterId = None
    ignoreSubsequent = True

    def __init__(self, cluster_id=None, project=None, product=None, comp=None):

        self.stops = set(stopwords.words('english'))
        self.max_seq_length = 150
        self.text_cols = ['wrdEmbSet', 'dup_wrdEmbSet']
        self.embedding_dim = 150
        self.cut_off = 90

        n_hidden = 50
        gradient_clipping_norm = 1.25
        batch_size = 64
        n_epoch = 1 #25


        self.clusterId = cluster_id
        print(cluster_id)
#         self.projectData = self.load_data(collection, project, product, comp, True)
        #print(self.projectData)
        #print('0')

        clear_session()

        filename = config.W2V_MODEL_PREFIX + str(cluster_id) + '.bin'
        self.w2vmodel = Word2Vec.load(filename)


        f = config.VOCAB_MODEL_PREFIX + str(cluster_id) + '.json'
        self.vocabulary = json.load(open(f, 'r'))
        thefile = config.INV_VOCAB_MODEL_PREFIX + str(cluster_id) + '.json'
        with open (thefile, 'rb') as fp:
            self.inverse_vocabulary = pickle.load(fp)

        self.words = list(self.w2vmodel.wv.vocab)
        thefile = config.EMEDBED_MODEL_PREFIX + str(cluster_id) + '.json'
        print(thefile)
        with open (thefile, 'rb') as fp:
            self.embeddings = pickle.load(fp)
        #print(self.embeddings)
        #print(len(self.embeddings))

        left_input = Input(shape=(self.max_seq_length,), dtype='int32')
        #print(left_input)
        right_input = Input(shape=(self.max_seq_length,), dtype='int32')
        embedding_layer = Embedding(len(self.embeddings), self.embedding_dim, weights = [self.embeddings], input_length=self.max_seq_length, trainable=False)
        #print(embedding_layer)
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)
        shared_lstm = LSTM(n_hidden)
        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)
        malstm_distance = Merge(mode=lambda x: self.exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
        self.malstm = Model([left_input, right_input], [malstm_distance])

        optimizer = Adadelta(clipnorm=gradient_clipping_norm)
        self.malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

        # load weights into new model
        filename = config.TXT_MODEL_PREFIX + str(cluster_id) + '.h5'
        self.malstm.load_weights(filename)

    def text_to_word_list(self, text):
        text = str(text)
        text = text.lower()
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", "  ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", "  ", text)
        text = re.sub(r"\+", "  ", text)
        text = re.sub(r"\-", "  ", text)
        text = re.sub(r"\=", "  ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r">", "  ", text)
        text = re.sub(r"<", "  ", text)
        text = re.sub(r"--", "  ", text)
        text = re.sub(r"`", "  ", text)
        #text = re.sub(r"...", "  ", text)
        text = re.sub(r"#", "  ", text)
        text = re.sub(r"@", "  ", text)
        text = re.sub(r"$", "  ", text)
        text = re.sub(r"%", "  ", text)
        text = re.sub(r"&", "  ", text)
        text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
        text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)
        #text = re.sub(r"*", "  ", text)
        #text = re.sub(r"[", "  ", text)
        #text = re.sub(r"]", "  ", text)
        #text = re.sub(r"(", "  ", text)
        #text = re.sub(r")", "  ", text)
        text = re.sub(r"_", "  ", text)
        text = re.sub(r"\\", "  ", text)
        text = re.sub(r"~", "  ", text)
        #print(text)
        text = re.sub(r"(\d+)(k)", r"", text)
        text = re.sub(r":", "  ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        #text = text.split()
        text_list = []
        for word in nltk.word_tokenize(text):
            case = self.get_word(word)
            if case:
                text_list.append(word)
        return text_list

    def get_word(self, word):
        chars = ["Object","?", "'s", ">", "<", ",", ":", "'", "''", "--", "`", "``", "...", "", "!", "#", '"', '$', '%', '&', '(', ')', '*', '+', '-', '.', '/', ';', '=', '@', '[', '\\', ']', '^', '_', '{', '}', '|', '~', '\t', '\n', '', 'test', 'case', 'id', 'short', 'description', 'and', 'on', 'if', 'the', 'you', 'of', 'is', 'which', 'what', 'this', 'why', 'during', 'at', 'are', 'to', 'in', 'with', 'for', 'cc', 'email', 'from', 'subject', 'a', 'that', 'yet', 'so', 'raise', 'or', 'then', 'there', 're', 'thanks', 'i', 'as', 'me', 'am', 'attaching', 'thread', 'file', 'along', 'files', 'was', 'it', 'n', 'do', 'does', 'well', 'com']

        if word not in chars and '*' not in word and '=' not in word and '++' not in word and '___' not in word and (not word.isdigit()):
            return True
        return False

    def exponent_neg_manhattan_distance(self, left, right):
            return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

    def build_test_data_text(self, sentence, w2vmodel, words, vocabulary, inverse_vocabulary, cluster_id, stops):
        q1=[0]
        a = -1
        s = 0
        q2n = []
        for word in self.text_to_word_list(sentence):
            # Check for unwanted words
            if word.lower() in stops and word.lower() not in words: #word2vec.vocab:
                continue
            if word not in vocabulary:
                s = 1
            else:
                q2n.append(vocabulary[word])
        q1[a] = q2n
        return q1

    def get_test_data(self, df, test_df, w2vmodel, words, vocabulary, inverse_vocabulary, cluster_id, stops):
        c = pd.DataFrame()
        #print(df)
        new_feature_columns = ['DUPLICATE_OF', 'DUP_Headline', 'DUP_ENCL-Description', 'DUP_PROJECT', 'DUP_PRODUCT', 'DUP_COMPONENT' ,'DUP_wrdEmbSet', 'is_duplicate']
        feature_columns_to_use = ['IDENTIFIER', 'Headline', 'ENCL-Description', 'PROJECT', 'PRODUCT', 'COMPONENT', 'wrdEmbSet']
        for i in range(0, len(feature_columns_to_use)):
            c[new_feature_columns[i]] = df[feature_columns_to_use[i]]
        for i in range(0, len(feature_columns_to_use) - 2):
            print(feature_columns_to_use[i+1])
            c[feature_columns_to_use[i + 1]] = test_df[feature_columns_to_use[i + 1]]
        sentence = test_df["Headline"] + " " + test_df["ENCL-Description"]
        #print(sentence)
        wrdEmbSet_int = self.build_test_data_text(sentence, w2vmodel, words, vocabulary, inverse_vocabulary, cluster_id, stops)
        wrdEmbSet = wrdEmbSet_int*df.shape[0]
        #print(len(wrdEmbSet_int[0]))
        c['wrdEmbSet'] = wrdEmbSet
        return c


class BugDups(models.Model):
    Identifier = models.CharField(max_length=50)
    Headline = models.CharField(max_length=200)
    Probability = models.DecimalField(max_digits=5, decimal_places=2)
    Product = models.CharField(max_length=200)
    Project = models.CharField(max_length=200)
    Component = models.CharField(max_length=200)
    Status = models.CharField(max_length=200)
    Similarity = models.CharField(max_length=200)
    SUBMITTED_DATE = models.CharField(max_length=200)

    @staticmethod
    def load_data_new(db, collection, cluster_id, project, product, component):
        comp_collection = db.dbConn['BugDupsTrainSet_components']
        cursor = comp_collection.find({'cluster_id': str(cluster_id)})
        d = list(cursor)[0]

        #original_comp = 'port'
        components = []
        components.append(component)
        if(component in d.keys()):
            for mapped_comp in set(d[component]):
                components.append(mapped_comp)
        '''
        for mapped_comp in set(d[component]):
            print(mapped_comp)
            components.append(mapped_comp)
            if mapped_comp in d.keys():
                for c in d[mapped_comp]:
                    components.append(c)
        '''
        print(collection)
        cursor = collection.find({"PROJECT":project,"PRODUCT":product, "COMPONENT":{"$in" :list(set(components))}},{ 'IDENTIFIER':1, 'Headline':1, 'ENCL-Description':1, 'PRODUCT':1, 'COMPONENT':1, 'wrdEmbSet':1, "PROJECT":1, "LIFECYCLE_STATE_CODE":1, "SUBMITTED_DATE":1 })
        df =  pd.DataFrame(list(cursor))
        print(df.columns)
        print(df.shape[0])
#         df["SUBMITTED_DATE"] = pd.to_datetime(df['SUBMITTED_DATE'], unit='ms')
        return df

    @staticmethod
    def get_cluster_id(project):
        db = DbConn()
        collection = db.dbConn[config.ProjectCluster]
        cursor = collection.find({})
        clusters =  pd.DataFrame(list(cursor))
        project_clusters = []
        groups = clusters.groupby('Cluster')
        for name, group in groups:
            project_clusters.append(list(group['Project']))
        print(project_clusters)
        for i in range(0, len(project_clusters)):
            if(project in project_clusters[i]):
                cluster_id = i + 1
                return cluster_id
        return -1

    WORD = re.compile(r'\w+')
    @staticmethod
    def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    @staticmethod
    def text_to_vector(text):
        words = WORD.findall(str(text))
        return Counter(words)

    @staticmethod
    def test_model_text(malstm, embeddings, vocabulary, model, words, test_df, cluster, stops, max_seq_length, text_cols):
        #del model
        X_test = {'left': test_df.wrdEmbSet, 'right': test_df.DUP_wrdEmbSet}
        # Convert labels to their numpy representations
        # Zero padding
        for dataset, side in itertools.product([X_test], ['left', 'right']):
            dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)
        # Model variables
        #print("Loaded model from disk")
        # evaluate loaded model on test data
        #print(X_test['left'], X_test['right'])
        predictions = malstm.predict([X_test['left'], X_test['right']])
        return predictions

    @staticmethod
    def check_duplicates( bugData ):

        print("In check_duplicates ")
        dupList = []
        cluster_id = BugDups.get_cluster_id(bugData["PROJECT"])
        print("Cluster ID is")
        print(cluster_id)
        #print(bugData)
        if cluster_id < 0:
            return dupList
        bugDupInst = BugDupsModel.getInstance(cluster_id, bugData["PROJECT"], bugData["PRODUCT"], bugData["COMPONENT"])
        print("Got BugDupInst")
        #print(bugDupInst.clusterId)

        if bugDupInst and bugDupInst.clusterId != cluster_id:
            bugDupInst._forgetClassInstanceReferenceForTesting()
            bugDupInst = BugDupsModel.getInstance(cluster_id, bugData["PROJECT"], bugData["PRODUCT"], bugData["COMPONENT"])
            #bugDupInst = BugDups.getInstance()
        else:
            bugDupInst = BugDupsModel.getInstance()

        coll_name = config.BugDupsTrain_Prefix + str(cluster_id) + "_complete"
        print(coll_name)
        db = DbConn()
        collection = db.dbConn[coll_name]

        df = BugDups.load_data_new(db, collection, cluster_id, bugData["PROJECT"], bugData["PRODUCT"], bugData["COMPONENT"])
#         bugData = df.iloc[1]
        #print(df[df['IDENTIFIER']=='CSCtc56807'])
        print(df.shape)
#         df = df[(df['PRODUCT'] == bugData['PRODUCT']) & (df['PROJECT'] == bugData['PROJECT'])]
        #print(df.shape)
        #print(df[df['IDENTIFIER']=='CSCuq51291'])

        df1 = bugDupInst.get_test_data(df, bugData, bugDupInst.w2vmodel, bugDupInst.words, bugDupInst.vocabulary, bugDupInst.inverse_vocabulary, bugDupInst.clusterId, bugDupInst.stops)
#         print(df1)
        text_predictions = BugDups.test_model_text(bugDupInst.malstm, bugDupInst.embeddings, bugDupInst.vocabulary, bugDupInst.w2vmodel, bugDupInst.words, df1, bugDupInst.clusterId, bugDupInst.stops, bugDupInst.max_seq_length, bugDupInst.text_cols)
        #print(text_predictions)
        result = pd.DataFrame()
        result['DUPLICATE_OF'] = df1['DUPLICATE_OF']
        result['Headline'] = df1['DUP_Headline']
        result['Product'] = df1['DUP_PRODUCT']
        result['Project'] = df1['DUP_PROJECT']
        result['Component'] = df1['DUP_COMPONENT']
        result['Status'] = df['LIFECYCLE_STATE_CODE']
        result['SUBMITTED_DATE'] = df['SUBMITTED_DATE']
        print(df.shape, df1.shape)
        #result['Status'] = df1['DUP_LIFECYCLE_STATE_CODE']
        p = []
        for i in text_predictions:
            p.append(i[0]*100)

        result['pred_text'] = p
        result = result.drop_duplicates(subset='DUPLICATE_OF', keep="last")
        #print(result[result['DUPLICATE_OF']=='CSCzv37475'])
        #print(result[result['DUPLICATE_OF']=='CSCzv36398'])
        result = result.sort_values(['pred_text'], ascending=[0])
        result = result[result['pred_text'] > bugDupInst.cut_off]
        if(result.shape[0] > 100):
            v = 100
        else:
            v = result.shape[0]
        print(result)
        if(v != 0):
            result = result.iloc[:v]
            #These below lines till 705 are added to have cosine similarity
            cos_similarities = []
            t1 = bugData['Headline']
            t2 = list(result['Headline'])
            vector1 = BugDups.text_to_vector(t1)
            for i in range(len(t2)):
                vector2 = BugDups.text_to_vector(t2[i])
                cos_similarities.append(BugDups.get_cosine(vector1, vector2))
            idx = [x for x,value in enumerate(cos_similarities) if(value >= 0.6)]
            values = [value for x,value in enumerate(cos_similarities) if(value >= 0.6)]
            if(len(idx) > 0):
                #print(idx)
                result = result.iloc[idx]
                #print(result.shape[0])
                result['cos_sim'] = values
                result = result.sort_values(['cos_sim'], ascending=[0])

            for index, row in result.iterrows():
                temp = dict()
                print(row)
                temp["Identifier"] = row["DUPLICATE_OF"]
                temp["Probability"] = row["pred_text"]
                temp['Similarity'] = row['cos_sim']
                temp["Headline"] = row["Headline"]
                temp['Product'] = row['Product']
                temp['Project'] = row['Project']
                temp['Component'] = row['Component']
                temp['Status'] = row['Status']
                temp['SUBMITTED_DATE'] = row['SUBMITTED_DATE']
                dup = BugDups(**temp)
                dupList.append(dup)

        return dupList

class ViewSetDuplicates(models.Model):
    Identifier = models.CharField(max_length=50)
    Headline = models.CharField(max_length=200)
    Probability = models.DecimalField(decimal_places=2,max_digits=5)
    DuplicateBug = models.CharField(max_length=50)
    ActualDuplicate = models.CharField(max_length=50)

    @staticmethod
    def get_viewset_bug_dups(queryID, viewID=0, bu=None):
        db = DbConn()

        print("In BugDups::get_viewset_bug_dups")
        print(viewID)
        print(queryID)
        print(bu)
        tablePrefix = config.BugDupViewPrefix
        collName = ""
        if viewID == 0:
            collName = tablePrefix + bu + "_" + queryID + "_results"
        else:
            collName = tablePrefix + viewID + "_" + queryID + "_results"

        dups = []

        print(collName)
        resColl = db.dbConn[collName]
        temp = resColl.find().sort([("last_run_date",-1)]).limit(1)
        lastRun = temp[0]['last_run_date']
#         lastRun = lastRun.replace(hour=0, minute=0, second=0, microsecond=0)
        print(lastRun)
        query = []
        match_dict = {"$match":{"$and" :[{"last_run_date": {"$gte": lastRun}}]}}
        project_dict =  {"$project":{"_id":0, "Identifier": "$IDENTIFIER","Headline":"$Headline","DuplicateBug":"$DUPLICATE_LIST","Probability":"$PROBABILITIES","ActualDuplicate":"$actual_duplicate"}}
        query = [ match_dict, project_dict ]
        print(query)
        for res in resColl.aggregate(pipeline=query, allowDiskUse=True):
#             temp = {}
#             temp["Identifier"] = res["Identifier"]
#             temp["Identifier"] = res["Headline"]
#             temp["DuplicateBug"] = res["Headline"]
#
#
            b = ViewSetDuplicates(**res)
            dups.append(b)

        return dups

