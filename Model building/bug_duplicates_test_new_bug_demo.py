#Working mode that runs on the fly.
# python ./bug_duplicates_test_new_bug.py --env Prod --id 4
# /auto/vgapps-cstg02-vapps/analytics/csap/models/files/bugDups/
import pymongo
import pandas as pd
import random
import configparser
import argparse
import jsondatetime as json2
import sys
sys.path.insert(0, "/data/ingestion/")
from Utils import *
from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import nltk
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import matplotlib.pyplot as plt
import itertools
import datetime
import numpy as np
import jsondatetime as json2
import keras
from keras.models import model_from_json
from keras.models import load_model
import h5py
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing import text
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
import sys
sys.path.insert(0, "/data/ingestion/")
from Utils import *
import datetime
import configparser
import shutil
import argparse
import json
import pickle
import xgboost as xgb
import numpy as np
from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV
from xgboost import XGBClassifier
from xgboost import plot_tree
import pylab as pl
from sklearn.metrics import roc_curve, auc

#Setting up the config.ini file parameters
settings = configparser.ConfigParser()
settings.read('/data/ingestion/config.ini')

#Parser options
options = None

pd.options.display.max_colwidth = 300

def parse_options():
    parser = argparse.ArgumentParser(description="""Script to predict Bug Duplicates""")
    parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
    parser.add_argument("--id", default="", help ='id', type=str, metavar='c')
    args = parser.parse_args()
    return args

def build_test_data_text(sentence, w2vmodel, words, vocabulary, inverse_vocabulary, cluster_id, stops):
	q1=[0]
	a = -1
	s = 0
	q2n = []  
	for word in text_to_word_list(sentence):
		# Check for unwanted words
		if word.lower() in stops and word.lower() not in words: #word2vec.vocab:
			continue
		if word not in vocabulary:
			s = 1
		else:
			q2n.append(vocabulary[word])
	q1[a] = q2n
	return q1


chars = ["?", "'s", ">", "<", ",", ":", "'", "''", "--", "`", "``", "...", "", "!", "#", '"', '$', '%', '&', '(', ')', '*', '+', '-', '.', '/', ';', '=', '@', '[', '\\', ']', '^', '_', '{', '}', '|', '~', '\t', '\n', '', 'test', 'case', 'id', 'short', 'description', 'and', 'on', 'if', 'the', 'you', 'of', 'is', 'which', 'what', 'this', 'why', 'during', 'at', 'are', 'to', 'in', 'with', 'for', 'cc', 'email', 'from', 'subject', 'a', 'that', 'yet', 'so', 'raise', 'or', 'then', 'there', 're', 'thanks', 'i', 'as', 'me', 'am', 'attaching', 'thread', 'file', 'along', 'files', 'was', 'it', 'n', 'do', 'does', 'well', 'com']
def get_word(word):
    if word not in chars and '*' not in word and '=' not in word and '++' not in word and '___' not in word and (not word.isdigit()):
        return True
    return False


def text_to_word_list(text):
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
        case = get_word(word)
        if case:
            text_list.append(word)
    return text_list

#Data transformation and imputation of missing values
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)        
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)


def get_test_data(df, test_df, w2vmodel, words, vocabulary, inverse_vocabulary, cluster_id, stops):
	c = pd.DataFrame()
	new_feature_columns = ['DUPLICATE_OF', 'Headline2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'complete2', 'is_duplicate']
	feature_columns_to_use = ['Headline', 'PROJECT', 'PRODUCT', 'COMPONENT']
	#complete1 = [test_df['complete1']]*df.shape[0]
	for col in new_feature_columns:
		c[col] = df[col]
	for col in feature_columns_to_use:
		c[col] = test_df[col]
	sentence = test_df["Headline"] + " " + test_df["ENCL-Description"]
	complete1_int = build_test_data_text(sentence, w2vmodel, words, vocabulary, inverse_vocabulary, cluster_id, stops)
	complete1 = complete1_int*df.shape[0]
	c['complete1'] = complete1
	return c


def load_data(db, collection, is_train):
	#collection = db['BugDupsTrainSet_all_639_1968_new']
	cursor = collection.find({})
	df =  pd.DataFrame(list(cursor))
	df["SUBMITTED_DATE"] = pd.to_datetime(df['SUBMITTED_DATE'], unit='ms')
	if(is_train):
		return df
	else:
		a = df[df['SUBMITTED_DATE'] >= str('2018-01')]
		return df[df['SUBMITTED_DATE'] >= str('2018-01')]



def test_model_text(malstm, embeddings, vocabulary, model, words, test_df, cluster, stops, max_seq_length, text_cols):
	#del model
	X_test = {'left': test_df.complete1, 'right': test_df.complete2}
	# Convert labels to their numpy representations
	# Zero padding
	for dataset, side in itertools.product([X_test], ['left', 'right']):
		dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)
	# Model variables
	#print("Loaded model from disk")
	# evaluate loaded model on test data
	predictions = malstm.predict([X_test['left'], X_test['right']])
	return predictions


def bugDuplicate_new_input(df, b1, cluster_id, stops, max_seq_length, text_cols, embedding_dim, cut_off):
	embedding_vecor_length = 32
	model1 = Sequential()
	model1.add(Embedding(3458230+1, embedding_vecor_length, input_length= 6))#len(X_test[0])))
	model1.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model1.add(MaxPooling1D(pool_size=2))
	model1.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	model1.add(Dense(1, activation='sigmoid'))
	model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	filename = '/data/csap_models/bugDups/model3_lstm_cnn_cat_' + str(cluster_id) + '.h5'
	model1 = load_model(filename)
	filename = '/data/csap_models/bugDups/w2vmodel_' + str(cluster_id) + '.bin'
	w2vmodel = Word2Vec.load(filename)
	f = '/data/csap_models/bugDups/vocab_model_' + str(cluster_id) + '.json'
	vocabulary = json.load(open(f, 'r'))
	thefile = '/data/csap_models/bugDups/inv_vocab_model_' + str(cluster_id) + '.json'
	with open (thefile, 'rb') as fp:
		inverse_vocabulary = pickle.load(fp)

	words = list(w2vmodel.wv.vocab)
	thefile = "/data/csap_models/bugDups/embeddings_model_" + str(cluster_id) + '.json'
	with open (thefile, 'rb') as fp:
		embeddings = pickle.load(fp)

	n_hidden = 50
	gradient_clipping_norm = 1.25
	batch_size = 64
	n_epoch = 1 #25
	def exponent_neg_manhattan_distance(left, right):
		return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

	left_input = Input(shape=(max_seq_length,), dtype='int32')
	right_input = Input(shape=(max_seq_length,), dtype='int32')
	embedding_layer = Embedding(len(embeddings), embedding_dim, weights = [embeddings], input_length=max_seq_length, trainable=False)
	# Embedded version of the inputs
	encoded_left = embedding_layer(left_input)
	encoded_right = embedding_layer(right_input)
	# Since this is a siamese network, both sides share the same LSTM
	shared_lstm = LSTM(n_hidden)
	left_output = shared_lstm(encoded_left)
	right_output = shared_lstm(encoded_right)
	# Calculates the distance as defined by the MaLSTM model
	malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
	# Pack it all up into a model
	malstm = Model([left_input, right_input], [malstm_distance])
	# Adadelta optimizer, with gradient clipping by norm
	optimizer = Adadelta(clipnorm=gradient_clipping_norm)
	malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
	# load weights into new model
	filename = '/data/csap_models/bugDups/text_model_' + str(cluster_id) + '.h5'
	malstm.load_weights(filename)
	#print(b1[['IDENTIFIER', 'Headline', 'ENCL-Description', 'PRODUCT', 'PROJECT', 'COMPONENT']])
	#duplicate_of = b1['DUPLICATE_OF']#
	#is_duplicates = b1['is_duplicate']#
df = e.copy()
df = df[(df['PRODUCT2'] == b1['PRODUCT']) & (df['PROJECT2'] == b1['PROJECT'])]
df1 = get_test_data(df, b1, w2vmodel, words, vocabulary, inverse_vocabulary, cluster_id, stops)#
#X_test, Y_test = build_test_data_cat(df1)
#cat_predictions = test_model_cat(model1, X_test, Y_test, cluster_id)
#vocabulary, w2vmodel, words, test_df = build_test_data_text(w2vmodel, vocabulary, inverse_vocabulary, df1, cluster_id)
text_predictions = test_model_text(malstm, embeddings, vocabulary, w2vmodel, words, df1, cluster_id, stops, max_seq_length, text_cols)
result = pd.DataFrame()
#result['IDENTIFIER'] = df1['IDENTIFIER']#
result['DUPLICATE_OF'] = df1['DUPLICATE_OF']
#p = []
#for i in cat_predictions:
#	p.append(i[0]*100)
#d['pred_cat'] = p
p = []
for i in text_predictions:
	p.append(i[0]*100)

result['pred_text'] = p
#result = d[d['pred_cat'] >= cut_off]
result = result.drop_duplicates(subset='DUPLICATE_OF', keep="last")
result = result.sort_values(['pred_text'], ascending=[0])
#print(result)
result = result[result['pred_text'] > cut_off]
#print(result)
if(result.shape[0] > 10):
	v = 10
else:
	v = result.shape[0]

if(v != 0):
	df2 = pd.DataFrame()
	#l = []
	#l.append(result.iloc[0,]['IDENTIFIER'])
	df2['id'] = [1]
	df2['DUPLICATE_LIST'] = ' '.join(list(result.iloc[0:v]['DUPLICATE_OF']))
	df2['PROBABILITIES'] = ' '.join(str(x) for x in list(result.iloc[0:v]['pred_text']))

else:
	return pd.DataFrame()
return df2


if __name__ == "__main__":
	options = parse_options()
	if(options.env.lower() == "prod"):
		key = "csap_prod_database"

	elif(options.env.lower() == "stage"):
		key = "csap_stage_database"

	db = get_db(settings, key)
	cluster_id = 3
	coll_name = "BugDupsTrainSet_" + str(cluster_id)+ "_complete"
	#coll_name = "BugDupsTestSet_673_2061_complete"
	collection = db[coll_name]
	e = load_data(db, collection, False)

	j = int(options.id) #4-y, 18-y, 3-n, 27-n
	b1 = e.iloc[j, ]
	print(b1['IDENTIFIER'])
	#b1 = b1[['PRODUCT', 'PROJECT', 'COMPONENT', 'Headline', 'ENCL-Description']]
	#print(b1)

	stops = set(stopwords.words('english'))
	max_seq_length = 150
	text_cols = ['complete1', 'complete2']
	embedding_dim = 150
	cut_off = 90
	final_df = bugDuplicate_new_input(e, b1, cluster_id, stops, max_seq_length, text_cols, embedding_dim, cut_off) #b1 is the input from GUI
	if(final_df.shape[0] != 0):
		print(final_df)
	else:
		print("No dulicates")