#Works for testing on view id and query id
# python ./bug_duplicates_test_view_set.py --env Prod --viewID 673 --queryID 2061
#/auto/vgapps-cstg02-vapps/analytics/csap/models/files/bugDups/
import pymongo
import pandas as pd
import random
import jsondatetime as json2
from time import time
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
stops = set(stopwords.words('english'))
max_seq_length = 150
text_cols = ['complete1', 'complete2']
embedding_dim = 150 #300

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


def parse_options():
    parser = argparse.ArgumentParser(description="""Script to predict Potential CFD testing data""")
    parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
    parser.add_argument("--viewID", default="", help ='View ID', type=str, metavar='v')
    parser.add_argument("--queryID", default="", help ='Query ID', type=str, metavar='q')
    args = parser.parse_args()
    return args


def get_test_data(df, test_df):
	c = pd.DataFrame()
	new_feature_columns = ['DUPLICATE_OF', 'Headline2', 'ENCL-Description2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'complete2', 'is_duplicate']
	feature_columns_to_use = ['IDENTIFIER', 'Headline', 'ENCL-Description', 'PROJECT', 'PRODUCT', 'COMPONENT']
	complete1 = [test_df['complete1']]*df.shape[0]
	for col in new_feature_columns:
		c[col] = df[col]
	for col in feature_columns_to_use:
		c[col] = test_df[col]
	c['complete1'] = complete1
	return c

def load_data(db, collection, is_train):
	cursor = collection.find({})
	df =  pd.DataFrame(list(cursor))
	df["SUBMITTED_DATE"] = pd.to_datetime(df['SUBMITTED_DATE'], unit='ms')
	#df['complete1'] = df["Headline"].astype(str) + " " + df["ENCL-Description"].astype(str)
	#df['complete2'] = df["Headline2"].astype(str) + " " + df["ENCL-Description2"].astype(str)
	if(is_train):
		return df
	else:
		return df[df['SUBMITTED_DATE'] >= str('2018-01')]


def test_model_text(malstm, embeddings, vocabulary, model, words, test_df, cluster):
	#del model
	X = test_df[text_cols]
	Y = test_df['is_duplicate']
	X_test = X
	X_test = {'left': X_test.complete1, 'right': X_test.complete2}
	# Convert labels to their numpy representations
	Y_train = Y.values
	# Zero padding
	for dataset, side in itertools.product([X_test], ['left', 'right']):
		dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)
	#print(dataset)
	# Model variables
	#print("Loaded model from disk")
	# evaluate loaded model on test data
	predictions = malstm.predict([X_test['left'], X_test['right']])
	return predictions


def bugDuplicate_view_set(e, cluster_id, stops, max_seq_length, text_cols, embedding_dim, view_id, query_id, cut_off):
	embedding_vecor_length = 32
	model1 = Sequential()
	model1.add(Embedding(3458230+1, embedding_vecor_length, input_length= 6))#len(X_test[0])))
	model1.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model1.add(MaxPooling1D(pool_size=2))
	model1.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	model1.add(Dense(1, activation='sigmoid'))
	model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	filename = '/data/csap_models/bugDups/model3_lstm_cnn_cat_' + str(cluster_id) + '.h5'
	model1 = load_model(filename) #model = load_model('/data/csap_models/bugDups/model3_lstm_cnn_cat.h5')

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

	final_df = pd.DataFrame()
	duplicate_bugs_length = sum(e['is_duplicate'] == 1)

	for j in range(0, duplicate_bugs_length):
		b1 = e.iloc[j, ]
		duplicate_of = b1['DUPLICATE_OF']
		is_duplicates = b1['is_duplicate']
		df = e[(e['PRODUCT2'] == b1['PRODUCT'])]# & (e['PROJECT2'] == b1['PROJECT'])
		print(j, duplicate_bugs_length, df.shape[0])
		df1 = get_test_data(df, b1)
		#X_test, Y_test = build_test_data_cat(df1)
		#cat_predictions = test_model_cat(model1, X_test, Y_test, cluster_id)
		#vocabulary, w2vmodel, words, test_df = build_test_data_text(w2vmodel, vocabulary, inverse_vocabulary, df1, cluster_id)
		#text_predictions = test_model_text(malstm, embeddings, vocabulary, w2vmodel, words, df1, cluster_id)
		X_test = {'left': df1.complete1, 'right': df1.complete2}
		for dataset, side in itertools.product([X_test], ['left', 'right']):
			dataset[side] = pad_sequences(dataset[side], maxlen=150)
		#print("predictions started")
		text_predictions = malstm.predict([X_test['left'], X_test['right']])
		#print("predictions done")
		result = pd.DataFrame()
		result['IDENTIFIER'] = df1['IDENTIFIER']
		result['DUPLICATE_OF'] = df1['DUPLICATE_OF']
		result['Headline'] = df1['Headline']
		result['ENCL-Description'] = df1['ENCL-Description']
		#p = []
		#for i in cat_predictions:
		#	p.append(i[0]*100)
		#d['pred_cat'] = p
		p = []
		for i in text_predictions:
			p.append(i[0]*100)
		result['pred_text'] = p
		result = result.drop_duplicates(subset='DUPLICATE_OF', keep="last")
		result = result.sort_values(['pred_text'], ascending=[0])
		result = result[result['pred_text'] > cut_off]
		if(result.shape[0] > 10):
			v = 10
		else:
			v = result.shape[0]
		if(v != 0):
			df2 = pd.DataFrame()
			l = []
			l.append(result.iloc[0,]['IDENTIFIER'])
			df2['IDENTIFIER'] = l
			df2['Headline'] = df1['Headline'].iloc[0]
			df2['ENCL-Description'] = df1['ENCL-Description'].iloc[0]
			#df2['PRODUCT'] = result.iloc[0,]['PRODUCT']
			#df2['PROJECT'] = b1['PROJECT']
			#df2['Headline'] = b1['Headline']
			df2['DUPLICATE_LIST'] = ' '.join(list(result.iloc[0:v]['DUPLICATE_OF']))
			df2['PROBABILITIES'] = ' '.join(str(x) for x in list(result.iloc[0:v]['pred_text']))
			if(is_duplicates == 1):
				df2['actual_duplicate'] = duplicate_of
			else:
				df2['actual_duplicate'] = ""
			final_df = final_df.append(df2)
			
	res_coll_name = "BugDupsTestSet_" + str(view_id)+ "_" + str(query_id) + "_results"
	collection = db[res_coll_name]

	final_df = final_df.drop_duplicates(subset='IDENTIFIER', keep="last")
	final_df.reset_index(drop = True, inplace = True)
	final_df = final_df.drop_duplicates(subset='IDENTIFIER', keep="last")
	filename = res_coll_name + '.csv'
	final_df.to_csv(filename, encoding='utf-8')
	records = json2.loads(final_df.T.to_json(date_format='iso')).values()

	collection.create_index([("IDENTIFIER", pymongo.ASCENDING)], unique=True)
	collection.insert(records)
	return final_df



'''
#Verifying the accuracy of the model
dup_lists = list(final_df['DUPLICATE_LIST'].unique())
actual_dups = list(final_df['actual_duplicate'])
i = 0
acc = 0
ids = []
for dup_list in dup_lists:
	a = dup_list.split(' ')
	if(actual_dups[i] in a):
		ids.append(i)
		acc = acc + 1
	i = i + 1

#Find bugs from res_coll_name which have more than 98% match. For those bugs run the test_data_prep_new. Now the dup_list for new bug contains original bug.

collection = db['BugDupsTestSet_3_results_d']
cursor = collection.find({})
results_df =  pd.DataFrame(list(cursor))
results_df = results_df[results_df['pred_cat'] > 0.98]
results_df = results_df.sort_values(['pred_text'], ascending=[0])

#[4, 18, 23, 27, 33, 34, 204, 290, 339, 443, 510, 651, 666, 683, 873, 905, 992, 1052, 1118, 1181, 1262, 1283, 1461, 1483, 1781, 1794, 1840]
#CSCvh40104, CSCvh44844, CSCvh44238, CSCvh53571, CSCvh83909, CSCvh98317, CSCvh44631-18, CSCvh44466-4
for index, row in results_df.iterrows():
	row['IDENTIFIER']
'''

if __name__ == "__main__":
	options = parse_options()
	if(options.env.lower() == "prod"):
		key = "csap_prod_database"

	elif(options.env.lower() == "stage"):
		key = "csap_stage_database"

	db = get_db(settings, key)
	cluster_id = 3
	coll_name = "BugDupsTestSet_" + str(options.viewID) + "_" + str(options.queryID) +"_complete"
	print(coll_name)
	collection = db[coll_name]
	e = load_data(db, collection, True)
	print(e)
	df = e.copy()
	cut_off = 90
	final_df = bugDuplicate_view_set(e, cluster_id, stops, max_seq_length, text_cols, embedding_dim, int(options.viewID), int(options.queryID), cut_off)