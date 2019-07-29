#Works for testing on view id and query id
# python ./bug_duplicates_test_view_set.py --env Prod --viewID 436 --queryID 1452
#/auto/vgapps-cstg02-vapps/analytics/csap/models/files/bugDups/ changed to /data/csap_models/BugDups/
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
train_prefix = str(settings.get("BugDuplicates","training_prefix"))
model_path = str(settings.get("BugDuplicates","modelfilepath"))
test_prefix = str(settings.get("BugDuplicates","test_prefix"))

#Parser options
options = None

pd.options.display.max_colwidth = 300
stops = set(stopwords.words('english'))
max_seq_length = 150
text_cols = ['wrdEmbSet', 'DUP_wrdEmbSet']
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

def build_test_data_cat_2(df):
	new_feature_columns = ['DUP_DE_MANAGER_USERID', 'DUP_SEVERITY_CODE', 'DUP_PROJECT', 'DUP_PRODUCT', 'DUP_COMPONENT', 'DUP_FEATURE', 'DUP_IMPACT', 'DUP_ORIGIN', 'DUP_RISK_OWNER'] #Works the best
	feature_columns_to_use = ['DE_MANAGER_USERID', 'SEVERITY_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'IMPACT', 'ORIGIN', 'RISK_OWNER'] #Works the best
	avoid_cols = ['DE_MANAGER_USERID', 'COMPONENT', 'PROJECT', 'PRODUCT', 'FEATURE', 'IMPACT', 'ORIGIN','RISK_OWNER', 'FEATURE'] #Works the best
	cols = [] #feature_columns_to_use + new_feature_columns + ['is_duplicate']
	for i in range(0, len(new_feature_columns)):
		cols.append(feature_columns_to_use[i])
		cols.append(new_feature_columns[i])
	k = 0
	avoid_cols_indexes = []
	for col in cols:
		if col in avoid_cols:
			avoid_cols_indexes.append(k)
		k = k + 1
	big_X = df[cols]
	nonnumeric_columns_1 = ['DE_MANAGER_USERID', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'IMPACT', 'ORIGIN', 'RISK_OWNER']
	nonnumeric_columns_2 = ['DUP_DE_MANAGER_USERID', 'DUP_PROJECT', 'DUP_PRODUCT', 'DUP_COMPONENT', 'DUP_FEATURE', 'DUP_IMPACT', 'DUP_ORIGIN', 'DUP_RISK_OWNER']
	nonnumeric_columns = []
	for i in range(0, len(nonnumeric_columns_1)):
		nonnumeric_columns.append(nonnumeric_columns_1[i])
		nonnumeric_columns.append(nonnumeric_columns_2[i])
	big_X = big_X.replace(np.nan, '', regex=True)
	big_X_imputed = DataFrameImputer().fit_transform(big_X.iloc[:,:])
	le = LabelEncoder()
	for feature in nonnumeric_columns:
		big_X_imputed[feature] = big_X_imputed[feature].astype(str)
	for i in range(0, len(nonnumeric_columns), 2):
		#print(i)
		u1 = list(big_X_imputed[nonnumeric_columns[i]].unique())
		u2 = list(big_X_imputed[nonnumeric_columns[i + 1]].unique())
		u1 = u1 + u2
		myset = list(set(u1))
		d = dict()
		for j in range(0, len(myset)):
			d[myset[j]] = j
		big_X_imputed = big_X_imputed.replace({nonnumeric_columns[i]: d})
		big_X_imputed = big_X_imputed.replace({nonnumeric_columns[i + 1]: d})
	train_X = big_X_imputed.iloc[:, :len(cols)].as_matrix()
	X_train = train_X
	for i in range(0, len(X_train)):
		for k in avoid_cols_indexes:
			if(X_train[i][k] == X_train[i][k + 1]):
				X_train[i][k] = 1
				X_train[i][k + 1] = 1
			else:
				X_train[i][k] = 0
				X_train[i][k + 1] = 1
	return X_train


def test_model_cat(model, X_test):
	prediction = model.predict(X_test)
	#print(X_test[4:6])
	return prediction

def get_test_data(df, test_df):
	c = pd.DataFrame()
	new_feature_columns = ['DUPLICATE_OF', 'DUP_Headline', 'DUP_ENCL-Description', 'DUP_DE_MANAGER_USERID', 'DUP_SEVERITY_CODE', 'DUP_PROJECT', 'DUP_PRODUCT', 'DUP_COMPONENT', 'DUP_FEATURE', 'DUP_IMPACT', 'DUP_ORIGIN', 'DUP_RISK_OWNER', 'DUP_wrdEmbSet']#['DUPLICATE_OF', 'Headline2', 'ENCL-Description2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'complete2', 'is_duplicate']
	feature_columns_to_use = ['IDENTIFIER', 'Headline', 'ENCL-Description', 'DE_MANAGER_USERID', 'SEVERITY_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'IMPACT', 'ORIGIN', 'RISK_OWNER', 'wrdEmbSet']
	#['IDENTIFIER', 'Headline', 'ENCL-Description', 'PROJECT', 'PRODUCT', 'COMPONENT']
	complete1 = [test_df['wrdEmbSet']]*df.shape[0]
	i = 0
	for col in new_feature_columns:
		c[col] = df[feature_columns_to_use[i]]
		i = i + 1
	for col in feature_columns_to_use:
		col_value = [test_df[col]]*df.shape[0]
		c[col] = col_value #test_df[col]
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
	X = test_df[text_cols]
	X_test = X
	X_test = {'left': X_test.wrdEmbSet, 'right': X_test.DUP_wrdEmbSet}
	for dataset, side in itertools.product([X_test], ['left', 'right']):
		dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)
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
	filename = model_path + 'usecase2/model3_lstm_cnn_cat_' + str(cluster_id) + '.h5' #'/data/csap_models/bugDups/usecase2/model3_lstm_cnn_cat_' + str(cluster_id) + '.h5'
	model1 = load_model(filename)

	filename = model_path + 'w2vmodel_' + str(cluster_id) + '.bin' #'/data/csap_models/bugDups/w2vmodel_' + str(cluster_id) + '.bin'
	w2vmodel = Word2Vec.load(filename)
	f = model_path + 'vocab_model_' + str(cluster_id) + '.json' #'/data/csap_models/bugDups/vocab_model_' + str(cluster_id) + '.json'
	vocabulary = json.load(open(f, 'r'))
	thefile = model_path + 'inv_vocab_model_' + str(cluster_id) + '.json' #'/data/csap_models/bugDups/inv_vocab_model_' + str(cluster_id) + '.json'
	with open (thefile, 'rb') as fp:
		inverse_vocabulary = pickle.load(fp)

	words = list(w2vmodel.wv.vocab)

	thefile = model_path + 'embeddings_model_' + str(cluster_id) + '.json' #"/data/csap_models/bugDups/embeddings_model_" + str(cluster_id) + '.json'
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
	filename = model_path + "text_model_" + str(cluster_id) + '.h5' #'/data/csap_models/bugDups/text_model_' + str(cluster_id) + '.h5'
	malstm.load_weights(filename)

	res_coll_name = test_prefix + str(view_id)+ "_" + str(query_id) + "_results" #"BugDupsTestSet_" + str(view_id)+ "_" + str(query_id) + "_results"
	collection = db[res_coll_name]
	collection.drop()
	final_df = pd.DataFrame()
	duplicate_bugs_length = e.shape[0] #sum(e['is_duplicate'] == 1)
	now = datetime.datetime.now()
	comp_collection = db['BugDupsTrainSet_components']
	cursor = comp_collection.find({'cluster_id': str(cluster_id)})
	d =	list(cursor)[0]

	for j in range(0, duplicate_bugs_length):
		b1 = e.iloc[j, ]
		duplicate_of = b1['DUPLICATE_OF']
		status = b1['LIFECYCLE_STATE_CODE']
		identifier = b1['IDENTIFIER']
		original_comp = b1['COMPONENT']
		
		components = []
		components.append(original_comp)
		if(original_comp in d.keys()):
			for mapped_comp in set(d[original_comp]):
				components.append(mapped_comp)
		
		df = e[(e['PRODUCT'] == b1['PRODUCT']) & (e['PROJECT'] == b1['PROJECT']) & (e['COMPONENT'].isin(components))]
		#df = e[(e['PRODUCT'] == b1['PRODUCT']) & (e['PROJECT'] == b1['PROJECT'])]#e[(e['PRODUCT2'] == b1['PRODUCT']) & (e['PROJECT2'] == b1['PROJECT'])
		#df.drop(df.index[j], inplace=True)
		df1 = get_test_data(df, b1)
		#break
		'''
		X_test = build_test_data_cat_2(df1)
		cat_predictions = test_model_cat(model1, X_test)
		df1['cat_predictions'] = cat_predictions
		new_df = df1[df1['cat_predictions'] >= 0.9]
		new_cat_predictions = new_df['cat_predictions']
		'''
		print(j, duplicate_bugs_length, df1.shape[0], duplicate_of, final_df.shape[0])
		#vocabulary, w2vmodel, words, test_df = build_test_data_text(w2vmodel, vocabulary, inverse_vocabulary, df1, cluster_id)
		text_predictions = test_model_text(malstm, embeddings, vocabulary, w2vmodel, words, df1, cluster_id)#new_df, cluster_id)
		#df1 = new_df
		result = pd.DataFrame()
		result['IDENTIFIER'] = df1['IDENTIFIER']
		result['DUPLICATE_OF'] = df1['DUPLICATE_OF']
		result['Headline'] = df1['Headline']
		result['ENCL-Description'] = df1['ENCL-Description']
		'''
		p = []
		for i in new_cat_predictions:
			p.append(i*100)
		result['pred_cat'] = p
		'''
		p = []
		for i in text_predictions:
			p.append(i[0]*100)
		result['pred_text'] = p
		result = result.drop_duplicates(subset='DUPLICATE_OF', keep="last")
		result = result.sort_values(['pred_text'], ascending=[0])
		result = result[result['pred_text'] > cut_off]
		result = result[result['DUPLICATE_OF'] != identifier]
		if(result.shape[0] > 5):
			v = 5
		else:
			v = result.shape[0]
		if(v != 0):
			df2 = pd.DataFrame()
			l = []
			l.append(result.iloc[0,]['IDENTIFIER'])
			df2['IDENTIFIER'] = l
			df2['Headline'] = df1['Headline'].iloc[0]
			df2['ENCL-Description'] = df1['ENCL-Description'].iloc[0]
			df2['DUPLICATE_LIST'] = ' '.join(list(result.iloc[0:v]['DUPLICATE_OF']))
			df2['TEXT_PROBABILITIES'] = ' '.join(str(x) for x in list(result.iloc[0:v]['pred_text']))
			#df2['CAT_PROBABILITIES'] = ' '.join(str(x) for x in list(result.iloc[0:v]['pred_cat']))
			df2['actual_duplicate'] = duplicate_of
			df2['STATUS'] = status
			df2['last_run_date'] = now.strftime("%Y-%m-%d")
			final_df = final_df.append(df2)

		if(final_df.shape[0] > 100):
			final_df.reset_index(drop = True, inplace = True)
			#filename = '/data/csap_models/bugDups/' + res_coll_name + '.csv'
			#final_df.to_csv(filename, encoding='utf-8')
			#records = json2.loads(final_df.T.to_json(date_format='iso')).values()
			records = json.loads(final_df.T.to_json()).values()
			collection.create_index([("IDENTIFIER", pymongo.ASCENDING)], unique=True)
			collection.insert(records)
			del final_df
			final_df = pd.DataFrame()

	final_df = final_df.drop_duplicates(subset='IDENTIFIER', keep="last")
	final_df.reset_index(drop = True, inplace = True)
	#records = json2.loads(final_df.T.to_json(date_format='iso')).values()
	records = json.loads(final_df.T.to_json()).values()
	collection.create_index([("IDENTIFIER", pymongo.ASCENDING)], unique=True)
	collection.insert(records)
	return final_df

'''
final_df = pd.read_csv("BugDupsTestSet_436_1452_results.csv", encoding='utf-8')
final_df1 = pd.DataFrame()
for index, row in final_df.iterrows():
	print(index)
	if(row['IDENTIFIER'] in row['DUPLICATE_LIST']):
		a = row['DUPLICATE_LIST'].split(' ')
		c = row['CAT_PROBABILITIES'].split(' ')
		t = row['TEXT_PROBABILITIES'].split(' ')
		i = a.index(row['IDENTIFIER'])
		#a = a.remove(row['IDENTIFIER'])
		a = a[:i] + a[i+1:]
		c = c[:i] + c[i+1:]
		t = t[:i] + t[i+1:]
		if(a != None):
			row['DUPLICATE_LIST'] = ' '.join(list(a))
			row['CAT_PROBABILITIES'] = ' '.join(list(c))
			row['TEXT_PROBABILITIES'] = ' '.join(list(t))
		else:
			row['DUPLICATE_LIST'] = ''
			row['CAT_PROBABILITIES'] = ''
			row['TEXT_PROBABILITIES'] = ''
	final_df1 = final_df1.append(row)
/auto/vgapps-cstg02-vapps/analytics/csap/models/files/bugDups/

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
def get_cluster_id(db, project):
    collection = db[settings.get('Potential_CFD', 'proj_cluster')]
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


if __name__ == "__main__":
	options = parse_options()
	if(options.env.lower() == "prod"):
		key = "csap_prod_database"

	elif(options.env.lower() == "stage"):
		key = "csap_stage_database"

	db = get_db(settings, key)
	coll_name = test_prefix + str(options.viewID) + "_" + str(options.queryID) +"_complete" # "BugDupsTestSet_" + str(options.viewID) + "_" + str(options.queryID) +"_complete"
	print(coll_name)
	collection = db[coll_name]
	e = load_data(db, collection, True)

	c = test_prefix + str(options.viewID) + "_" + str(options.queryID) +"_results"
	col = db[c]
	cursor = col.find({})
	last_date_df =  pd.DataFrame(list(cursor))
	if(last_date_df.shape[0] > 0):
		last_run_date = last_date_df['last_run_date'].iloc[-1]
		e = e[e['emb_last_run_date'] > last_run_date]

	print(e.shape)
	if(e.shape[0] > 0):
		project = list(e['PROJECT'].unique())[0]
		cluster_id = get_cluster_id(db, project)
		print(cluster_id)
		df = e.copy()
		cut_off = 95
		final_df = bugDuplicate_view_set(e, cluster_id, stops, max_seq_length, text_cols, embedding_dim, options.viewID, int(options.queryID), cut_off)
