#This is only for viewID and queryID. This needs to be run after the cluterID model is trained on trainSet
# python ./bug_duplicates_build_complete_list.py --env Prod --viewID 673 --queryID 2061
# python ./bug_duplicates_build_complete_list.py --env Prod --cluster 3

#The build and all if fine for clusterID and training.
#But for testing we need not build budTestSet_viewID_queryID, instead we can direcly build a potCFD kind of thing with a new column complete for each bug.
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

stops = set(stopwords.words('english'))
max_seq_length = 150 #150 #If only headline, let's use small length
text_cols = ['complete1', 'complete2']
embedding_dim = 150 #300


def parse_options():
    parser = argparse.ArgumentParser(description="""Script to predict Potential CFD testing data""")
    parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
    parser.add_argument("--clusterID", default="", help ='Comma seperated cluster-ids', type=str, metavar='c')
    parser.add_argument("--viewID", default="", help ='View ID', type=str, metavar='v')
    parser.add_argument("--queryID", default="", help ='Query ID', type=str, metavar='q')
    args = parser.parse_args()
    
    return args

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


#Loading the training data
def load_data(db, collection, is_train):
	cursor = collection.find({})
	df =  pd.DataFrame(list(cursor))
	df['complete1'] = df["Headline"].astype(str) + " " + df["ENCL-Description"].astype(str)
	df['complete2'] = df["Headline2"].astype(str) + " " + df["ENCL-Description2"].astype(str)
	return df


def build_test_data_text(test_df, cluster, db, is_view, view_id, query_id):
	filename = '/auto/vgapps-cstg02-vapps/analytics/csap/models/files/bugDups/w2vmodel_' + str(cluster) + '.bin'
	model = Word2Vec.load(filename)
	print("Loaded the W2V")
	words = list(model.wv.vocab)

	f = '/auto/vgapps-cstg02-vapps/analytics/csap/models/files/bugDups/vocab_model_' + str(cluster) + '.json'
	vocabulary = json.load(open(f, 'r'))

	thefile = '/auto/vgapps-cstg02-vapps/analytics/csap/models/files/bugDups/inv_vocab_model_' + str(cluster) + '.json'
	with open (thefile, 'rb') as fp:
		inverse_vocabulary = pickle.load(fp)

	c=0
	q1=[0]*(test_df.shape[0])
	q2=[0]*(test_df.shape[0])
	a = -1
	s = 0
	# Iterate over the questions only of both training and test datasets
	for dataset in [test_df]:#[train_df, test_df]:
		for index, row in dataset.iterrows():
			print(a, test_df.shape[0])
			a = a+1
			# Iterate through the text of both questions of the row
			for question in text_cols:
				c = c + 1
				q2n = []  # q2n -> question numbers representation
				for word in text_to_word_list(row[question]):
					# Check for unwanted words
					if word.lower() in stops and word.lower() not in words: #word2vec.vocab:
						continue
					if word not in vocabulary:
						#vocabulary[word] = len(inverse_vocabulary)
						#q2n.append(0)#len(inverse_vocabulary))
						#inverse_vocabulary.append(word)
						s = 1
					else:
						q2n.append(vocabulary[word])
				# Replace questions as word to question as number representation
				if(c%2 != 0):
					 q1[a] = q2n
				else:
					 q2[a] = q2n
				#dataset.set_value(index, question, q2n)

	test_df['complete1'] = q1
	test_df['complete2'] = q2
	if(is_view == True):
		c = "BugDupsTestSet_" + str(view_id) + "_" + str(query_id) + "_complete"
	else:
		c = "BugDupsTestSet_" + str(cluster) + "_complete"

	print(c)
	collection = db[c]
	d = test_df[['IDENTIFIER', 'complete1', 'Headline', 'ENCL-Description', 'DE_MANAGER_USERID', 'SEVERITY_CODE', 'LIFECYCLE_STATE_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'AGE',  'FEATURE', 'RELEASE_NOTE', 'SA_ATTACHMENT_INDIC', 'CR_ATTACHMENT_INDIC', 'UT_ATTACHMENT_INDIC', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'TICKETS_COUNT', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'TS_INDIC', 'SS_INDIC', 'OIB_INDIC', 'STATE_ASSIGN_INDIC', 'STATE_CLOSE_INDIC', 'STATE_DUPLICATE_INDIC', 'STATE_FORWARD_INDIC', 'STATE_HELD_INDIC', 'STATE_INFO_INDIC', 'STATE_JUNK_INDIC', 'STATE_MORE_INDIC', 'STATE_NEW_INDIC', 'STATE_OPEN_INDIC', 'STATE_POSTPONE_INDIC', 'STATE_RESOLVE_INDIC', 'STATE_SUBMIT_INDIC', 'STATE_UNREP_INDIC', 'STATE_VERIFY_INDIC', 'STATE_WAIT_INDIC', 'CFR_INDIC', 'S12RD_INDIC', 'S123RD_INDIC', 'MISSING_SS_EVAL_INDIC', 'S123_INDIC', 'S12_INDIC', 'RNE_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY',  'TEST_EDP_PHASE', 'RESOLVER_ANALYSIS_INDIC', 'SUBMITTER_ANALYSIS_INDIC', 'EDP_ANALYSIS_INDIC', 'RETI_ANALYSIS_INDIC', 'DESIGN_REVIEW_ESCAPE_INDIC', 'STATIC_ANALYSIS_ESCAPE_INDIC', 'FUNC_TEST_ESCAPE_INDIC', 'SELECT_REG_ESCAPE_INDIC', 'CODE_REVIEW_ESCAPE_INDIC', 'UNIT_TEST_ESCAPE_INDIC', 'DEV_ESCAPE_INDIC', 'FEATURE_TEST_ESCAPE_INDIC', 'REG_TEST_ESCAPE_INDIC', 'SYSTEM_TEST_ESCAPE_INDIC', 'SOLUTION_TEST_ESCAPE_INDIC', 'INT_TEST_ESCAPE_INDIC', 'GO_TEST_ESCAPE_INDIC', 'COMPLETE_ESCAPE_INDIC', 'SR_CNT', 'PSIRT_INDIC',  'BADCODEFLAG',   'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'URC_DISPOSED_INDIC', 'CLOSED_DISPOSED_INDIC', 'REGRESSION_BUG_FLAG', 'SUBMITTED_DATE', 'DUPLICATE_OF', 'complete2', 'Headline2', 'ENCL-Description2', 'DE_MANAGER_USERID2', 'SEVERITY_CODE2', 'LIFECYCLE_STATE_CODE2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'AGE2', 'FEATURE2', 'RELEASE_NOTE2', 'SA_ATTACHMENT_INDIC2', 'CR_ATTACHMENT_INDIC2', 'UT_ATTACHMENT_INDIC2', 'IMPACT2', 'ORIGIN2', 'IS_CUSTOMER_VISIBLE2', 'TICKETS_COUNT2', 'INCOMING_INDIC2', 'BACKLOG_INDIC2', 'DISPOSED_INDIC2', 'TS_INDIC2', 'SS_INDIC2', 'OIB_INDIC2', 'STATE_ASSIGN_INDIC2', 'STATE_CLOSE_INDIC2', 'STATE_DUPLICATE_INDIC2', 'STATE_FORWARD_INDIC2', 'STATE_HELD_INDIC2', 'STATE_INFO_INDIC2', 'STATE_JUNK_INDIC2', 'STATE_MORE_INDIC2', 'STATE_NEW_INDIC2', 'STATE_OPEN_INDIC2', 'STATE_POSTPONE_INDIC2', 'STATE_RESOLVE_INDIC2', 'STATE_SUBMIT_INDIC2', 'STATE_UNREP_INDIC2', 'STATE_VERIFY_INDIC2', 'STATE_WAIT_INDIC2', 'CFR_INDIC2', 'S12RD_INDIC2', 'S123RD_INDIC2', 'MISSING_SS_EVAL_INDIC2', 'S123_INDIC2', 'S12_INDIC2', 'RNE_INDIC2', 'UPDATED_BY2', 'DEV_ESCAPE_ACTIVITY2', 'RELEASED_CODE2', 'TEST_EDP_ACTIVITY2', 'TEST_EDP_PHASE2', 'RESOLVER_ANALYSIS_INDIC2', 'SUBMITTER_ANALYSIS_INDIC2', 'EDP_ANALYSIS_INDIC2', 'RETI_ANALYSIS_INDIC2', 'DESIGN_REVIEW_ESCAPE_INDIC2', 'STATIC_ANALYSIS_ESCAPE_INDIC2', 'FUNC_TEST_ESCAPE_INDIC2', 'SELECT_REG_ESCAPE_INDIC2', 'CODE_REVIEW_ESCAPE_INDIC2', 'UNIT_TEST_ESCAPE_INDIC2', 'DEV_ESCAPE_INDIC2', 'FEATURE_TEST_ESCAPE_INDIC2', 'REG_TEST_ESCAPE_INDIC2', 'SYSTEM_TEST_ESCAPE_INDIC2', 'SOLUTION_TEST_ESCAPE_INDIC2', 'INT_TEST_ESCAPE_INDIC2', 'GO_TEST_ESCAPE_INDIC2', 'COMPLETE_ESCAPE_INDIC2', 'SR_CNT2', 'PSIRT_INDIC2', 'BADCODEFLAG2',  'RISK_OWNER2', 'SIR2', 'PSIRT_FLAG2', 'URC_DISPOSED_INDIC2', 'CLOSED_DISPOSED_INDIC2', 'REGRESSION_BUG_FLAG2', 'is_duplicate']] #[['is_duplicate', 'SUBMITTED_DATE', 'IDENTIFIER', 'DUPLICATE_OF', 'Headline', 'ENCL-Description', 'Headline2', 'ENCL-Description2', 'complete1', 'complete2', 'PRODUCT', 'PRODUCT2', 'PROJECT', 'PROJECT2', 'COMPONENT', 'COMPONENT2']]
	print(d.columns)
	'''
	d = pd.DataFrame()
	d['IDENTIFIER'] = t_df['IDENTIFIER']
	d['complete1'] = t_df['complete1']
	d['DUPLICATE_OF'] = t_df['DUPLICATE_OF']
	d['complete2'] = t_df['complete2']
	d['Headline2'] = t_df['Headline2']
	d['Headline'] = t_df['Headline']
	d['ENCL-Description'] = t_df['ENCL-Description']
	d['ENCL-Description2'] = t_df['ENCL-Description2']
	d['is_duplicate'] = t_df['is_duplicate']
	d['PRODUCT'] = t_df['PRODUCT']
	d['PRODUCT2'] = t_df['PRODUCT2']
	d['PROJECT'] = t_df['PROJECT']
	d['PROJECT2'] = t_df['PROJECT2']
	d['COMPONENT'] = t_df['COMPONENT']
	d['COMPONENT2'] = t_df['COMPONENT2']
	d['SUBMITTED_DATE'] = t_df['SUBMITTED_DATE']
	'''
	records = json.loads(d.T.to_json()).values()
	print('Records loaded')
	collection.drop()
	collection.create_index([("IDENTIFIER", pymongo.ASCENDING), ("DUPLICATE_OF", pymongo.ASCENDING)], unique=True)
	collection.insert(records)

	return 0


def main():
	options = parse_options()
	if(options.env.lower() == "prod"):
		key = "csap_prod_database"

	elif(options.env.lower() == "stage"):
		key = "csap_stage_database"

	db = get_db(settings, key)

	if(options.viewID == ""):
		cluster_id = int(options.clusterID)
		coll_name = "BugDupsTrainSet_" + str(cluster_id)
		is_view = False
		collection = db[coll_name] #db['BugDupsTrainSet_all_3']
		print(collection)
		df = load_data(db, collection, True)
		build_test_data_text(df, cluster_id, db, is_view, 0, 0)
	else:
		cluster_id = 3
		coll_name = "BugDupsTestSet_" + str(options.viewID) + "_" + str(options.queryID)
		is_view = True
		collection = db[coll_name] #db['BugDupsTrainSet_all_639_1968_new']
		print(collection)
		df = load_data(db, collection, True)
		build_test_data_text(df, cluster_id, db, is_view, int(options.viewID), int(options.queryID))
	

if __name__ == "__main__":
    main()
