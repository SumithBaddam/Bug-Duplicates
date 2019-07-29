#This is only for viewID and queryID. This needs to be run after the cluterID model is trained on trainSet
# python ./bug_duplicates_build_complete_list_viewID.py --env Prod --viewID 436 --queryID 1452

#/auto/vgapps-cstg02-vapps/analytics/csap/models/files/bugDups/
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
potcfd_view_prefix = str(settings.get("Potential_CFD","viewPrefix"))

#Parser options
options = None

stops = set(stopwords.words('english'))
max_seq_length = 150 #150 #If only headline, let's use small length
text_cols = ['wrdEmbSet']
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
	df['wrdEmbSet'] = df["Headline"].astype(str) + " " + df["ENCL-Description"].astype(str)
	#df['complete2'] = df["Headline2"].astype(str) + " " + df["ENCL-Description2"].astype(str)
	return df


def build_test_data_text(test_df, cluster, db, is_view, view_id, query_id):
	filename = str(model_path) + 'w2vmodel_' + str(cluster) + '.bin' #'/data/csap_models/bugDups/w2vmodel_' + str(cluster) + '.bin'
	model = Word2Vec.load(filename)
	print("Loaded the W2V")
	words = list(model.wv.vocab)

	f = str(model_path) + 'vocab_model_' + str(cluster) + '.json' #'/data/csap_models/bugDups/vocab_model_' + str(cluster) + '.json'
	vocabulary = json.load(open(f, 'r'))

	thefile = str(model_path) + 'inv_vocab_model_' + str(cluster) + '.json' #'/data/csap_models/bugDups/inv_vocab_model_' + str(cluster) + '.json'
	with open (thefile, 'rb') as fp:
		inverse_vocabulary = pickle.load(fp)

	c=0
	q1=[0]*(test_df.shape[0])
	a = -1
	s = 0
	for dataset in [test_df]:
		for index, row in dataset.iterrows():
			print(a, test_df.shape[0])
			a = a + 1
			for question in text_cols:
				c = c + 1
				q2n = []
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
				q1[a] = q2n
				
	print(len(q1), test_df.shape[0])
	test_df['wrdEmbSet'] = q1
	if(is_view == True):
		c = str(test_prefix ) + str(view_id) + "_" + str(query_id) + "_complete" #"BugDupsTestSet_" + str(view_id) + "_" + str(query_id) + "_complete"
	else:
		c = "BugDupsTestSet_" + str(cluster) + "_complete"

	print(c)
	collection = db[c]
	#['IDENTIFIER', 'complete1', 'Headline', 'ENCL-Description', 'DE_MANAGER_USERID', 'SEVERITY_CODE', 'LIFECYCLE_STATE_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'IMPACT', 'ORIGIN', 'RISK_OWNER', 'SUBMITTED_DATE', 'DUPLICATE_OF']
	d = test_df[['IDENTIFIER', 'wrdEmbSet', 'Headline', 'ENCL-Description', 'DE_MANAGER_USERID', 'SEVERITY_CODE', 'LIFECYCLE_STATE_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'AGE',  'FEATURE', 'RELEASE_NOTE', 'SA_ATTACHMENT_INDIC', 'CR_ATTACHMENT_INDIC', 'UT_ATTACHMENT_INDIC', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'TICKETS_COUNT', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'TS_INDIC', 'SS_INDIC', 'OIB_INDIC', 'STATE_ASSIGN_INDIC', 'STATE_CLOSE_INDIC', 'STATE_DUPLICATE_INDIC', 'STATE_FORWARD_INDIC', 'STATE_HELD_INDIC', 'STATE_INFO_INDIC', 'STATE_JUNK_INDIC', 'STATE_MORE_INDIC', 'STATE_NEW_INDIC', 'STATE_OPEN_INDIC', 'STATE_POSTPONE_INDIC', 'STATE_RESOLVE_INDIC', 'STATE_SUBMIT_INDIC', 'STATE_UNREP_INDIC', 'STATE_VERIFY_INDIC', 'STATE_WAIT_INDIC', 'CFR_INDIC', 'S12RD_INDIC', 'S123RD_INDIC', 'MISSING_SS_EVAL_INDIC', 'S123_INDIC', 'S12_INDIC', 'RNE_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY',  'TEST_EDP_PHASE', 'RESOLVER_ANALYSIS_INDIC', 'SUBMITTER_ANALYSIS_INDIC', 'EDP_ANALYSIS_INDIC', 'RETI_ANALYSIS_INDIC', 'DESIGN_REVIEW_ESCAPE_INDIC', 'STATIC_ANALYSIS_ESCAPE_INDIC', 'FUNC_TEST_ESCAPE_INDIC', 'SELECT_REG_ESCAPE_INDIC', 'CODE_REVIEW_ESCAPE_INDIC', 'UNIT_TEST_ESCAPE_INDIC', 'DEV_ESCAPE_INDIC', 'FEATURE_TEST_ESCAPE_INDIC', 'REG_TEST_ESCAPE_INDIC', 'SYSTEM_TEST_ESCAPE_INDIC', 'SOLUTION_TEST_ESCAPE_INDIC', 'INT_TEST_ESCAPE_INDIC', 'GO_TEST_ESCAPE_INDIC', 'COMPLETE_ESCAPE_INDIC', 'SR_CNT', 'PSIRT_INDIC',  'BADCODEFLAG',   'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'URC_DISPOSED_INDIC', 'CLOSED_DISPOSED_INDIC', 'REGRESSION_BUG_FLAG', 'SUBMITTED_DATE', 'DUPLICATE_OF']]
	d = d.drop_duplicates(subset='IDENTIFIER', keep="last")
	now = datetime.datetime.now()
	d['emb_last_run_date'] = now.strftime("%Y-%m-%d")
	d.reset_index(drop = True, inplace = True)
	records = json.loads(d.T.to_json()).values()
	print('Records loaded')
	collection.drop()
	collection.create_index([("IDENTIFIER", pymongo.ASCENDING), ("DUPLICATE_OF", pymongo.ASCENDING)], unique=True)
	collection.insert(records)

	return 0

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


def main():
	options = parse_options()
	if(options.env.lower() == "prod"):
		key = "csap_prod_database"

	elif(options.env.lower() == "stage"):
		key = "csap_stage_database"

	db = get_db(settings, key)

	#cluster_id = 1
	coll_name = str(potcfd_view_prefix) + str(options.viewID) + "_" + str(options.queryID) #"PotentialCFD_ViewSet_" + str(options.viewID) + "_" + str(options.queryID)
	is_view = True
	collection = db[coll_name]
	print(collection)

	df = load_data(db, collection, True)
	c = str(test_prefix ) + str(options.viewID) + "_" + str(options.queryID) + "_complete"
	col = db[c]
	cursor = col.find({})
	last_date_df =  pd.DataFrame(list(cursor))
	if(last_date_df.shape[0] > 0):
		emb_last_run_date = last_date_df['emb_last_run_date'].iloc[-1]
		df = df[df['csap_last_run_date'] > emb_last_run_date]

	print(df.shape)
	if(df.shape[0] > 0):
		project = list(df['PROJECT'].unique())[0]
		cluster_id = get_cluster_id(db, project)
		print(cluster_id)
		build_test_data_text(df, cluster_id, db, is_view, options.viewID, int(options.queryID))
	else:
		print("Nothing new in ", options.viewID, options.queryID)

if __name__ == "__main__":
    main()