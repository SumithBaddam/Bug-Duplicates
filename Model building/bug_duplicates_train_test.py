# python ./bug_duplicates_train_test.py --env Prod --train True --cluster 3
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

#Parser options
options = None

stops = set(stopwords.words('english'))
max_seq_length = 150 #150 #If only headline - 18, else - 100 ??
text_cols = ['wrdEmbSet', 'dup_wrdEmbSet']
embedding_dim = 150 #300

def get_cluster(db, df):
	#Getting the clusters data
	collection = db[settings.get('Potential_CFD', 'proj_cluster')]
	cursor = collection.find({})
	clusters =  pd.DataFrame(list(cursor))
	project_clusters = []
	groups = clusters.groupby('Cluster')
	for name, group in groups:
		project_clusters.append(list(group['Project']))
	print(project_clusters)
	req_cluster = list(df['PROJECT'].unique())
	print(req_cluster)
	p = 0
	cluster_id = 0
	f_c = []
	for cluster in project_clusters:
		p = p + 1
		if set(req_cluster).issubset(cluster):
			cluster_id = p
			f_c = cluster
	te_col_name = settings.get('Potential_CFD', 'testPrefix') + str(cluster_id)
	print(cluster_id)
	return cluster_id



chars = ["?", "'s", ">", "<", ",", ":", "'", "''", "--", "`", "``", "...", "", "!", "#", '"', '$', '%', '&', '(', ')', '*', '+', '-', '.', '/', ';', '=', '@', '[', '\\', ']', '^', '_', '{', '}', '|', '~', '\t', '\n', '', 'test', 'case', 'id', 'short', 'description', 'and', 'on', 'if', 'the', 'you', 'of', 'is', 'which', 'what', 'this', 'why', 'during', 'at', 'are', 'to', 'in', 'with', 'for', 'cc', 'email', 'from', 'subject', 'a', 'that', 'yet', 'so', 'raise', 'or', 'then', 'there', 're', 'thanks', 'i', 'as', 'me', 'am', 'attaching', 'thread', 'file', 'along', 'files', 'was', 'it', 'n', 'do', 'does', 'well', 'com', 'object']
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
    parser.add_argument("--train", default=True, help ='True/False', type=str, metavar='t')
    parser.add_argument("--cluster", default="", help ='Comma seperated cluster-ids', type=str, metavar='c')
    parser.add_argument("--viewID", default="", help ='View ID', type=str, metavar='v')
    parser.add_argument("--queryID", default="", help ='Query ID', type=str, metavar='q')
    args = parser.parse_args()
    
    return args

#Loading the training data
def load_data(db, collection, is_train):
	cursor = collection.find({})
	df =  pd.DataFrame(list(cursor))
	#majority = df[df["is_duplicate"] == 0]
	#minority = df[df["is_duplicate"] == 1]
	#majority = majority.sample(n=len(minority)*1)
	#df = majority
	#df = df.append(minority)
	df['wrdEmbSet'] = df["Headline"].astype(str) + " " + df["ENCL-Description"].astype(str)
	df['dup_wrdEmbSet'] = df["DUP_Headline"].astype(str) + " " + df["DUP_ENCL-Description"].astype(str)
	if(is_train):
		return df
	else:
		return df[df['SUBMITTED_DATE'] >= str('2018-01')]


def load_data_cluster(db, cluster_id):
	#Getting the clusters data
	collection = db[settings.get('Potential_CFD', 'proj_cluster')]
	cursor = collection.find({})
	clusters =  pd.DataFrame(list(cursor))
	project_clusters = []
	cluster_status = True
	groups = clusters.groupby('Cluster')

	for name, group in groups:
		project_clusters.append(list(group['Project']))

	print(project_clusters)

	cluster = project_clusters[cluster_id - 1]

	df = pd.DataFrame()
	#Fetching the data for each project in the cluster
	for proj in cluster:
		collection = db[settings.get('Potential_CFD', 'trainPrefix')+ proj.replace('.', '_')]
		cursor = collection.find({}) 
		print(proj)
		df2 =  pd.DataFrame(list(cursor))
		df = df.append(df2)
	
	#print(df['PROJECT'].unique())
	return df


def build_data_cat(df, is_train):
	#new_feature_columns = ['DE_MANAGER_USERID2', 'SEVERITY_CODE2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'FEATURE2', 'RELEASE_NOTE2', 'SA_ATTACHMENT_INDIC2', 'CR_ATTACHMENT_INDIC2', 'UT_ATTACHMENT_INDIC2', 'IMPACT2', 'ORIGIN2', 'IS_CUSTOMER_VISIBLE2', 'TICKETS_COUNT2', 'INCOMING_INDIC2', 'BACKLOG_INDIC2', 'DISPOSED_INDIC2', 'TS_INDIC2', 'SS_INDIC2', 'OIB_INDIC2', 'STATE_ASSIGN_INDIC2', 'STATE_CLOSE_INDIC2', 'STATE_FORWARD_INDIC2', 'STATE_HELD_INDIC2', 'STATE_INFO_INDIC2', 'STATE_JUNK_INDIC2', 'STATE_MORE_INDIC2', 'STATE_NEW_INDIC2', 'STATE_OPEN_INDIC2', 'STATE_POSTPONE_INDIC2', 'STATE_RESOLVE_INDIC2', 'STATE_SUBMIT_INDIC2', 'STATE_UNREP_INDIC2', 'STATE_VERIFY_INDIC2', 'STATE_WAIT_INDIC2', 'CFR_INDIC2', 'S12RD_INDIC2', 'S123RD_INDIC2', 'MISSING_SS_EVAL_INDIC2', 'S123_INDIC2', 'S12_INDIC2', 'RNE_INDIC2', 'UPDATED_BY2', 'DEV_ESCAPE_ACTIVITY2', 'RELEASED_CODE2', 'TEST_EDP_ACTIVITY2', 'TEST_EDP_PHASE2', 'RESOLVER_ANALYSIS_INDIC2', 'SUBMITTER_ANALYSIS_INDIC2', 'EDP_ANALYSIS_INDIC2', 'RETI_ANALYSIS_INDIC2', 'DESIGN_REVIEW_ESCAPE_INDIC2', 'STATIC_ANALYSIS_ESCAPE_INDIC2', 'FUNC_TEST_ESCAPE_INDIC2', 'SELECT_REG_ESCAPE_INDIC2', 'CODE_REVIEW_ESCAPE_INDIC2', 'UNIT_TEST_ESCAPE_INDIC2', 'DEV_ESCAPE_INDIC2', 'FEATURE_TEST_ESCAPE_INDIC2', 'REG_TEST_ESCAPE_INDIC2', 'SYSTEM_TEST_ESCAPE_INDIC2', 'SOLUTION_TEST_ESCAPE_INDIC2', 'INT_TEST_ESCAPE_INDIC2', 'GO_TEST_ESCAPE_INDIC2', 'COMPLETE_ESCAPE_INDIC2', 'PSIRT_INDIC2', 'BADCODEFLAG2',  'RISK_OWNER2', 'SIR2', 'PSIRT_FLAG2', 'URC_DISPOSED_INDIC2', 'CLOSED_DISPOSED_INDIC2', 'REGRESSION_BUG_FLAG2']
	#feature_columns_to_use = ['DE_MANAGER_USERID', 'SEVERITY_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'RELEASE_NOTE', 'SA_ATTACHMENT_INDIC', 'CR_ATTACHMENT_INDIC', 'UT_ATTACHMENT_INDIC', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'TICKETS_COUNT', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'TS_INDIC', 'SS_INDIC', 'OIB_INDIC', 'STATE_ASSIGN_INDIC', 'STATE_CLOSE_INDIC', 'STATE_FORWARD_INDIC', 'STATE_HELD_INDIC', 'STATE_INFO_INDIC', 'STATE_JUNK_INDIC', 'STATE_MORE_INDIC', 'STATE_NEW_INDIC', 'STATE_OPEN_INDIC', 'STATE_POSTPONE_INDIC', 'STATE_RESOLVE_INDIC', 'STATE_SUBMIT_INDIC', 'STATE_UNREP_INDIC', 'STATE_VERIFY_INDIC', 'STATE_WAIT_INDIC', 'CFR_INDIC', 'S12RD_INDIC', 'S123RD_INDIC', 'MISSING_SS_EVAL_INDIC', 'S123_INDIC', 'S12_INDIC', 'RNE_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY',  'TEST_EDP_PHASE', 'RESOLVER_ANALYSIS_INDIC', 'SUBMITTER_ANALYSIS_INDIC', 'EDP_ANALYSIS_INDIC', 'RETI_ANALYSIS_INDIC', 'DESIGN_REVIEW_ESCAPE_INDIC', 'STATIC_ANALYSIS_ESCAPE_INDIC', 'FUNC_TEST_ESCAPE_INDIC', 'SELECT_REG_ESCAPE_INDIC', 'CODE_REVIEW_ESCAPE_INDIC', 'UNIT_TEST_ESCAPE_INDIC', 'DEV_ESCAPE_INDIC', 'FEATURE_TEST_ESCAPE_INDIC', 'REG_TEST_ESCAPE_INDIC', 'SYSTEM_TEST_ESCAPE_INDIC', 'SOLUTION_TEST_ESCAPE_INDIC', 'INT_TEST_ESCAPE_INDIC', 'GO_TEST_ESCAPE_INDIC', 'COMPLETE_ESCAPE_INDIC', 'PSIRT_INDIC',  'BADCODEFLAG', 'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'URC_DISPOSED_INDIC', 'CLOSED_DISPOSED_INDIC', 'REGRESSION_BUG_FLAG']
	#new_feature_columns = ['DE_MANAGER_USERID2', 'SEVERITY_CODE2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'FEATURE2', 'SA_ATTACHMENT_INDIC2', 'CR_ATTACHMENT_INDIC2', 'UT_ATTACHMENT_INDIC2', 'IMPACT2', 'ORIGIN2', 'IS_CUSTOMER_VISIBLE2', 'INCOMING_INDIC2', 'BACKLOG_INDIC2', 'DISPOSED_INDIC2', 'TS_INDIC2', 'SS_INDIC2', 'OIB_INDIC2', 'STATE_ASSIGN_INDIC2', 'STATE_CLOSE_INDIC2', 'STATE_FORWARD_INDIC2', 'STATE_HELD_INDIC2', 'STATE_INFO_INDIC2', 'STATE_JUNK_INDIC2', 'STATE_MORE_INDIC2', 'STATE_NEW_INDIC2', 'STATE_OPEN_INDIC2', 'STATE_POSTPONE_INDIC2', 'STATE_RESOLVE_INDIC2', 'STATE_SUBMIT_INDIC2', 'STATE_UNREP_INDIC2', 'STATE_VERIFY_INDIC2', 'STATE_WAIT_INDIC2', 'CFR_INDIC2', 'S12RD_INDIC2', 'S123RD_INDIC2', 'MISSING_SS_EVAL_INDIC2', 'S123_INDIC2', 'S12_INDIC2', 'RNE_INDIC2', 'UPDATED_BY2', 'DEV_ESCAPE_ACTIVITY2', 'RELEASED_CODE2', 'TEST_EDP_ACTIVITY2', 'TEST_EDP_PHASE2', 'RESOLVER_ANALYSIS_INDIC2', 'SUBMITTER_ANALYSIS_INDIC2', 'EDP_ANALYSIS_INDIC2', 'RETI_ANALYSIS_INDIC2', 'DESIGN_REVIEW_ESCAPE_INDIC2', 'STATIC_ANALYSIS_ESCAPE_INDIC2', 'FUNC_TEST_ESCAPE_INDIC2', 'SELECT_REG_ESCAPE_INDIC2', 'CODE_REVIEW_ESCAPE_INDIC2', 'UNIT_TEST_ESCAPE_INDIC2', 'DEV_ESCAPE_INDIC2', 'FEATURE_TEST_ESCAPE_INDIC2', 'REG_TEST_ESCAPE_INDIC2', 'SYSTEM_TEST_ESCAPE_INDIC2', 'SOLUTION_TEST_ESCAPE_INDIC2', 'INT_TEST_ESCAPE_INDIC2', 'GO_TEST_ESCAPE_INDIC2', 'COMPLETE_ESCAPE_INDIC2', 'PSIRT_INDIC2', 'BADCODEFLAG2',  'RISK_OWNER2', 'PSIRT_FLAG2', 'URC_DISPOSED_INDIC2', 'CLOSED_DISPOSED_INDIC2', 'REGRESSION_BUG_FLAG2']
	#feature_columns_to_use = ['DE_MANAGER_USERID', 'SEVERITY_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'SA_ATTACHMENT_INDIC', 'CR_ATTACHMENT_INDIC', 'UT_ATTACHMENT_INDIC', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'TS_INDIC', 'SS_INDIC', 'OIB_INDIC', 'STATE_ASSIGN_INDIC', 'STATE_CLOSE_INDIC', 'STATE_FORWARD_INDIC', 'STATE_HELD_INDIC', 'STATE_INFO_INDIC', 'STATE_JUNK_INDIC', 'STATE_MORE_INDIC', 'STATE_NEW_INDIC', 'STATE_OPEN_INDIC', 'STATE_POSTPONE_INDIC', 'STATE_RESOLVE_INDIC', 'STATE_SUBMIT_INDIC', 'STATE_UNREP_INDIC', 'STATE_VERIFY_INDIC', 'STATE_WAIT_INDIC', 'CFR_INDIC', 'S12RD_INDIC', 'S123RD_INDIC', 'MISSING_SS_EVAL_INDIC', 'S123_INDIC', 'S12_INDIC', 'RNE_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY',  'TEST_EDP_PHASE', 'RESOLVER_ANALYSIS_INDIC', 'SUBMITTER_ANALYSIS_INDIC', 'EDP_ANALYSIS_INDIC', 'RETI_ANALYSIS_INDIC', 'DESIGN_REVIEW_ESCAPE_INDIC', 'STATIC_ANALYSIS_ESCAPE_INDIC', 'FUNC_TEST_ESCAPE_INDIC', 'SELECT_REG_ESCAPE_INDIC', 'CODE_REVIEW_ESCAPE_INDIC', 'UNIT_TEST_ESCAPE_INDIC', 'DEV_ESCAPE_INDIC', 'FEATURE_TEST_ESCAPE_INDIC', 'REG_TEST_ESCAPE_INDIC', 'SYSTEM_TEST_ESCAPE_INDIC', 'SOLUTION_TEST_ESCAPE_INDIC', 'INT_TEST_ESCAPE_INDIC', 'GO_TEST_ESCAPE_INDIC', 'COMPLETE_ESCAPE_INDIC', 'PSIRT_INDIC',  'BADCODEFLAG', 'RISK_OWNER', 'PSIRT_FLAG', 'URC_DISPOSED_INDIC', 'CLOSED_DISPOSED_INDIC', 'REGRESSION_BUG_FLAG']
	
	
	#new_feature_columns = ['DE_MANAGER_USERID2', 'SEVERITY_CODE2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'FEATURE2', 'IMPACT2', 'ORIGIN2', 'RISK_OWNER2'] #Works the best
	#feature_columns_to_use = ['DE_MANAGER_USERID', 'SEVERITY_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'IMPACT', 'ORIGIN', 'RISK_OWNER'] #Works the best
	new_feature_columns = ['DUP_PROJECT', 'DUP_PRODUCT', 'DUP_COMPONENT']
	feature_columns_to_use = ['PROJECT', 'PRODUCT', 'COMPONENT']

	
	#AGE, SR_CNT, LIFECYCLE_STATE_CODE -- removed, (DE_MANAGER, COMPONENT, PROJECT, PRODUCT, FEATURE, IMPACT, ORIGIN, UPDATED_BY) -- for these just check if they are same or not 1/0.
	#avoid_cols = ['DE_MANAGER_USERID', 'COMPONENT', 'PROJECT', 'PRODUCT', 'FEATURE', 'IMPACT', 'ORIGIN','RISK_OWNER', 'FEATURE'] #Works the best

	avoid_cols = ['COMPONENT', 'PROJECT', 'PRODUCT']

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

	cols.append('is_duplicate')
	big_X = df[cols]

	#nonnumeric_columns_1 = ['DE_MANAGER_USERID', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'RELEASE_NOTE', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'TICKETS_COUNT', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY', 'TEST_EDP_PHASE', 'BADCODEFLAG',  'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'REGRESSION_BUG_FLAG']
	#nonnumeric_columns_2 = ['DE_MANAGER_USERID2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'FEATURE2', 'RELEASE_NOTE2', 'IMPACT2', 'ORIGIN2', 'IS_CUSTOMER_VISIBLE2', 'TICKETS_COUNT2', 'INCOMING_INDIC2', 'BACKLOG_INDIC2', 'DISPOSED_INDIC2', 'UPDATED_BY2', 'DEV_ESCAPE_ACTIVITY2', 'RELEASED_CODE2', 'TEST_EDP_ACTIVITY2', 'TEST_EDP_PHASE2', 'BADCODEFLAG2', 'RISK_OWNER2', 'SIR2', 'PSIRT_FLAG2', 'REGRESSION_BUG_FLAG2']
	#nonnumeric_columns_1 = ['DE_MANAGER_USERID', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY', 'TEST_EDP_PHASE', 'BADCODEFLAG',  'RISK_OWNER', 'PSIRT_FLAG', 'REGRESSION_BUG_FLAG']
	#nonnumeric_columns_2 = ['DE_MANAGER_USERID2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'FEATURE2', 'IMPACT2', 'ORIGIN2', 'IS_CUSTOMER_VISIBLE2', 'INCOMING_INDIC2', 'BACKLOG_INDIC2', 'DISPOSED_INDIC2', 'UPDATED_BY2', 'DEV_ESCAPE_ACTIVITY2', 'RELEASED_CODE2', 'TEST_EDP_ACTIVITY2', 'TEST_EDP_PHASE2', 'BADCODEFLAG2', 'RISK_OWNER2', 'PSIRT_FLAG2', 'REGRESSION_BUG_FLAG2']
	
	
	#nonnumeric_columns_1 = ['DE_MANAGER_USERID', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'IMPACT', 'ORIGIN', 'RISK_OWNER']
	#nonnumeric_columns_2 = ['DE_MANAGER_USERID2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'FEATURE2', 'IMPACT2', 'ORIGIN2', 'RISK_OWNER2']
	nonnumeric_columns_1 = ['PROJECT', 'PRODUCT', 'COMPONENT']
	nonnumeric_columns_2 = ['DUP_PROJECT', 'DUP_PRODUCT', 'DUP_COMPONENT']
	
	
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


	train_X = big_X_imputed.iloc[:, :len(cols)-1].as_matrix()
	Y = big_X['is_duplicate']
	#X_train = train_X
	if(is_train == True):
		validation_size = int(df.shape[0]/4)
		X_train, X_validation, Y_train, Y_validation = train_test_split(train_X, Y, test_size=validation_size)
		for i in range(0, len(X_train)):
			for k in avoid_cols_indexes:
				if(X_train[i][k] == X_train[i][k + 1]):
					X_train[i][k] = 1
					X_train[i][k + 1] = 1
				else:
					X_train[i][k] = 0
					X_train[i][k + 1] = 1

		for i in range(0, len(X_validation)):
			for k in avoid_cols_indexes:
				if(X_validation[i][k] == X_validation[i][k + 1]):
					X_validation[i][k] = 1
					X_validation[i][k + 1] = 1
				else:
					X_validation[i][k] = 0
					X_validation[i][k + 1] = 1

		Y_train = Y_train.values
		Y_validation = Y_validation.values
		return X_train, Y_train, X_validation, Y_validation
	
	else:
		X_train = train_X
		for i in range(0, len(X_train)):
			for k in avoid_cols_indexes:
				if(X_train[i][k] == X_train[i][k + 1]):
					X_train[i][k] = 1
					X_train[i][k + 1] = 1
				else:
					X_train[i][k] = 0
					X_train[i][k + 1] = 1

		Y_train = Y.values
		return X_train, Y_train


def build_model_cat(X_train, Y_train, X_validation, Y_validation, cluster):
	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(3458230+1, embedding_vecor_length, input_length=len(X_train[0])))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	#Fitting the data into the model
	model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=2, batch_size=64)
	filename = model_path + 'model3_lstm_cnn_cat_' + str(cluster) + '.h5' #'/data/csap_models/bugDups/model3_lstm_cnn_cat_' + str(cluster) + '.h5'
	model.save(filename) #model.save('/data/csap_models/bugDups/model3_lstm_cnn_cat.h5')
	#prediction = model.predict(X_train)
	return model #, prediction


def build_data_cat_2(df, is_train):
	#new_feature_columns = ['DE_MANAGER_USERID2', 'SEVERITY_CODE2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'FEATURE2', 'RELEASE_NOTE2', 'SA_ATTACHMENT_INDIC2', 'CR_ATTACHMENT_INDIC2', 'UT_ATTACHMENT_INDIC2', 'IMPACT2', 'ORIGIN2', 'IS_CUSTOMER_VISIBLE2', 'TICKETS_COUNT2', 'INCOMING_INDIC2', 'BACKLOG_INDIC2', 'DISPOSED_INDIC2', 'TS_INDIC2', 'SS_INDIC2', 'OIB_INDIC2', 'STATE_ASSIGN_INDIC2', 'STATE_CLOSE_INDIC2', 'STATE_FORWARD_INDIC2', 'STATE_HELD_INDIC2', 'STATE_INFO_INDIC2', 'STATE_JUNK_INDIC2', 'STATE_MORE_INDIC2', 'STATE_NEW_INDIC2', 'STATE_OPEN_INDIC2', 'STATE_POSTPONE_INDIC2', 'STATE_RESOLVE_INDIC2', 'STATE_SUBMIT_INDIC2', 'STATE_UNREP_INDIC2', 'STATE_VERIFY_INDIC2', 'STATE_WAIT_INDIC2', 'CFR_INDIC2', 'S12RD_INDIC2', 'S123RD_INDIC2', 'MISSING_SS_EVAL_INDIC2', 'S123_INDIC2', 'S12_INDIC2', 'RNE_INDIC2', 'UPDATED_BY2', 'DEV_ESCAPE_ACTIVITY2', 'RELEASED_CODE2', 'TEST_EDP_ACTIVITY2', 'TEST_EDP_PHASE2', 'RESOLVER_ANALYSIS_INDIC2', 'SUBMITTER_ANALYSIS_INDIC2', 'EDP_ANALYSIS_INDIC2', 'RETI_ANALYSIS_INDIC2', 'DESIGN_REVIEW_ESCAPE_INDIC2', 'STATIC_ANALYSIS_ESCAPE_INDIC2', 'FUNC_TEST_ESCAPE_INDIC2', 'SELECT_REG_ESCAPE_INDIC2', 'CODE_REVIEW_ESCAPE_INDIC2', 'UNIT_TEST_ESCAPE_INDIC2', 'DEV_ESCAPE_INDIC2', 'FEATURE_TEST_ESCAPE_INDIC2', 'REG_TEST_ESCAPE_INDIC2', 'SYSTEM_TEST_ESCAPE_INDIC2', 'SOLUTION_TEST_ESCAPE_INDIC2', 'INT_TEST_ESCAPE_INDIC2', 'GO_TEST_ESCAPE_INDIC2', 'COMPLETE_ESCAPE_INDIC2', 'PSIRT_INDIC2', 'BADCODEFLAG2',  'RISK_OWNER2', 'SIR2', 'PSIRT_FLAG2', 'URC_DISPOSED_INDIC2', 'CLOSED_DISPOSED_INDIC2', 'REGRESSION_BUG_FLAG2']
	#feature_columns_to_use = ['DE_MANAGER_USERID', 'SEVERITY_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'RELEASE_NOTE', 'SA_ATTACHMENT_INDIC', 'CR_ATTACHMENT_INDIC', 'UT_ATTACHMENT_INDIC', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'TICKETS_COUNT', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'TS_INDIC', 'SS_INDIC', 'OIB_INDIC', 'STATE_ASSIGN_INDIC', 'STATE_CLOSE_INDIC', 'STATE_FORWARD_INDIC', 'STATE_HELD_INDIC', 'STATE_INFO_INDIC', 'STATE_JUNK_INDIC', 'STATE_MORE_INDIC', 'STATE_NEW_INDIC', 'STATE_OPEN_INDIC', 'STATE_POSTPONE_INDIC', 'STATE_RESOLVE_INDIC', 'STATE_SUBMIT_INDIC', 'STATE_UNREP_INDIC', 'STATE_VERIFY_INDIC', 'STATE_WAIT_INDIC', 'CFR_INDIC', 'S12RD_INDIC', 'S123RD_INDIC', 'MISSING_SS_EVAL_INDIC', 'S123_INDIC', 'S12_INDIC', 'RNE_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY',  'TEST_EDP_PHASE', 'RESOLVER_ANALYSIS_INDIC', 'SUBMITTER_ANALYSIS_INDIC', 'EDP_ANALYSIS_INDIC', 'RETI_ANALYSIS_INDIC', 'DESIGN_REVIEW_ESCAPE_INDIC', 'STATIC_ANALYSIS_ESCAPE_INDIC', 'FUNC_TEST_ESCAPE_INDIC', 'SELECT_REG_ESCAPE_INDIC', 'CODE_REVIEW_ESCAPE_INDIC', 'UNIT_TEST_ESCAPE_INDIC', 'DEV_ESCAPE_INDIC', 'FEATURE_TEST_ESCAPE_INDIC', 'REG_TEST_ESCAPE_INDIC', 'SYSTEM_TEST_ESCAPE_INDIC', 'SOLUTION_TEST_ESCAPE_INDIC', 'INT_TEST_ESCAPE_INDIC', 'GO_TEST_ESCAPE_INDIC', 'COMPLETE_ESCAPE_INDIC', 'PSIRT_INDIC',  'BADCODEFLAG', 'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'URC_DISPOSED_INDIC', 'CLOSED_DISPOSED_INDIC', 'REGRESSION_BUG_FLAG']
	#new_feature_columns = ['DE_MANAGER_USERID2', 'SEVERITY_CODE2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'FEATURE2', 'SA_ATTACHMENT_INDIC2', 'CR_ATTACHMENT_INDIC2', 'UT_ATTACHMENT_INDIC2', 'IMPACT2', 'ORIGIN2', 'IS_CUSTOMER_VISIBLE2', 'INCOMING_INDIC2', 'BACKLOG_INDIC2', 'DISPOSED_INDIC2', 'TS_INDIC2', 'SS_INDIC2', 'OIB_INDIC2', 'STATE_ASSIGN_INDIC2', 'STATE_CLOSE_INDIC2', 'STATE_FORWARD_INDIC2', 'STATE_HELD_INDIC2', 'STATE_INFO_INDIC2', 'STATE_JUNK_INDIC2', 'STATE_MORE_INDIC2', 'STATE_NEW_INDIC2', 'STATE_OPEN_INDIC2', 'STATE_POSTPONE_INDIC2', 'STATE_RESOLVE_INDIC2', 'STATE_SUBMIT_INDIC2', 'STATE_UNREP_INDIC2', 'STATE_VERIFY_INDIC2', 'STATE_WAIT_INDIC2', 'CFR_INDIC2', 'S12RD_INDIC2', 'S123RD_INDIC2', 'MISSING_SS_EVAL_INDIC2', 'S123_INDIC2', 'S12_INDIC2', 'RNE_INDIC2', 'UPDATED_BY2', 'DEV_ESCAPE_ACTIVITY2', 'RELEASED_CODE2', 'TEST_EDP_ACTIVITY2', 'TEST_EDP_PHASE2', 'RESOLVER_ANALYSIS_INDIC2', 'SUBMITTER_ANALYSIS_INDIC2', 'EDP_ANALYSIS_INDIC2', 'RETI_ANALYSIS_INDIC2', 'DESIGN_REVIEW_ESCAPE_INDIC2', 'STATIC_ANALYSIS_ESCAPE_INDIC2', 'FUNC_TEST_ESCAPE_INDIC2', 'SELECT_REG_ESCAPE_INDIC2', 'CODE_REVIEW_ESCAPE_INDIC2', 'UNIT_TEST_ESCAPE_INDIC2', 'DEV_ESCAPE_INDIC2', 'FEATURE_TEST_ESCAPE_INDIC2', 'REG_TEST_ESCAPE_INDIC2', 'SYSTEM_TEST_ESCAPE_INDIC2', 'SOLUTION_TEST_ESCAPE_INDIC2', 'INT_TEST_ESCAPE_INDIC2', 'GO_TEST_ESCAPE_INDIC2', 'COMPLETE_ESCAPE_INDIC2', 'PSIRT_INDIC2', 'BADCODEFLAG2',  'RISK_OWNER2', 'PSIRT_FLAG2', 'URC_DISPOSED_INDIC2', 'CLOSED_DISPOSED_INDIC2', 'REGRESSION_BUG_FLAG2']
	#feature_columns_to_use = ['DE_MANAGER_USERID', 'SEVERITY_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'SA_ATTACHMENT_INDIC', 'CR_ATTACHMENT_INDIC', 'UT_ATTACHMENT_INDIC', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'TS_INDIC', 'SS_INDIC', 'OIB_INDIC', 'STATE_ASSIGN_INDIC', 'STATE_CLOSE_INDIC', 'STATE_FORWARD_INDIC', 'STATE_HELD_INDIC', 'STATE_INFO_INDIC', 'STATE_JUNK_INDIC', 'STATE_MORE_INDIC', 'STATE_NEW_INDIC', 'STATE_OPEN_INDIC', 'STATE_POSTPONE_INDIC', 'STATE_RESOLVE_INDIC', 'STATE_SUBMIT_INDIC', 'STATE_UNREP_INDIC', 'STATE_VERIFY_INDIC', 'STATE_WAIT_INDIC', 'CFR_INDIC', 'S12RD_INDIC', 'S123RD_INDIC', 'MISSING_SS_EVAL_INDIC', 'S123_INDIC', 'S12_INDIC', 'RNE_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY',  'TEST_EDP_PHASE', 'RESOLVER_ANALYSIS_INDIC', 'SUBMITTER_ANALYSIS_INDIC', 'EDP_ANALYSIS_INDIC', 'RETI_ANALYSIS_INDIC', 'DESIGN_REVIEW_ESCAPE_INDIC', 'STATIC_ANALYSIS_ESCAPE_INDIC', 'FUNC_TEST_ESCAPE_INDIC', 'SELECT_REG_ESCAPE_INDIC', 'CODE_REVIEW_ESCAPE_INDIC', 'UNIT_TEST_ESCAPE_INDIC', 'DEV_ESCAPE_INDIC', 'FEATURE_TEST_ESCAPE_INDIC', 'REG_TEST_ESCAPE_INDIC', 'SYSTEM_TEST_ESCAPE_INDIC', 'SOLUTION_TEST_ESCAPE_INDIC', 'INT_TEST_ESCAPE_INDIC', 'GO_TEST_ESCAPE_INDIC', 'COMPLETE_ESCAPE_INDIC', 'PSIRT_INDIC',  'BADCODEFLAG', 'RISK_OWNER', 'PSIRT_FLAG', 'URC_DISPOSED_INDIC', 'CLOSED_DISPOSED_INDIC', 'REGRESSION_BUG_FLAG']
	
	
	new_feature_columns = ['DUP_DE_MANAGER_USERID', 'DUP_SEVERITY_CODE', 'DUP_PROJECT', 'DUP_PRODUCT', 'DUP_COMPONENT', 'DUP_FEATURE', 'DUP_IMPACT', 'DUP_ORIGIN', 'DUP_RISK_OWNER'] #Works the best
	feature_columns_to_use = ['DE_MANAGER_USERID', 'SEVERITY_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'IMPACT', 'ORIGIN', 'RISK_OWNER'] #Works the best
	#new_feature_columns = ['PROJECT2', 'PRODUCT2', 'COMPONENT2']
	#feature_columns_to_use = ['PROJECT', 'PRODUCT', 'COMPONENT']

	
	#AGE, SR_CNT, LIFECYCLE_STATE_CODE -- removed, (DE_MANAGER, COMPONENT, PROJECT, PRODUCT, FEATURE, IMPACT, ORIGIN, UPDATED_BY) -- for these just check if they are same or not 1/0.
	avoid_cols = ['DE_MANAGER_USERID', 'COMPONENT', 'PROJECT', 'PRODUCT', 'FEATURE', 'IMPACT', 'ORIGIN','RISK_OWNER', 'FEATURE'] #Works the best

	#avoid_cols = ['COMPONENT', 'PROJECT', 'PRODUCT']

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

	cols.append('is_duplicate')
	big_X = df[cols]

	#nonnumeric_columns_1 = ['DE_MANAGER_USERID', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'RELEASE_NOTE', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'TICKETS_COUNT', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY', 'TEST_EDP_PHASE', 'BADCODEFLAG',  'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'REGRESSION_BUG_FLAG']
	#nonnumeric_columns_2 = ['DE_MANAGER_USERID2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'FEATURE2', 'RELEASE_NOTE2', 'IMPACT2', 'ORIGIN2', 'IS_CUSTOMER_VISIBLE2', 'TICKETS_COUNT2', 'INCOMING_INDIC2', 'BACKLOG_INDIC2', 'DISPOSED_INDIC2', 'UPDATED_BY2', 'DEV_ESCAPE_ACTIVITY2', 'RELEASED_CODE2', 'TEST_EDP_ACTIVITY2', 'TEST_EDP_PHASE2', 'BADCODEFLAG2', 'RISK_OWNER2', 'SIR2', 'PSIRT_FLAG2', 'REGRESSION_BUG_FLAG2']
	#nonnumeric_columns_1 = ['DE_MANAGER_USERID', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY', 'TEST_EDP_PHASE', 'BADCODEFLAG',  'RISK_OWNER', 'PSIRT_FLAG', 'REGRESSION_BUG_FLAG']
	#nonnumeric_columns_2 = ['DE_MANAGER_USERID2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'FEATURE2', 'IMPACT2', 'ORIGIN2', 'IS_CUSTOMER_VISIBLE2', 'INCOMING_INDIC2', 'BACKLOG_INDIC2', 'DISPOSED_INDIC2', 'UPDATED_BY2', 'DEV_ESCAPE_ACTIVITY2', 'RELEASED_CODE2', 'TEST_EDP_ACTIVITY2', 'TEST_EDP_PHASE2', 'BADCODEFLAG2', 'RISK_OWNER2', 'PSIRT_FLAG2', 'REGRESSION_BUG_FLAG2']
	
	
	nonnumeric_columns_1 = ['DE_MANAGER_USERID', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'IMPACT', 'ORIGIN', 'RISK_OWNER']
	nonnumeric_columns_2 = ['DUP_DE_MANAGER_USERID', 'DUP_PROJECT', 'DUP_PRODUCT', 'DUP_COMPONENT', 'DUP_FEATURE', 'DUP_IMPACT', 'DUP_ORIGIN', 'DUP_RISK_OWNER']
	#nonnumeric_columns_1 = ['PROJECT', 'PRODUCT', 'COMPONENT']
	#nonnumeric_columns_2 = ['PROJECT2', 'PRODUCT2', 'COMPONENT2']
	
	
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


	train_X = big_X_imputed.iloc[:, :len(cols)-1].as_matrix()
	Y = big_X['is_duplicate']
	#X_train = train_X
	if(is_train == True):
		validation_size = int(df.shape[0]/4)
		X_train, X_validation, Y_train, Y_validation = train_test_split(train_X, Y, test_size=validation_size)
		for i in range(0, len(X_train)):
			for k in avoid_cols_indexes:
				if(X_train[i][k] == X_train[i][k + 1]):
					X_train[i][k] = 1
					X_train[i][k + 1] = 1
				else:
					X_train[i][k] = 0
					X_train[i][k + 1] = 1

		for i in range(0, len(X_validation)):
			for k in avoid_cols_indexes:
				if(X_validation[i][k] == X_validation[i][k + 1]):
					X_validation[i][k] = 1
					X_validation[i][k + 1] = 1
				else:
					X_validation[i][k] = 0
					X_validation[i][k + 1] = 1

		Y_train = Y_train.values
		Y_validation = Y_validation.values
		return X_train, Y_train, X_validation, Y_validation
	
	else:
		X_train = train_X
		for i in range(0, len(X_train)):
			for k in avoid_cols_indexes:
				if(X_train[i][k] == X_train[i][k + 1]):
					X_train[i][k] = 1
					X_train[i][k + 1] = 1
				else:
					X_train[i][k] = 0
					X_train[i][k + 1] = 1

		Y_train = Y.values
		return X_train, Y_train


def build_model_cat_2(X_train, Y_train, X_validation, Y_validation, cluster):
	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(3458230+1, embedding_vecor_length, input_length=len(X_train[0])))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	#Fitting the data into the model
	model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=2, batch_size=64)
	filename = model_path + 'usecase2/model3_lstm_cnn_cat_' + str(cluster) + '.h5' #'/data/csap_models/bugDups/usecase2/model3_lstm_cnn_cat_' + str(cluster) + '.h5'
	model.save(filename) #model.save('/data/csap_models/bugDups/model3_lstm_cnn_cat.h5')
	#prediction = model.predict(X_train)
	return model #, prediction


def test_model_cat(X_test, Y_test, cluster):
	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(3458230+1, embedding_vecor_length, input_length=len(X_test[0])))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	#print(model.summary())
	#Model predictions
	filename = model_path + 'model3_lstm_cnn_cat_' + str(cluster) + '.h5' #'/data/csap_models/bugDups/model3_lstm_cnn_cat_' + str(cluster) + '.h5'
	model = load_model(filename) #model = load_model('/data/csap_models/bugDups/model3_lstm_cnn_cat.h5')
	prediction = model.predict(X_test)
	print(prediction, Y_test)
	return prediction

def test_model_cat_2(X_test, Y_test, cluster):
	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(3458230+1, embedding_vecor_length, input_length=len(X_test[0])))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	#print(model.summary())
	#Model predictions
	filename = model_path + 'usecase2/model3_lstm_cnn_cat_' + str(cluster) + '.h5' #'/data/csap_models/bugDups/usecase2/model3_lstm_cnn_cat_' + str(cluster) + '.h5'
	model = load_model(filename) #model = load_model('/data/csap_models/bugDups/model3_lstm_cnn_cat.h5')
	prediction = model.predict(X_test)
	print(prediction, Y_test)
	return prediction


def build_data_text(train_df, cluster, db):
	# Prepare embedding
	vocabulary = dict()
	inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
	print("Starting the W2V")
	#word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True) #WE NEED TO HAVE OUR OWN W2V MODEL
	#Our own word2vec model
	sentences = []
	for dataset in [train_df]:
		for index, row in dataset.iterrows():
			#print(text_to_word_list(row["wrdEmbSet"]))
			#print(index)
			if(type(row['wrdEmbSet']) != float):
				sentences.append(text_to_word_list(row["wrdEmbSet"]))
			if(type(row['dup_wrdEmbSet']) != float):
				sentences.append(text_to_word_list(row["dup_wrdEmbSet"]))
	# train model
	model = Word2Vec(sentences, min_count=50, size = embedding_dim)
	# summarize the loaded model
	print(model)
	# summarize vocabulary
	words = list(model.wv.vocab)
	# save model
	filename = model_path + 'w2vmodel_' + str(cluster) + '.bin' #'/data/csap_models/bugDups/w2vmodel_' + str(cluster) + '.bin'
	model.save(filename)
	print("built word2vec")
	c = 0
	q1 =[0]*(train_df.shape[0])
	q2 = [0]*(train_df.shape[0])
	# Iterate over the questions only of both training and test datasets
	for dataset in [train_df]:#[train_df, test_df]:
		for index, row in dataset.iterrows():
			#print(index)
			# Iterate through the text of both questions of the row
			for question in text_cols:
				c = c + 1
				q2n = []  # q2n -> question numbers representation
				for word in text_to_word_list(row[question]):
					# Check for unwanted words
					if word.lower() in stops and word.lower() not in words: #word2vec.vocab:
						#print(word)
						continue
					if word not in vocabulary:
						vocabulary[word] = len(inverse_vocabulary)
						q2n.append(len(inverse_vocabulary))
						inverse_vocabulary.append(word)
					else:
						q2n.append(vocabulary[word])
				# Replace questions as word to question as number representation
				if(c%2 != 0):
					 q1[index] = q2n
				else:
					 q2[index] = q2n
				#dataset.set_value(index, question, q2n)
	print("Build vocab and sentences")
	train_df['wrdEmbSet'] = q1
	train_df['dup_wrdEmbSet'] = q2
	#Write the vocabulary and inverse vocabulary into a file
	f = model_path + 'vocab_model_' + str(cluster) + '.json' #'/data/csap_models/bugDups/vocab_model_' + str(cluster) + '.json'
	#f = "/users/sumreddi/vocab_model_3.json"
	f = open(f,'w')
	json1 = json.dumps(vocabulary, indent=4)
	f.write(json1)
	f.close()
	thefile = model_path + 'inv_vocab_model_' + str(cluster) + '.json' #'/data/csap_models/bugDups/inv_vocab_model_' + str(cluster) + '.json'
	#thefile = "/users/sumreddi/inv_vocab_model_3.json" #str(settings.get("Potential_CFD","temp_path_mod_bugDuplicates")) + '/top_words_cluster_' +str(p)+'.txt'
	with open(thefile, 'wb') as fp:
		pickle.dump(inverse_vocabulary, fp, protocol=2)
	#Store train_df in a collection
	#train_df.to_csv('/data/csap_models/bugDups/Train_csv_3_complete.csv', encoding='utf-8')
	c = train_prefix + str(cluster) + "_complete" #"BugDupsTrainSet_" + str(cluster) + "_complete"
	collection = db[c]
	#train_df.reset_index(drop = True, inplace = True)
	t_df = train_df[['is_duplicate', 'SUBMITTED_DATE', 'IDENTIFIER', 'DUPLICATE_OF', 'Headline', 'ENCL-Description', 'DUP_Headline', 'DUP_ENCL-Description', 'wrdEmbSet', 'dup_wrdEmbSet', 'PRODUCT', 'DUP_PRODUCT', 'PROJECT', 'DUP_PROJECT', 'COMPONENT', 'DUP_COMPONENT']]
	#t_df['complete1'] = t_df['complete1'].astype(str)
	#t_df['complete2'] = t_df['complete2'].astype(str)
	#records = json2.loads(t_df.T.to_json(date_format='iso')).values()
	'''
	d = pd.DataFrame()
	d['IDENTIFIER'] = t_df['IDENTIFIER']
	d['wrdEmbSet'] = t_df['wrdEmbSet']
	d['DUPLICATE_OF'] = t_df['DUPLICATE_OF']
	d['dup_wrdEmbSet'] = t_df['dup_wrdEmbSet']
	d['DUP_Headline'] = t_df['DUP_Headline']
	d['Headline'] = t_df['Headline']
	d['ENCL-Description'] = t_df['ENCL-Description']
	d['DUP_ENCL-Description'] = t_df['DUP_ENCL-Description']
	d['is_duplicate'] = t_df['is_duplicate']
	d['PRODUCT'] = t_df['PRODUCT']
	d['DUP_PRODUCT'] = t_df['DUP_PRODUCT']
	d['PROJECT'] = t_df['PROJECT']
	d['DUP_PROJECT'] = t_df['DUP_PROJECT']
	d['COMPONENT'] = t_df['COMPONENT']
	d['DUP_COMPONENT'] = t_df['DUP_COMPONENT']
	d['SUBMITTED_DATE'] = t_df['SUBMITTED_DATE']
	records = json.loads(d.T.to_json()).values()
	collection.drop()
	collection.create_index([("IDENTIFIER", pymongo.ASCENDING), ("DUPLICATE_OF", pymongo.ASCENDING)], unique=True)
	collection.insert(records)
	'''
	return vocabulary, model, train_df


def build_model_text(vocabulary, model, validation_size, train_df, cluster):
	validation_size = int(train_df.shape[0]/3)
	words = list(model.wv.vocab)
	embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
	embeddings[0] = 0  # So that the padding will be ignored
	print("building embeddings")
	# Build the embedding matrix
	for word, index in vocabulary.items():
		#print(index, len(vocabulary))
		if word in words:#word2vec.vocab:
			embeddings[index] = model[word]#word2vec.word_vec(word)

	thefile = model_path + "embeddings_model_" + str(cluster)+'.json' #"/data/csap_models/bugDups/embeddings_model_" + str(cluster)+'.json'
	with open(thefile, 'wb') as fp:
		pickle.dump(embeddings, fp, protocol=2)

	print("completed embeddings")
	#validation_size = 20000
	training_size = len(train_df) - validation_size

	X = train_df[text_cols]
	Y = train_df['is_duplicate']

	X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

	# Split to dicts
	X_train = {'left': X_train.wrdEmbSet, 'right': X_train.dup_wrdEmbSet}
	X_validation = {'left': X_validation.wrdEmbSet, 'right': X_validation.dup_wrdEmbSet}
	#X_test = {'left': test_df.complete1, 'right': test_df.complete2}

	# Convert labels to their numpy representations
	Y_train = Y_train.values
	Y_validation = Y_validation.values

	# Zero padding
	for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
		#print(dataset)
		dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

	# Model variables
	n_hidden = 50
	gradient_clipping_norm = 1.25
	batch_size = 64
	n_epoch = 18

	print("Building the model")

	def exponent_neg_manhattan_distance(left, right):
		''' Helper function for the similarity estimate of the LSTMs outputs'''
		return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

	# The visible layer
	left_input = Input(shape=(max_seq_length,), dtype='int32')
	right_input = Input(shape=(max_seq_length,), dtype='int32')

	embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)

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

	# Start training
	training_start_time = time()

	print("Starting to train")

	malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
								validation_data=([X_validation['left'], X_validation['right']], Y_validation))
	print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))

	#Active learning
	predictions = malstm.predict([X_validation['left'], X_validation['right']])
	predictions = [item for sublist in predictions for item in sublist]
	round_predictions = [round(x) for x in predictions]
	a = (Y_validation == 1)
	#b = [False]*len(round_predictions)
	b = [True if value == 1 else False for index, value in enumerate(round_predictions)]
	#b = (round_predictions == 1)
	sum(a & b)

	comb_predictions = (Y_validation == round_predictions)

	wrong_predictions = [index for index, value in enumerate(comb_predictions) if value == False]
	yes_indexes = [index for index, value in enumerate(Y_train) if value == 1]
	yes = list(set(wrong_predictions).intersection(yes_indexes))


	left = X_train['left'][yes]
	right = X_train['right'][yes]
	labels = [1]*len(left)

	malstm_trained = malstm.fit([left, right], labels, batch_size=batch_size, nb_epoch=1,
								validation_data=([X_validation['left'], X_validation['right']], Y_validation))
	'''
	no_indexes = [index for index, value in enumerate(Y_train) if value == 0]
	no = list(set(wrong_predictions).intersection(no_indexes))

	left = X_train['left'][no]
	right = X_train['right'][no]
	labels = [0]*len(left)

	malstm_trained = malstm.fit([left, right], labels, batch_size=batch_size, nb_epoch=1,
								validation_data=([X_validation['left'], X_validation['right']], Y_validation))
	'''
	filename = model_path + 'text_model_' + str(cluster) + '.h5' #'/data/csap_models/bugDups/text_model_' + str(cluster) + '.h5'
	malstm.save_weights(filename)
	print("Saved model to disk")

	return malstm


def build_test_data_text(test_df, cluster):
	filename = model_path + '/w2vmodel_' + str(cluster) + '.bin' #'/data/csap_models/bugDups/w2vmodel_' + str(cluster) + '.bin'
	model = Word2Vec.load(filename)
	print("Loaded the W2V")
	words = list(model.wv.vocab)
	f = model_path + 'vocab_model_' + str(cluster) + '.json' #'/data/csap_models/bugDups/vocab_model_' + str(cluster) + '.json'
	vocabulary = json.load(open(f, 'r'))
	thefile = model_path + 'inv_vocab_model_' + str(cluster) + '.json' #'/data/csap_models/bugDups/inv_vocab_model_' + str(cluster) + '.json'
	with open (thefile, 'rb') as fp:
		inverse_vocabulary = pickle.load(fp)
	c=0
	q1=[0]*(test_df.shape[0])
	q2=[0]*(test_df.shape[0])
	a = -1
	s = 0
	for dataset in [test_df]:
		for index, row in dataset.iterrows():
			print(index)
			a = a+1
			for question in text_cols:
				c = c + 1
				q2n = []  # q2n -> question numbers representation
				for word in text_to_word_list(row[question]):
					# Check for unwanted words
					if word.lower() in stops and word.lower() not in words: #word2vec.vocab:
						continue
					if word not in vocabulary:
						s = 1
					else:
						q2n.append(vocabulary[word])
				if(c%2 != 0):
					 q1[a] = q2n
				else:
					 q2[a] = q2n
				#dataset.set_value(index, question, q2n)
	test_df['wrdEmbSet'] = q1
	test_df['dup_wrdEmbSet'] = q2
	return vocabulary, model, words, test_df


def test_model_text(vocabulary, model, words, test_df, cluster):
	embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
	embeddings[0] = 0  # So that the padding will be ignored
	'''
	# Build the embedding matrix
	for word, index in vocabulary.items():
		#print(index)
		if word in words:#word2vec.vocab:
			embeddings[index] = model[word]#word2vec.word_vec(word)
	'''
	thefile = model_path + "embeddings_model_" + str(cluster)+'.json' #"/data/csap_models/bugDups/embeddings_model_" + str(cluster)+'.json'
	with open (thefile, 'rb') as fp:
		embeddings2 = pickle.load(fp)

	del model

	X = test_df[text_cols]
	Y = test_df['is_duplicate']
	X_test = X
	X_test = {'left': X_test.wrdEmbSet, 'right': X_test.dup_wrdEmbSet}

	Y_train = Y.values

	for dataset, side in itertools.product([X_test], ['left', 'right']):
		dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

	n_hidden = 50
	gradient_clipping_norm = 1.25
	batch_size = 64
	n_epoch = 1 #25
	def exponent_neg_manhattan_distance(left, right):
		return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

	left_input = Input(shape=(max_seq_length,), dtype='int32')
	right_input = Input(shape=(max_seq_length,), dtype='int32')

	embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)

	encoded_left = embedding_layer(left_input)
	encoded_right = embedding_layer(right_input)

	shared_lstm = LSTM(n_hidden)

	left_output = shared_lstm(encoded_left)
	right_output = shared_lstm(encoded_right)

	malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

	malstm = Model([left_input, right_input], [malstm_distance])

	optimizer = Adadelta(clipnorm=gradient_clipping_norm)

	malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

	# load weights into new model
	filename = model_path + 'text_model_' + str(cluster) + '.h5' #'/data/csap_models/bugDups/text_model_' + str(cluster) + '.h5'
	malstm.load_weights(filename)
	print("Loaded model from disk")

	# evaluate loaded model on test data
	predictions = malstm.predict([X_test['left'], X_test['right']])

	return predictions


def main():
	options = parse_options()
	if(options.env.lower() == "prod"):
		key = "csap_prod_database"

	elif(options.env.lower() == "stage"):
		key = "csap_stage_database"

	db = get_db(settings, key)

	if(options.train.lower() == "true"):
		for cluster_id in range(1, 5):
			coll_name = train_prefix + str(cluster_id) #str(options.cluster) #"BugDupsTrainSet_" + str(options.cluster)
			collection = db[coll_name] #db['BugDupsTrainSet_all_3']

			print(collection)
			df = load_data(db, collection, True)

			#X_train, Y_train, X_validation, Y_validation = build_data_cat(df, True)
			#print("done cat 1")
			#cat_model = build_model_cat(X_train, Y_train, X_validation, Y_validation, int(options.cluster))
			#print("done model 1")
			#X_train_2, Y_train_2, X_validation_2, Y_validation_2 = build_data_cat_2(df, True)
			#print("done cat 2")
			#cat_model_2 = build_model_cat_2(X_train_2, Y_train_2, X_validation_2, Y_validation_2, int(options.cluster))
			#print("done model 2")
			vocabulary, w2vmodel, train_df = build_data_text(df, int(cluster_id), db)
			print("done vocab")
			model = build_model_text(vocabulary, w2vmodel, 500, train_df, int(cluster_id))
			print("done text model")
	'''
	else:
		coll_name = "BugDupsTrainSet_" + str(options.cluster)
		collection = db[coll_name] #db['BugDupsTrainSet_all_3']
		print(collection)
		df = load_data(db, collection, False)
		cluster_id = int(options.cluster)
		print(cluster_id)

		X_test, Y_test = build_data_cat(df, False)
		cat_predictions = test_model_cat(X_test, Y_test, cluster_id)
		vocabulary, w2vmodel, words, test_df = build_test_data_text(df, cluster_id)
		text_predictions = test_model_text(vocabulary, w2vmodel, words, test_df, cluster_id)
		print(text_predictions, cat_predictions)

		d = pd.DataFrame()
		d['IDENTIFIER'] = df['IDENTIFIER']
		d['DUPLICATE_OF'] = df['DUPLICATE_OF']
		p = []
		for i in cat_predictions:
			p.append(i[0])
		d['pred_cat'] = p
		p = []
		for i in text_predictions:
			p.append(i[0])
		d['pred_text'] = p

		d['actual'] = df['is_duplicate']
		print(d)
		res_coll_name = 'BugDupsTestSet_' + str(options.cluster) + '_description_results'
		collection = db[res_coll_name]
		records = json2.loads(d.T.to_json(date_format='iso')).values()

		collection.create_index([("IDENTIFIER", pymongo.ASCENDING), ("DUPLICATE_OF", pymongo.ASCENDING)], unique=True)
		print(collection.index_information())
		collection.insert(records)
	'''

if __name__ == "__main__":
    main()