# python ./bug_duplicates_train_test.py --env Prod --train True --cluster 1
# python ./bug_duplicates_train_test.py --env Prod --train False --viewID 436 --queryID 1452

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
stops = set(stopwords.words('english'))
max_seq_length = 150
text_cols = ['complete1', 'complete2']
embedding_dim = 150 #300

#Ensemble model for training
def stacking_train(df, cluster):
    pred_columns=['actual','pred_cat','pred_text']
    stacking_df=df[pred_columns]
    train_x = stacking_df.drop('actual', axis=1).as_matrix()
    train_y = stacking_df['actual'].as_matrix()
    #X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(train_x, train_y) #SMOTE
    model = XGBClassifier(objective='binary:logistic')
    #model.fit(X_resampled, y_resampled)
    model.fit(train_x, train_y)
    print('fitting done')
    filename = '/data/ingestion/bugDuplicates/bugDup_ensemble.txt'# + str(cluster) + '.txt'
    if os.path.exists(filename):
        os.remove(filename)
        print("File Removed!")
    with open(filename, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    y_pred = model.predict_proba(train_x)[:,1]
    return y_pred

#Ensemble model for testing
def stacking_test(test_df, cluster):
    pred_columns=['pred_cat','pred_text']
    stacking_df=test_df[pred_columns]
    filename = '/data/ingestion/bugDuplicates/bugDup_ensemble.txt'# + str(cluster) + '.txt'
    with open(filename, 'rb') as f:
        model= pickle.load(f)   
    test_x = stacking_df.as_matrix()
    y_pred = model.predict_proba(test_x)[:,1]  
    return y_pred


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
	cluster_id = 3
	return cluster_id


#chars = ["?", "'s", ">", "<", ",", ":", "'", "''", "--", "`", "``", "...", "", "!", "#", '"', '$', '%', '&', '(', ')', '*', '+', '-', '.', '/', ';', '=', '@', '[', '\\', ']', '^', '_', '{', '}', '|', '~', '\t', '\n', '']
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
    parser.add_argument("--train", default=True, help ='True/False', type=str, metavar='t')
    parser.add_argument("--cluster", default="", help ='Comma seperated cluster-ids', type=str, metavar='c')
    parser.add_argument("--viewID", default="", help ='View ID', type=str, metavar='v')
    parser.add_argument("--queryID", default="", help ='Query ID', type=str, metavar='q')
    args = parser.parse_args()
    return args

#Loading the training data
def load_data(db, collection):
	#collection = db['BugDupsTrainSet_3']
	cursor = collection.find({})
	df =  pd.DataFrame(list(cursor))
	df['complete1'] = df["Headline"].astype(str) + " " + df["ENCL-Description"].astype(str)
	df['complete2'] = df["Headline2"].astype(str) + " " + df["ENCL-Description2"].astype(str)
	return df


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
	#print(project_clusters)
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
	#new_feature_columns = ['DE_MANAGER_USERID2', 'SEVERITY_CODE2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'FEATURE2', 'IMPACT2', 'ORIGIN2', 'RISK_OWNER2']
	#feature_columns_to_use = ['DE_MANAGER_USERID', 'SEVERITY_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'IMPACT', 'ORIGIN', 'RISK_OWNER']
	train_df = df
	validate_df = df[df['SUBMITTED_DATE'] >= '2018-01']
	new_feature_columns = ['PROJECT2', 'PRODUCT2', 'COMPONENT2']
	feature_columns_to_use = ['PROJECT', 'PRODUCT', 'COMPONENT']
	#AGE, SR_CNT, LIFECYCLE_STATE_CODE -- removed, (DE_MANAGER, COMPONENT, PROJECT, PRODUCT, FEATURE, IMPACT, ORIGIN, UPDATED_BY) -- for these just check if they are same or not 1/0.
	#avoid_cols = ['DE_MANAGER_USERID', 'COMPONENT', 'PROJECT', 'PRODUCT', 'FEATURE', 'IMPACT', 'ORIGIN', 'UPDATED_BY', 'RISK_OWNER', 'FEATURE', 'DEV_ESCAPE_ACTIVITY']
	#avoid_cols = ['DE_MANAGER_USERID', 'COMPONENT', 'PROJECT', 'PRODUCT', 'FEATURE', 'IMPACT', 'ORIGIN','RISK_OWNER', 'FEATURE']
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
	big_X_train = train_df[cols]
	big_X_val = validate_df[cols]
	#nonnumeric_columns_1 = ['DE_MANAGER_USERID', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'RELEASE_NOTE', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'TICKETS_COUNT', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY', 'TEST_EDP_PHASE', 'BADCODEFLAG',  'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'REGRESSION_BUG_FLAG']
	#nonnumeric_columns_2 = ['DE_MANAGER_USERID2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'FEATURE2', 'RELEASE_NOTE2', 'IMPACT2', 'ORIGIN2', 'IS_CUSTOMER_VISIBLE2', 'TICKETS_COUNT2', 'INCOMING_INDIC2', 'BACKLOG_INDIC2', 'DISPOSED_INDIC2', 'UPDATED_BY2', 'DEV_ESCAPE_ACTIVITY2', 'RELEASED_CODE2', 'TEST_EDP_ACTIVITY2', 'TEST_EDP_PHASE2', 'BADCODEFLAG2', 'RISK_OWNER2', 'SIR2', 'PSIRT_FLAG2', 'REGRESSION_BUG_FLAG2']
	#nonnumeric_columns_1 = ['DE_MANAGER_USERID', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY', 'TEST_EDP_PHASE', 'BADCODEFLAG',  'RISK_OWNER', 'PSIRT_FLAG', 'REGRESSION_BUG_FLAG']
	#nonnumeric_columns_2 = ['DE_MANAGER_USERID2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'FEATURE2', 'IMPACT2', 'ORIGIN2', 'IS_CUSTOMER_VISIBLE2', 'INCOMING_INDIC2', 'BACKLOG_INDIC2', 'DISPOSED_INDIC2', 'UPDATED_BY2', 'DEV_ESCAPE_ACTIVITY2', 'RELEASED_CODE2', 'TEST_EDP_ACTIVITY2', 'TEST_EDP_PHASE2', 'BADCODEFLAG2', 'RISK_OWNER2', 'PSIRT_FLAG2', 'REGRESSION_BUG_FLAG2']
	nonnumeric_columns_1 = ['PROJECT', 'PRODUCT', 'COMPONENT']
	nonnumeric_columns_2 = ['PROJECT2', 'PRODUCT2', 'COMPONENT2']
	nonnumeric_columns = []
	for i in range(0, len(nonnumeric_columns_1)):
		nonnumeric_columns.append(nonnumeric_columns_1[i])
		nonnumeric_columns.append(nonnumeric_columns_2[i])
	big_X_train = big_X_train.replace(np.nan, '', regex=True)
	big_X_imputed_train = DataFrameImputer().fit_transform(big_X_train.iloc[:,:])
	big_X_val = big_X_val.replace(np.nan, '', regex=True)
	big_X_imputed_val = DataFrameImputer().fit_transform(big_X_val.iloc[:,:])
	le = LabelEncoder()
	for feature in nonnumeric_columns:
		big_X_imputed_train[feature] = big_X_imputed_train[feature].astype(str)
		big_X_imputed_val[feature] = big_X_imputed_val[feature].astype(str)
	for i in range(0, len(nonnumeric_columns), 2):
		#print(i)
		u1 = list(big_X_imputed_train[nonnumeric_columns[i]].unique())
		u2 = list(big_X_imputed_train[nonnumeric_columns[i + 1]].unique())
		u1 = u1 + u2
		myset = list(set(u1))
		d = dict()
		for j in range(0, len(myset)):
			d[myset[j]] = j
		big_X_imputed_train = big_X_imputed_train.replace({nonnumeric_columns[i]: d})
		big_X_imputed_train = big_X_imputed_train.replace({nonnumeric_columns[i + 1]: d})
	for i in range(0, len(nonnumeric_columns), 2):
		#print(i)
		u1 = list(big_X_imputed_val[nonnumeric_columns[i]].unique())
		u2 = list(big_X_imputed_val[nonnumeric_columns[i + 1]].unique())
		u1 = u1 + u2
		myset = list(set(u1))
		d = dict()
		for j in range(0, len(myset)):
			d[myset[j]] = j
		big_X_imputed_val = big_X_imputed_val.replace({nonnumeric_columns[i]: d})
		big_X_imputed_val = big_X_imputed_val.replace({nonnumeric_columns[i + 1]: d})
	X_train = big_X_imputed_train.iloc[:, :len(cols)-1].as_matrix()
	Y_train = big_X_train['is_duplicate']
	X_validation = big_X_imputed_val.iloc[:, :len(cols)-1].as_matrix()
	Y_validation = big_X_val['is_duplicate']
	#X_train = train_X
	if(is_train == True):
		#validation_size = int(df.shape[0]/4)
		#X_train, X_validation, Y_train, Y_validation = train_test_split(train_X, Y, test_size=validation_size)
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
		#Store X_train and Y_train in a collection...
		return X_train, Y_train, X_validation, Y_validation


def build_test_data_cat(df):
	train_df = df
	new_feature_columns = ['PROJECT2', 'PRODUCT2', 'COMPONENT2']
	feature_columns_to_use = ['PROJECT', 'PRODUCT', 'COMPONENT']
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
	big_X_train = train_df[cols]
	nonnumeric_columns_1 = ['PROJECT', 'PRODUCT', 'COMPONENT']
	nonnumeric_columns_2 = ['PROJECT2', 'PRODUCT2', 'COMPONENT2']
	nonnumeric_columns = []
	for i in range(0, len(nonnumeric_columns_1)):
		nonnumeric_columns.append(nonnumeric_columns_1[i])
		nonnumeric_columns.append(nonnumeric_columns_2[i])
	big_X_train = big_X_train.replace(np.nan, '', regex=True)
	big_X_imputed_train = DataFrameImputer().fit_transform(big_X_train.iloc[:,:])
	le = LabelEncoder()
	for feature in nonnumeric_columns:
		big_X_imputed_train[feature] = big_X_imputed_train[feature].astype(str)
	for i in range(0, len(nonnumeric_columns), 2):
		#print(i)
		u1 = list(big_X_imputed_train[nonnumeric_columns[i]].unique())
		u2 = list(big_X_imputed_train[nonnumeric_columns[i + 1]].unique())
		u1 = u1 + u2
		myset = list(set(u1))
		d = dict()
		for j in range(0, len(myset)):
			d[myset[j]] = j
		big_X_imputed_train = big_X_imputed_train.replace({nonnumeric_columns[i]: d})
		big_X_imputed_train = big_X_imputed_train.replace({nonnumeric_columns[i + 1]: d})
	X_train = big_X_imputed_train.iloc[:, :len(cols)-1].as_matrix()
	Y_train = big_X_train['is_duplicate']
	for i in range(0, len(X_train)):
		for k in avoid_cols_indexes:
			if(X_train[i][k] == X_train[i][k + 1]):
				X_train[i][k] = 1
				X_train[i][k + 1] = 1
				#print(k)
			else:
				X_train[i][k] = 0
				X_train[i][k + 1] = 1
	Y_train = Y_train.values
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
	#print(model.summary())
	#Fitting the data into the model
	model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=2, batch_size=64)
	filename = '/data/ingestion/bugDuplicates/model3_lstm_cnn_cat_' + str(cluster) + '.h5'
	model.save(filename) #model.save('/data/ingestion/bugDuplicates/model3_lstm_cnn_cat.h5')
	#prediction = model.predict(X_train)
	return model #, prediction


def test_model_cat(model, X_test, Y_test, cluster):
	prediction = model.predict(X_test)
	#print(X_test[4:6])
	return prediction


def build_data_text(train_df, cluster, db):
	# Prepare embedding
	vocabulary = dict()
	inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
	print("Loading the W2V")
	#word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True) #WE NEED TO HAVE OUR OWN W2V MODEL
	#Our own word2vec model
	sentences = []
	for dataset in [train_df]:
		for index, row in dataset.iterrows():
			#print(text_to_word_list(row["complete1"]))
			#print(index)
			if(type(row['complete1']) != float):
				sentences.append(text_to_word_list(row["complete1"]))
			if(type(row['complete2']) != float):
				sentences.append(text_to_word_list(row["complete2"]))
	# train model
	model = Word2Vec(sentences, min_count=1, size = 150)
	# summarize the loaded model
	#print(model)
	# summarize vocabulary
	words = list(model.wv.vocab)
	# save model
	filename = '/data/ingestion/bugDuplicates/w2vmodel_' + str(cluster) + '.bin'
	model.save(filename) #model.save('/users/sumreddi/model_3.bin')
	c=0
	q1=[0]*(train_df.shape[0])
	q2=[0]*(train_df.shape[0])
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
					if word in stops and word not in words: #word2vec.vocab:
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
	train_df['complete1'] = q1
	train_df['complete2'] = q2
	#Write the vocabulary and inverse vocabulary into a file
	f = '/data/ingestion/bugDuplicates/vocab_model_' + str(cluster) + '.json'
	#f = "/users/sumreddi/vocab_model_3.json"
	f = open(f,'w')
	json1 = json.dumps(vocabulary, indent=4)
	f.write(json1)
	f.close()
	thefile = '/data/ingestion/bugDuplicates/inv_vocab_model_' + str(cluster) + '.json'
	#thefile = "/users/sumreddi/inv_vocab_model_3.json" #str(settings.get("Potential_CFD","temp_path_mod_bugDuplicates")) + '/top_words_cluster_' +str(p)+'.txt'
	with open(thefile, 'wb') as fp:
		pickle.dump(inverse_vocabulary, fp, protocol=2)
	#Store train_df in a collection
	collection = db['BugDupsTrainSet_3_complete']
	train_df.reset_index(drop = True, inplace = True)
	records = json2.loads(train_df.T.to_json(date_format='iso')).values()
	collection.create_index([("IDENTIFIER", pymongo.ASCENDING), ("DUPLICATE_OF", pymongo.ASCENDING)], unique=True)
	collection.insert(records)
	return vocabulary, model, train_df


def build_model_text(vocabulary, model, validation_size, train_df, cluster):
	validation_size = int(train_df.shape[0]/3)
	words = list(model.wv.vocab)
	embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
	embeddings[0] = 0  # So that the padding will be ignored
	# Build the embedding matrix
	for word, index in vocabulary.items():
		#print(index)
		if word in words:#word2vec.vocab:
			embeddings[index] = model[word]#word2vec.word_vec(word)
	#validation_size = 20000
	training_size = len(train_df) - validation_size
	X = train_df[text_cols]
	Y = train_df['is_duplicate']
	X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)
	# Split to dicts
	X_train = {'left': X_train.complete1, 'right': X_train.complete2}
	X_validation = {'left': X_validation.complete1, 'right': X_validation.complete2}
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
	n_epoch = 50
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
	#print("Starting to train")
	malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
								validation_data=([X_validation['left'], X_validation['right']], Y_validation))
	#print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))
	# serialize weights to HDF5
	filename = '/data/ingestion/bugDuplicates/text_model_' + str(cluster) + '.h5'
	malstm.save_weights(filename) #malstm.save_weights("/users/sumreddi/model3_new.h5")
	#print("Saved model to disk")
	#predictions = malstm.predict([X_train['left'], X_train['right']])
	return malstm


def build_test_data_text(model, vocabulary, inverse_vocabulary, test_df, cluster):
	#print("Loaded the W2V")
	words = list(model.wv.vocab)
	c=0
	s=0
	q1=[0]*(test_df.shape[0])
	q2=[0]*(test_df.shape[0])
	a = -1
	# Iterate over the questions only of both training and test datasets
	for dataset in [test_df]:#[train_df, test_df]:
		for index, row in dataset.iterrows():
			#print(a)
			a = a+1
			# Iterate through the text of both questions of the row
			for question in text_cols:
				c = c + 1
				q2n = []  # q2n -> question numbers representation
				for word in text_to_word_list(row[question]):
					# Check for unwanted words
					if word in stops and word not in words: #word2vec.vocab:
						continue
					if word not in vocabulary:
						#vocabulary[word] = len(inverse_vocabulary)
						#q2n.append(0)#len(inverse_vocabulary))
						#inverse_vocabulary.append(word)
						s = 1
					else:
						q2n.append(vocabulary[word])
				# Replace questions as word to question as number representation
				#print(q2n)
				if(c%2 != 0):
					 q1[a] = q2n
				else:
					 q2[a] = q2n
				#dataset.set_value(index, question, q2n)
	test_df['complete1'] = q1
	test_df['complete2'] = q2
	return vocabulary, model, words, test_df


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
	# Model variables
	#print("Loaded model from disk")
	# evaluate loaded model on test data
	predictions = malstm.predict([X_test['left'], X_test['right']])
	return predictions


def main():

key = "csap_prod_database"
db = get_db(settings, key)
view_id = "436"
query_id = "1452"
coll_name = "BugDupsTestSet_" + str(view_id) + "_" + str(query_id)
collection = db[coll_name]
print(collection)
df = load_data(db, collection)
cluster_id = get_cluster(db, df)
#print(df)

print(cluster_id)
X_test, Y_test = build_test_data_cat(df)
cat_predictions = test_model_cat(X_test, Y_test, cluster_id)
vocabulary, w2vmodel, words, test_df = build_test_data_text(df, cluster_id, db)
text_predictions = test_model_text(vocabulary, w2vmodel, words, test_df, cluster_id)
#print(text_predictions, cat_predictions)

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
ensemble_predictions = stacking_test(d, cluster_id)
d['pred_ensemble'] = list(ensemble_predictions)
print(d)

if __name__ == "__main__":
    main()