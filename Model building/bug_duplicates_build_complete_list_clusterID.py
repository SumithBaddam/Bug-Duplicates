#This is only for viewID and queryID. This needs to be run after the cluterID model is trained on trainSet
# python ./bug_duplicates_build_complete_list_clusterID.py --env Prod --cluster 3

# /auto/vgapps-cstg02-vapps/analytics/csap/models/files/bugDups/

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
import configparser
import shutil
import argparse
import json
import pickle
import xgboost as xgb
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
max_seq_length = 150 #150 #If only headline, let's use small length
text_cols = ['wrdEmbSet']
embedding_dim = 150 #300


def parse_options():
    parser = argparse.ArgumentParser(description="""Script to create BugDuplicates testing data""")
    parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
    parser.add_argument("--clusterID", default="", help ='Comma seperated cluster-ids', type=str, metavar='c')
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
	return df


def build_test_data_text(test_df, cluster, db):
	filename = model_path + 'w2vmodel_' + str(cluster) + '.bin' #'/data/csap_models/bugDups/w2vmodel_' + str(cluster) + '.bin'
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
	a = -1
	s = 0
	for dataset in [test_df]:
		for index, row in dataset.iterrows():
			#print(a, test_df.shape[0])
			a = a + 1
			for question in text_cols:
				c = c + 1
				q2n = []  
				for word in text_to_word_list(row[question]):
					# Check for unwanted words
					if word.lower() in stops and word.lower() not in words: #word2vec.vocab:
						continue
					if word not in vocabulary:
						s = 1
					else:
						q2n.append(vocabulary[word])
				q1[a] = q2n
				
	print(len(q1), test_df.shape[0])
	test_df['wrdEmbSet'] = q1
	c = train_prefix + str(cluster) + "_complete" #"BugDupsTrainSet_" + str(cluster) + "_complete"
	print(c)
	collection = db[c]
	#['IDENTIFIER', 'complete1', 'Headline', 'ENCL-Description', 'DE_MANAGER_USERID', 'SEVERITY_CODE', 'LIFECYCLE_STATE_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'IMPACT', 'ORIGIN', 'RISK_OWNER', 'SUBMITTED_DATE', 'DUPLICATE_OF']
	#on_the_fly_columns = ['PROJECT', 'PRODUCT', 'COMPONENT']
	#view_set_run_columns = ['DE_MANAGER_USERID', 'SEVERITY_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'IMPACT', 'ORIGIN', 'RISK_OWNER', 'SUBMITTED_DATE', 'DUPLICATE_OF']
	test_df = test_df.drop_duplicates(subset='IDENTIFIER', keep="last")
	projects = list(test_df['PROJECT'].unique())
	collection.drop()
	collection.create_index([("IDENTIFIER", pymongo.ASCENDING), ("DUPLICATE_OF", pymongo.ASCENDING)], unique=True)
	for proj in projects:
		test_df_proj = test_df[test_df['PROJECT'] == proj]
		d = test_df_proj[['IDENTIFIER', 'wrdEmbSet', 'Headline', 'ENCL-Description', 'DE_MANAGER_USERID', 'SEVERITY_CODE', 'LIFECYCLE_STATE_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'AGE',  'FEATURE', 'RELEASE_NOTE', 'SA_ATTACHMENT_INDIC', 'CR_ATTACHMENT_INDIC', 'UT_ATTACHMENT_INDIC', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'TICKETS_COUNT', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'TS_INDIC', 'SS_INDIC', 'OIB_INDIC', 'STATE_ASSIGN_INDIC', 'STATE_CLOSE_INDIC', 'STATE_DUPLICATE_INDIC', 'STATE_FORWARD_INDIC', 'STATE_HELD_INDIC', 'STATE_INFO_INDIC', 'STATE_JUNK_INDIC', 'STATE_MORE_INDIC', 'STATE_NEW_INDIC', 'STATE_OPEN_INDIC', 'STATE_POSTPONE_INDIC', 'STATE_RESOLVE_INDIC', 'STATE_SUBMIT_INDIC', 'STATE_UNREP_INDIC', 'STATE_VERIFY_INDIC', 'STATE_WAIT_INDIC', 'CFR_INDIC', 'S12RD_INDIC', 'S123RD_INDIC', 'MISSING_SS_EVAL_INDIC', 'S123_INDIC', 'S12_INDIC', 'RNE_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY',  'TEST_EDP_PHASE', 'RESOLVER_ANALYSIS_INDIC', 'SUBMITTER_ANALYSIS_INDIC', 'EDP_ANALYSIS_INDIC', 'RETI_ANALYSIS_INDIC', 'DESIGN_REVIEW_ESCAPE_INDIC', 'STATIC_ANALYSIS_ESCAPE_INDIC', 'FUNC_TEST_ESCAPE_INDIC', 'SELECT_REG_ESCAPE_INDIC', 'CODE_REVIEW_ESCAPE_INDIC', 'UNIT_TEST_ESCAPE_INDIC', 'DEV_ESCAPE_INDIC', 'FEATURE_TEST_ESCAPE_INDIC', 'REG_TEST_ESCAPE_INDIC', 'SYSTEM_TEST_ESCAPE_INDIC', 'SOLUTION_TEST_ESCAPE_INDIC', 'INT_TEST_ESCAPE_INDIC', 'GO_TEST_ESCAPE_INDIC', 'COMPLETE_ESCAPE_INDIC', 'SR_CNT', 'PSIRT_INDIC',  'BADCODEFLAG',   'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'URC_DISPOSED_INDIC', 'CLOSED_DISPOSED_INDIC', 'REGRESSION_BUG_FLAG', 'SUBMITTED_DATE', 'DUPLICATE_OF']]
		d = d.drop_duplicates(subset='IDENTIFIER', keep="last")
		d.reset_index(drop = True, inplace = True)
		records = json.loads(d.T.to_json()).values()
		print('Records loaded: ', proj)
		#collection.drop()
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


def load_data(db, cluster_id):
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
	#cluster = ['CSC.embu.dev', 'CSC.security', 'CSC.content-security', 'CSC.netbu']
	#cluster = ['CSC.labtrunk']
	'''
	proj = 'CSC.embu'
	products1 = ['iptm', 'servappl', 'pix-mdc','cmf','wlse1105', 'ct4000-4100', 'mothra', 'vhm', 'isc','qpm','ogs','ciscoworks','enlight','maximus','wcs','campus','csm-tag-vpn','ctm-server301','pluto-1.0','csrc-bpr','cable-manager','aclmanager','ctm','cipm-apollo','vpn-mon','fault-manager','nsm-agent','slam','ct2000','cv-web','cnr-aic','cdcm','netflow-analyzer','ctm-client3.0','apm','essentials','ap1000','ctm-client2.2','bac-sp','qpm-kk','qpmcops','dmm','ptc','galaxy1-0','security-monitor','cerm-2.0','rpms-dev','esse1105','ismg-core','panorama','router-mc','mcp','ngenius','campusmgr','cnsdir_client','shasta-1.0','picasso','cwsi','ipm','cns-aaa-qa','anr','nvt','hse1105','dsu','sms','mwc-prov','cstm','catnip','nt-conn-slm','natk','cnde','cemf','ctm-client301','ids-mdc','urt','cpc','netc','callhome','cdm','ctm-server3.0.2','cnc_rule_packages','java-toolkit','trafficdirector','csrc-dpr','cic-nt','cnsappl-wishlist','cspm-migration','ctm-client','wlc','cml','csrc-aic','videotron-cem','ct4400','location_server','rpms-wish','graphing-service','ctm-server','csmdm','cv4x','nmim','perf-slm','atf','netprofile','mw-nms','idu','spe','ciscoview','ncm','fault-history','mccm','cic-sm','cnm','cvdm','ctm-client3.0.2','mbuildboard','isc-sec','policy','cornerstone','ssng','rpms','cnsappl','ncs','ctm-server3.0','cem2.0-videotron','ipsi','ccnsc-sp','cnsar','somweb','uii','cw2k-automation','triveni','langford','netflow-collector','conn-slm','infocenter','ettx_admin','ptc-mt','cam','cvm','mqgraphs','netforensics','cepnm-cp','ptc-2.1','docsis-cpe-config','ap1100','den-ext','atmdirector','panorama_new','bpr-tools','cea-isdn','cwm','das','5300ems','mcg','cvision','wireless-mgr','ubr7200','com','ctm-server2.2','gsu','lochness','anr-aic','faststep-java','tunnelbuilder','cfm','ugca','cw-windows','ap1800','visionway-client','vct','fault-sdk','apconfig','qoszilla','snms','cat','switchprobe','cnssdk','sesm','cwtools','rmor','vms','ganesha','mse-wips','Lochy','cw2000-dms','cvpn','vsm','netview-option','ana','ctm-server3.1','device-discovery','som','primecollab','lbs_partner','wcm-ngwc','cdmni1','sw-profile-report','cnsds','mccm3','apsyda','mvso','c6130','secaudit','policy-manager','cww-web','ipiu','vmsbundle','sa','qpm-pro','mwc-cm','job-prov','oer-app','anc_ddir','pkgtracker','netaudit','gsr-12000-ems','mgm','ice','tgu','cids-eng','emanager','ap','phoenix','nfc-meter','ugm-wish','prime-central','enterprise_sdn','ssldm','mse','cic-co','iwan-app','lan-switch-slm','discovery','ccdm','cat6000','cea-netflow','aclauditor','cnr-sec','cnsappl-test','gunslinger','mwamcu','ana-3x-vne-drivers','nam','sgm','anc-reg','wdat','anc-portal','edi','anc_prono','corvilnet','edi-tools','virtualswitchmgr','iphdu','cns-cnfgsrv','cepnm','mobexp','nafas','edi-idu','mwc','nmtg-test-portal','dmsapps-ta','cmdsvc','scm-nac','pa_wifi_reporter','ipc-ops-mgr','smos','cssdm','dna_assurance','cmx-sm','cvdm-vpn','cw-multiwireless','wpu','cepnm-dp','nmwlc6','rg','te-planner','etools','ipc-srv-monitor']
	products2 = ['unified_mgmt', 'cns-nm-security','cvdm-webvpn','wism','ime','herbie','pvm','cqmt','nmds-common','cmp','ap1510','bac-tw','ccnsc-vs-2.0','clm','isc-sustaining','cu-prov-mgr','nmewlc','cmp-ps','click-ap','cmm','vts','3750-w','lwapp-ap','ana-apps','ccm_audit','cwncm','ivsim','cqm','clipargen','c1200','ncp','insightcloud','wxsm','nso-mobility','ondc','ana-plat','remote_integration','c1240','ana-nccm','isc-tm','cu-ssmgr','anapm','ct2x00','chameleon','ana-vne-drivers','lmsportal','navigator','c1130','old-cmp','esc','aba-sm','scm-ocd','cunm-voice','openstack','hum','cnac','cwa','iptv_mgt','cnm-ip-infra','ana-automation','osc','datamodelmapper','beacat','cmp-internal-tools','ana-thirdparty','mtx','isc-labtrunk','csm_sp','ctram','cmd','c26-36ems','aba-sdi','ngena','apdm','primeinsight','prime_analytics','pace-portal','si-cse','aidan','ccx-eng','orion','c1520','ana-isdk','profile-c-bts','ce-ipran-solution','ktn_e_cd','cdma-p3-bts','bac','c1140','support-ctm','nm-sim','pace-components','br1310','ms-portal','overdrive','dnac-nfv','ana3x-tools','c1250','vsoc','all','mse-mir','ms-sol','ncm-alert','ana-3x-sow-vne','vnmc','hcm-sa','wret','ana-3x','c1430','lms','ana-kojak','nam-dev','ap1230','ms-ops-mgr','ana-cdn','ifm-sam','wcs-lite','dlc-rms','ifm','mediation','templatizer','xde','bsa','ana-3x-nc-vne','ana-dev-env','cepnm-auto','ddi-ng','support-ana','apic-em-sdg','c800','collabmgr','nm-sim-ng','emms','cnr-ipam','ncs-wan','nfa','primeac','wnbu-sim-tools','primeordful','evora','wlcca','padma','ana-tools','cpdsc','prime-home','primefaststream','cww3x','dess-garmisch','sibr','ipexpress','prime-sp-ems','instanteval','prime-ful-epn','bct','pi-techpack','n7k-pi','xmp','bac-acs','ems','cah','airprovision','wlc-mobile-app','utd-app','cepnm-labtrunk','nso-platform','cmlm-dp','ucact','c1800','br1410','wsa','edison','nso-vms','cmx','erp','msp-cloud','crs-1','c10000mgr','rfw','secpa','nsc-ngn','cids-pkgs','cerm-1.0','si-sage','ism','device-pack','c521','cpt','prime_a_apps','cemf-integration','staros','cnses','ugm-devtest','5300ems-videotron','spectre','cvm20','scm-export','tools','spm','cw-pc','cdm-6200','cns-ccs','unknown','ccp','cesse1105','ctm-client3.1','cww5','ape','cdm-v2','cat3500xl','cdma-pcapps','wancv','c1160','vs5300m','visionway-server','mibtoaster','iptrat','c7600','magga-plus','c6nam','nexus-1000v','katana','n7k-platform','c1524','cas-cwm','waas','click-ap-11ax','dna_analytics','cmx-cloud','enterprise_dnac']
	products = products1 + products2
	for prod in products:
		collection = db[settings.get('Potential_CFD', 'trainPrefix')+ proj.replace('.', '_')]
		cursor = collection.find({'PRODUCT': prod}) 
		print(prod)
		df2 =  pd.DataFrame(list(cursor))
		df = df.append(df2)
	'''
	for proj in cluster:
		collection = db[settings.get('Potential_CFD', 'trainPrefix')+ proj.replace('.', '_')]
		cursor = collection.find({}) 
		print(proj)
		df2 =  pd.DataFrame(list(cursor))
		df = df.append(df2)
	
	print(df['PROJECT'].unique())
	print(df.shape)
	df['wrdEmbSet'] = df["Headline"].astype(str) + " " + df["ENCL-Description"].astype(str)
	return df


def main():
	options = parse_options()
	if(options.env.lower() == "prod"):
		key = "csap_prod_database"

	elif(options.env.lower() == "stage"):
		key = "csap_stage_database"

	db = get_db(settings, key)

	#cluster_id = int(options.clusterID)
	for i in range(1, 5):
		cluster_id = i
		df = load_data(db, cluster_id)
		build_test_data_text(df, cluster_id, db)

if __name__ == "__main__":
    main()