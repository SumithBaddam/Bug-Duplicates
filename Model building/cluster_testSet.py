import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import sys
sys.path.insert(0, "/data/ingestion/")
from Utils import *
import itertools
from sklearn.cluster import KMeans
import pylab as pl

settings = configparser.ConfigParser()
settings.read('/data/ingestion/config.ini')
max_seq_length = 150 #150 #If only headline - 18, else - 100 ??

#Parser options
options = None

def parse_options():
    parser = argparse.ArgumentParser(description="""Script to create BugDuplicates testing data""")
    parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
    parser.add_argument("--clusterID", default="", help ='Comma seperated cluster-ids', type=str, metavar='c')
    args = parser.parse_args()
    
    return args


def load_data(db, cluster_id):
	collection = db["BugDupsTrainSet_" + str(cluster_id) + "_complete"]
	cursor =collection.find({}) #collection.find({}, {'IDENTIFIER':1, 'wrdEmbSet':1, '_id':0})
	df =  pd.DataFrame(list(cursor))
	#df['new'] = pd.Series(a)
	return df


def clustering_list_of_lists(df):
	X = pad_sequences(df['wrdEmbSet'], maxlen=max_seq_length)
	'''
	n_clusters = range(300, 310)
	score = []
	for i in n_clusters:
		print(i)
		kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
		score.append(kmeans.score(X))

	pl.plot(n_clusters,scores)
	pl.xlabel('Number of Clusters')
	pl.ylabel('Score')
	pl.title('Elbow Curve')
	pl.show()
	scores = [-2.104863658355412e+16, -2.0768526972696176e+16, -2.0543539833908116e+16, -2.0353554732238616e+16, -2.019878193809843e+16, -2.0069855569046164e+16, -1.9954444141580144e+16, -1.9845866228507696e+16, -1.9750966909281388e+16, -1.96348700393771e+16, -1.9585564387947924e+16, -1.948189715689347e+16, -1.938173508373863e+16, -1.9307066361295748e+16, -1.9247200035161612e+16, -1.914919797837463e+16, -1.9097542625491724e+16, -1.9051979669120452e+16, -1.895416673167122e+16, -1.8895059799826096e+16, -1.88458113438587e+16, -1.8751126318924644e+16, -1.8685198847404076e+16, -1.8661953117590212e+16, -1.8606006482392084e+16, -1.85270357420675e+16, -1.849825350645611e+16, -1.8394662100202108e+16, -1.8388222084724604e+16, -1.8323961311339452e+16, -1.8277440264954156e+16, -1.797230216177746e+16, -1.7927094424717562e+16, -1.789410266537435e+16, -1.7852541762253066e+16, -1.78310041928483e+16, -1.7797836476674086e+16]
	scores_100_110 = [-1.6627417829507266e+16, -1.6597463018132776e+16, -1.6601541754439872e+16]
	score_200 = [-1.5784765647188418e+16]
	'''
	optimal_clusters = 20
	kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
	df['labels'] = kmeans.labels_
	centroids = kmeans.cluster_centers_
	d = df[['IDENTIFIER', 'wrdEmbSet', 'Headline', 'ENCL-Description', 'DE_MANAGER_USERID', 'SEVERITY_CODE', 'LIFECYCLE_STATE_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'AGE',  'FEATURE', 'RELEASE_NOTE', 'SA_ATTACHMENT_INDIC', 'CR_ATTACHMENT_INDIC', 'UT_ATTACHMENT_INDIC', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'TICKETS_COUNT', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'TS_INDIC', 'SS_INDIC', 'OIB_INDIC', 'STATE_ASSIGN_INDIC', 'STATE_CLOSE_INDIC', 'STATE_DUPLICATE_INDIC', 'STATE_FORWARD_INDIC', 'STATE_HELD_INDIC', 'STATE_INFO_INDIC', 'STATE_JUNK_INDIC', 'STATE_MORE_INDIC', 'STATE_NEW_INDIC', 'STATE_OPEN_INDIC', 'STATE_POSTPONE_INDIC', 'STATE_RESOLVE_INDIC', 'STATE_SUBMIT_INDIC', 'STATE_UNREP_INDIC', 'STATE_VERIFY_INDIC', 'STATE_WAIT_INDIC', 'CFR_INDIC', 'S12RD_INDIC', 'S123RD_INDIC', 'MISSING_SS_EVAL_INDIC', 'S123_INDIC', 'S12_INDIC', 'RNE_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY',  'TEST_EDP_PHASE', 'RESOLVER_ANALYSIS_INDIC', 'SUBMITTER_ANALYSIS_INDIC', 'EDP_ANALYSIS_INDIC', 'RETI_ANALYSIS_INDIC', 'DESIGN_REVIEW_ESCAPE_INDIC', 'STATIC_ANALYSIS_ESCAPE_INDIC', 'FUNC_TEST_ESCAPE_INDIC', 'SELECT_REG_ESCAPE_INDIC', 'CODE_REVIEW_ESCAPE_INDIC', 'UNIT_TEST_ESCAPE_INDIC', 'DEV_ESCAPE_INDIC', 'FEATURE_TEST_ESCAPE_INDIC', 'REG_TEST_ESCAPE_INDIC', 'SYSTEM_TEST_ESCAPE_INDIC', 'SOLUTION_TEST_ESCAPE_INDIC', 'INT_TEST_ESCAPE_INDIC', 'GO_TEST_ESCAPE_INDIC', 'COMPLETE_ESCAPE_INDIC', 'SR_CNT', 'PSIRT_INDIC',  'BADCODEFLAG',   'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'URC_DISPOSED_INDIC', 'CLOSED_DISPOSED_INDIC', 'REGRESSION_BUG_FLAG', 'SUBMITTED_DATE', 'DUPLICATE_OF', 'labels']]
	d = d.drop_duplicates(subset='IDENTIFIER', keep="last")
	d.reset_index(drop = True, inplace = True)
	records = json.loads(d.T.to_json()).values()
	db.BugDupsTrainSet_3_cluster.insert(records)
	return 0


def testing_clustering(df):
	#X = pad_sequences(df['wrdEmbSet'], maxlen=max_seq_length)
	predictions = kmeans.predict([[0, 0], [4, 4]])


def main():
	options = parse_options()
	if(options.env.lower() == "prod"):
		key = "csap_prod_database"

	elif(options.env.lower() == "stage"):
		key = "csap_stage_database"

	db = get_db(settings, key)

	cluster_id = int(options.clusterID)
	df = load_data(db, cluster_id)
