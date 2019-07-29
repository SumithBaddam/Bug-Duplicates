# python ./bug_dups_data_helper.py --env Prod --cluster 3

#Flow of running the files
	#Data_helper
	#Train_test
	#Build complete list clusterID
	#Test:
		#Build complete list viewID
		#Run test_view_set_new or test_new_bug_new

#Data Helper for getting the Bug Data data in required format from Potential CFD datasets
import pymongo
import pandas as pd
import numpy as np
import random
from random import sample
import configparser
import argparse
import jsondatetime as json2
import datetime
import sys
sys.path.insert(0, "/data/ingestion/")
from Utils import *

#Setting up the config.ini file parameters
settings = configparser.ConfigParser()
settings.read('/data/ingestion/config.ini')
train_prefix = str(settings.get("BugDuplicates","training_prefix"))

#Parser options
options = None

def parse_options():
	parser = argparse.ArgumentParser(description="""Script to predict Potential CFD testing data""")
	parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
	parser.add_argument("--cluster", default="", help ='Comma seperated cluster-ids', type=str, metavar='c')
	parser.add_argument("--viewID", default="", help ='View ID', type=str, metavar='v')
	parser.add_argument("--queryID", default="", help ='Query ID', type=str, metavar='q')
	args = parser.parse_args()
	
	return args


def build_collection(df, duplicates, org_dup_ids, non_req_list, collection):
	print("ref_df starting")
	'''
	req_df = pd.DataFrame()
	j=0
	for i in range(0, int(df.shape[0]/100000)+1):
		print(i)
		a = df.iloc[j: j+1000000, ]
		b = a[a['IDENTIFIER'].isin(org_dup_ids)]
		req_df = req_df.append(b)
	'''
	req_df = df[df['IDENTIFIER'].isin(org_dup_ids)]
	print("req_df done")
	feature_columns_to_use = ['IDENTIFIER', 'Headline', 'ENCL-Description', 'DE_MANAGER_USERID', 'SEVERITY_CODE', 'LIFECYCLE_STATE_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'AGE',  'FEATURE', 'RELEASE_NOTE', 'SA_ATTACHMENT_INDIC', 'CR_ATTACHMENT_INDIC', 'UT_ATTACHMENT_INDIC', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'TICKETS_COUNT', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'TS_INDIC', 'SS_INDIC', 'OIB_INDIC', 'STATE_ASSIGN_INDIC', 'STATE_CLOSE_INDIC', 'STATE_DUPLICATE_INDIC', 'STATE_FORWARD_INDIC', 'STATE_HELD_INDIC', 'STATE_INFO_INDIC', 'STATE_JUNK_INDIC', 'STATE_MORE_INDIC', 'STATE_NEW_INDIC', 'STATE_OPEN_INDIC', 'STATE_POSTPONE_INDIC', 'STATE_RESOLVE_INDIC', 'STATE_SUBMIT_INDIC', 'STATE_UNREP_INDIC', 'STATE_VERIFY_INDIC', 'STATE_WAIT_INDIC', 'CFR_INDIC', 'S12RD_INDIC', 'S123RD_INDIC', 'MISSING_SS_EVAL_INDIC', 'S123_INDIC', 'S12_INDIC', 'RNE_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY',  'TEST_EDP_PHASE', 'RESOLVER_ANALYSIS_INDIC', 'SUBMITTER_ANALYSIS_INDIC', 'EDP_ANALYSIS_INDIC', 'RETI_ANALYSIS_INDIC', 'DESIGN_REVIEW_ESCAPE_INDIC', 'STATIC_ANALYSIS_ESCAPE_INDIC', 'FUNC_TEST_ESCAPE_INDIC', 'SELECT_REG_ESCAPE_INDIC', 'CODE_REVIEW_ESCAPE_INDIC', 'UNIT_TEST_ESCAPE_INDIC', 'DEV_ESCAPE_INDIC', 'FEATURE_TEST_ESCAPE_INDIC', 'REG_TEST_ESCAPE_INDIC', 'SYSTEM_TEST_ESCAPE_INDIC', 'SOLUTION_TEST_ESCAPE_INDIC', 'INT_TEST_ESCAPE_INDIC', 'GO_TEST_ESCAPE_INDIC', 'COMPLETE_ESCAPE_INDIC', 'SR_CNT', 'PSIRT_INDIC',  'BADCODEFLAG',   'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'URC_DISPOSED_INDIC', 'CLOSED_DISPOSED_INDIC', 'REGRESSION_BUG_FLAG', 'SUBMITTED_DATE']
	ids = list(req_df['IDENTIFIER'].unique())
	waste = []
	c=0
	print("new_duplicates staring")
	new_duplicates = duplicates[duplicates['DUPLICATE_OF'].isin(ids)]
	print("new_duplicates done")
	a = pd.DataFrame()
	b = pd.DataFrame()
	for i, row in new_duplicates.iterrows():
		c = c + 1
		#print(c, len(new_duplicates), 1)
		identifier = row['DUPLICATE_OF']
		if identifier in ids:
			b = b.append(req_df[req_df['IDENTIFIER'] == identifier][feature_columns_to_use])
		if(b.shape[0] >= 1000):
			a = a.append(b)
			del b
			b = pd.DataFrame()
	a = a.append(b)
	print(a.shape, new_duplicates.shape)
	#new_feature_columns = ['DUPLICATE_OF','Headline2', 'ENCL-Description2', 'DE_MANAGER_USERID2', 'SEVERITY_CODE2', 'LIFECYCLE_STATE_CODE2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'AGE2', 'FEATURE2', 'RELEASE_NOTE2', 'SA_ATTACHMENT_INDIC2', 'CR_ATTACHMENT_INDIC2', 'UT_ATTACHMENT_INDIC2', 'IMPACT2', 'ORIGIN2', 'IS_CUSTOMER_VISIBLE2', 'TICKETS_COUNT2', 'INCOMING_INDIC2', 'BACKLOG_INDIC2', 'DISPOSED_INDIC2', 'TS_INDIC2', 'SS_INDIC2', 'OIB_INDIC2', 'STATE_ASSIGN_INDIC2', 'STATE_CLOSE_INDIC2', 'STATE_DUPLICATE_INDIC2', 'STATE_FORWARD_INDIC2', 'STATE_HELD_INDIC2', 'STATE_INFO_INDIC2', 'STATE_JUNK_INDIC2', 'STATE_MORE_INDIC2', 'STATE_NEW_INDIC2', 'STATE_OPEN_INDIC2', 'STATE_POSTPONE_INDIC2', 'STATE_RESOLVE_INDIC2', 'STATE_SUBMIT_INDIC2', 'STATE_UNREP_INDIC2', 'STATE_VERIFY_INDIC2', 'STATE_WAIT_INDIC2', 'CFR_INDIC2', 'S12RD_INDIC2', 'S123RD_INDIC2', 'MISSING_SS_EVAL_INDIC2', 'S123_INDIC2', 'S12_INDIC2', 'RNE_INDIC2', 'UPDATED_BY2', 'DEV_ESCAPE_ACTIVITY2', 'RELEASED_CODE2', 'TEST_EDP_ACTIVITY2', 'TEST_EDP_PHASE2', 'RESOLVER_ANALYSIS_INDIC2', 'SUBMITTER_ANALYSIS_INDIC2', 'EDP_ANALYSIS_INDIC2', 'RETI_ANALYSIS_INDIC2', 'DESIGN_REVIEW_ESCAPE_INDIC2', 'STATIC_ANALYSIS_ESCAPE_INDIC2', 'FUNC_TEST_ESCAPE_INDIC2', 'SELECT_REG_ESCAPE_INDIC2', 'CODE_REVIEW_ESCAPE_INDIC2', 'UNIT_TEST_ESCAPE_INDIC2', 'DEV_ESCAPE_INDIC2', 'FEATURE_TEST_ESCAPE_INDIC2', 'REG_TEST_ESCAPE_INDIC2', 'SYSTEM_TEST_ESCAPE_INDIC2', 'SOLUTION_TEST_ESCAPE_INDIC2', 'INT_TEST_ESCAPE_INDIC2', 'GO_TEST_ESCAPE_INDIC2', 'COMPLETE_ESCAPE_INDIC2', 'SR_CNT2', 'PSIRT_INDIC2', 'BADCODEFLAG2',  'RISK_OWNER2', 'SIR2', 'PSIRT_FLAG2', 'URC_DISPOSED_INDIC2', 'CLOSED_DISPOSED_INDIC2', 'REGRESSION_BUG_FLAG2', 'SUBMITTED_DATE2']
	new_feature_columns = ['DUPLICATE_OF','DUP_Headline','DUP_ENCL-Description','DUP_DE_MANAGER_USERID','DUP_SEVERITY_CODE','DUP_LIFECYCLE_STATE_CODE','DUP_PROJECT','DUP_PRODUCT','DUP_COMPONENT','DUP_AGE','DUP_FEATURE','DUP_RELEASE_NOTE','DUP_SA_ATTACHMENT_INDIC','DUP_CR_ATTACHMENT_INDIC','DUP_UT_ATTACHMENT_INDIC','DUP_IMPACT','DUP_ORIGIN','DUP_IS_CUSTOMER_VISIBLE','DUP_TICKETS_COUNT','DUP_INCOMING_INDIC','DUP_BACKLOG_INDIC','DUP_DISPOSED_INDIC','DUP_TS_INDIC','DUP_SS_INDIC','DUP_OIB_INDIC','DUP_STATE_ASSIGN_INDIC','DUP_STATE_CLOSE_INDIC','DUP_STATE_DUPLICATE_INDIC','DUP_STATE_FORWARD_INDIC','DUP_STATE_HELD_INDIC','DUP_STATE_INFO_INDIC','DUP_STATE_JUNK_INDIC','DUP_STATE_MORE_INDIC','DUP_STATE_NEW_INDIC','DUP_STATE_OPEN_INDIC','DUP_STATE_POSTPONE_INDIC','DUP_STATE_RESOLVE_INDIC','DUP_STATE_SUBMIT_INDIC','DUP_STATE_UNREP_INDIC','DUP_STATE_VERIFY_INDIC','DUP_STATE_WAIT_INDIC','DUP_CFR_INDIC','DUP_S12RD_INDIC','DUP_S123RD_INDIC','DUP_MISSING_SS_EVAL_INDIC','DUP_S123_INDIC','DUP_S12_INDIC','DUP_RNE_INDIC','DUP_UPDATED_BY','DUP_DEV_ESCAPE_ACTIVITY','DUP_RELEASED_CODE','DUP_TEST_EDP_ACTIVITY','DUP_TEST_EDP_PHASE','DUP_RESOLVER_ANALYSIS_INDIC','DUP_SUBMITTER_ANALYSIS_INDIC','DUP_EDP_ANALYSIS_INDIC','DUP_RETI_ANALYSIS_INDIC','DUP_DESIGN_REVIEW_ESCAPE_INDIC','DUP_STATIC_ANALYSIS_ESCAPE_INDIC','DUP_FUNC_TEST_ESCAPE_INDIC','DUP_SELECT_REG_ESCAPE_INDIC','DUP_CODE_REVIEW_ESCAPE_INDIC','DUP_UNIT_TEST_ESCAPE_INDIC','DUP_DEV_ESCAPE_INDIC','DUP_FEATURE_TEST_ESCAPE_INDIC','DUP_REG_TEST_ESCAPE_INDIC','DUP_SYSTEM_TEST_ESCAPE_INDIC','DUP_SOLUTION_TEST_ESCAPE_INDIC','DUP_INT_TEST_ESCAPE_INDIC','DUP_GO_TEST_ESCAPE_INDIC','DUP_COMPLETE_ESCAPE_INDIC','DUP_SR_CNT','DUP_PSIRT_INDIC','DUP_BADCODEFLAG', 'DUP_RISK_OWNER','DUP_SIR','DUP_PSIRT_FLAG','DUP_URC_DISPOSED_INDIC','DUP_CLOSED_DISPOSED_INDIC','DUP_REGRESSION_BUG_FLAG','DUP_SUBMITTED_DATE']
	for i in range(0, len(new_feature_columns)):
		#print(i)
		new_duplicates[new_feature_columns[i]] = list(a[feature_columns_to_use[i]])
	new_duplicates['is_duplicate'] = 1
	new_non_duplicates = pd.DataFrame()
	non_a_1 = pd.DataFrame()
	non_a_2 = pd.DataFrame()
	non_a_1_sample = pd.DataFrame()
	non_a_2_sample = pd.DataFrame()
	a = pd.DataFrame()
	if(len(non_req_list) < 1000):
		mod = 100
	if(len(non_req_list) < 100):
		mod = 1
	if(len(non_req_list) > 1000):
		mod = 1000
	for c in range(0, len(non_req_list)):
		tup_id = non_req_list[c]
		if(c%mod == 0):
			non_a_1 = non_a_1.append(non_a_1_sample)
			non_a_2 = non_a_2.append(non_a_2_sample)
			del non_a_1_sample 
			del non_a_2_sample 
			non_a_1_sample = pd.DataFrame()
			non_a_2_sample = pd.DataFrame()
		if(c%4 == 0):
			a = df[df['IDENTIFIER'] == tup_id[0]][feature_columns_to_use]
			non_a_1_sample = non_a_1_sample.append(a)
			non_a_2_sample = non_a_2_sample.append(df[df['IDENTIFIER'] == tup_id[1]][feature_columns_to_use])
		else:
			non_a_1_sample = non_a_1_sample.append(a)
			non_a_2_sample = non_a_2_sample.append(df[df['IDENTIFIER'] == tup_id[1]][feature_columns_to_use])
		#print(c, len(non_req_list))
	new_non_duplicates = non_a_1
	print(new_non_duplicates.columns)
	for i in range(0, len(new_feature_columns)):
		new_non_duplicates[new_feature_columns[i]] = list(non_a_2[feature_columns_to_use[i]])
	new_non_duplicates['is_duplicate'] = 0
	new_duplicates_1 = new_duplicates[list(new_non_duplicates.columns)]
	new_duplicates_1 = new_duplicates_1.append(new_non_duplicates)
	new_duplicates_1.reset_index(drop = True, inplace = True)
	now = datetime.datetime.now()
	new_duplicates_1['train_last_run_date'] = now.strftime("%Y-%m-%d")
	records = json2.loads(new_duplicates_1.T.to_json(date_format='iso')).values()
	#print(new_duplicates_1)
	#collection.create_index([("IDENTIFIER", pymongo.ASCENDING), ("DUPLICATE_OF", pymongo.ASCENDING)], unique=True)
	#print(collection.index_information())
	#collection.drop()
	collection.insert(records)
	print("Inserted data to results collection")
	return 0


def load_data(db, cluster_id, result_collection):
	#Getting the clusters data
	collection = db[settings.get('Potential_CFD', 'proj_cluster')]
	cursor = collection.find({})
	clusters =  pd.DataFrame(list(cursor))
	cursor = result_collection.find({})
	train_last_date_df =  pd.DataFrame(list(cursor))
	print(train_last_date_df)
	if(train_last_date_df.shape[0] > 0):
		train_last_run_date = train_last_date_df['train_last_run_date'].iloc[-1]
	else:
		train_last_run_date = datetime.datetime(2016, 1, 1, 0, 0)
	print(train_last_run_date)
	project_clusters = []
	cluster_status = True
	groups = clusters.groupby('Cluster')
	for name, group in groups:
		project_clusters.append(list(group['Project']))
	print(project_clusters)
	cluster = project_clusters[cluster_id - 1]
	df = pd.DataFrame()
	#Fetching the data for each project in the cluster
	#Fetch only if the date is greater than the latest date in BugDupsTrainSet_<1> collection
	for proj in cluster:
		collection = db[settings.get('Potential_CFD', 'trainPrefix')+ proj.replace('.', '_')]
		cursor = collection.find( {'csap_last_run_date': {'$gt': train_last_run_date}} )
		#cursor = collection.find({}) #collection.find({"csap_last_run_date" : { '$gte' : train_last_run_date }}) 
		print(proj)
		df2 =  pd.DataFrame(list(cursor))
		#df2 = df2[df2['csap_last_run_date'] > train_last_run_date]
		df = df.append(df2)
	print(df.shape)
	if(df.shape[0] > 500000):
		print(df['PROJECT'].unique())
		rindex =  np.array(sample(range(len(df)), 500000))
		df = df.ix[rindex]
		#df = df[:500000,]
	return df


def load_data_view_query(db, view_id, query_id):
	vi_col_name = settings.get('Potential_CFD', 'viewPrefix') + str(view_id) + '_' + str(query_id)
	print(vi_col_name)
	collection = db[vi_col_name]
	cursor = collection.find({})
	df =  pd.DataFrame(list(cursor))
	print(df)
	print(df['PROJECT'].unique())
	return df


def fetch_bugs_list(df):
	duplicates = df[df['DUPLICATE_OF'].isnull() == False]
	non_duplicates = df[df['DUPLICATE_OF'].isnull() == True]
	print(duplicates[['IDENTIFIER', 'DUPLICATE_OF']])
	req_list = list(zip(duplicates['IDENTIFIER'], duplicates['DUPLICATE_OF']))
	org_dup_ids = list(duplicates['DUPLICATE_OF'].unique())
	dup_ids = list(duplicates['IDENTIFIER'])
	non_dup_ids = list(non_duplicates['IDENTIFIER'])
	non_req_list = []
	a=0
	for id in dup_ids:
		#print(a)
		a = a + 1
		rand_items = random.sample(non_dup_ids, 1)
		for i in rand_items:
			req_list.append((id, i))
			non_req_list.append((id, i))
	print("org_non_dups_ids starting")
	org_non_dups_ids = [j for i,j in req_list]
	print("org_non_dups_ids done")
	return df, duplicates, org_dup_ids, non_req_list


def main():
	options = parse_options()
	if(options.env.lower() == "prod"):
		key = "csap_prod_database"
	else:
		key = "csap_stage_database"

	db = get_db(settings, key)

	if(options.viewID != "" and options.queryID != ""):
		#coll_name = "BugDupsTestSet_" + str(options.viewID) + "_" + str(options.queryID)
		train_df = load_data_view_query(db, int(options.viewID), int(options.queryID))
		train_df = train_df.drop_duplicates(subset='IDENTIFIER', keep="last")
	
	else:
		for cluster in range(1, 5):
			#cluster = 4
			coll_name = str(train_prefix) + str(cluster) #"BugDupsTrainSet_" + str(options.cluster)
			print(coll_name)
			result_collection = db[coll_name]
			train_df = load_data(db, int(cluster), result_collection)
			if(train_df.shape[0] > 100):
				train_df = train_df.drop_duplicates(subset='IDENTIFIER', keep="last")
				df, duplicates, org_dup_ids, non_req_list = fetch_bugs_list(train_df)
				print(result_collection)
				build_collection(df, duplicates, org_dup_ids, non_req_list, result_collection)
			else:
				print("There are no new bugs in this collection")
	return 0
if __name__ == "__main__":
    main()