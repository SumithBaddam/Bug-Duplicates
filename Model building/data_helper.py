#Data Helper for getting the data in required format
import pymongo
import pandas as pd
import random
import configparser
import argparse
import jsondatetime as json2
import sys
sys.path.insert(0, "/data/ingestion/")
from Utils import *

#Setting up the config.ini file parameters
settings = configparser.ConfigParser()
settings.read('/data/ingestion/config.ini')

#Parser options
options = None

def parse_options():
    parser = argparse.ArgumentParser(description="""Script to predict Potential CFD testing data""")
    parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
    parser.add_argument("--cluster", default="", help ='Comma seperated cluster-ids', type=str, metavar='c')
    args = parser.parse_args()
    
    return args


def build_collection(df, duplicates, org_dup_ids):
req_df = df[df['IDENTIFIER'].isin(org_dup_ids)]

feature_columns_to_use = ['IDENTIFIER', 'Headline', 'ENCL-Description', 'DE_MANAGER_USERID', 'SEVERITY_CODE', 'LIFECYCLE_STATE_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'AGE',  'FEATURE', 'RELEASE_NOTE', 'SA_ATTACHMENT_INDIC', 'CR_ATTACHMENT_INDIC', 'UT_ATTACHMENT_INDIC', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'TICKETS_COUNT', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'TS_INDIC', 'SS_INDIC', 'OIB_INDIC', 'STATE_ASSIGN_INDIC', 'STATE_CLOSE_INDIC', 'STATE_DUPLICATE_INDIC', 'STATE_FORWARD_INDIC', 'STATE_HELD_INDIC', 'STATE_INFO_INDIC', 'STATE_JUNK_INDIC', 'STATE_MORE_INDIC', 'STATE_NEW_INDIC', 'STATE_OPEN_INDIC', 'STATE_POSTPONE_INDIC', 'STATE_RESOLVE_INDIC', 'STATE_SUBMIT_INDIC', 'STATE_UNREP_INDIC', 'STATE_VERIFY_INDIC', 'STATE_WAIT_INDIC', 'CFR_INDIC', 'S12RD_INDIC', 'S123RD_INDIC', 'MISSING_SS_EVAL_INDIC', 'S123_INDIC', 'S12_INDIC', 'RNE_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY',  'TEST_EDP_PHASE', 'RESOLVER_ANALYSIS_INDIC', 'SUBMITTER_ANALYSIS_INDIC', 'EDP_ANALYSIS_INDIC', 'RETI_ANALYSIS_INDIC', 'DESIGN_REVIEW_ESCAPE_INDIC', 'STATIC_ANALYSIS_ESCAPE_INDIC', 'FUNC_TEST_ESCAPE_INDIC', 'SELECT_REG_ESCAPE_INDIC', 'CODE_REVIEW_ESCAPE_INDIC', 'UNIT_TEST_ESCAPE_INDIC', 'DEV_ESCAPE_INDIC', 'FEATURE_TEST_ESCAPE_INDIC', 'REG_TEST_ESCAPE_INDIC', 'SYSTEM_TEST_ESCAPE_INDIC', 'SOLUTION_TEST_ESCAPE_INDIC', 'INT_TEST_ESCAPE_INDIC', 'GO_TEST_ESCAPE_INDIC', 'COMPLETE_ESCAPE_INDIC', 'SR_CNT', 'PSIRT_INDIC',  'BADCODEFLAG',   'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'URC_DISPOSED_INDIC', 'CLOSED_DISPOSED_INDIC', 'REGRESSION_BUG_FLAG']
#nonnumeric_columns = ['DE_MANAGER_USERID', 'LIFECYCLE_STATE_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'RELEASE_NOTE', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'TICKETS_COUNT', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY', 'TEST_EDP_PHASE', 'BADCODEFLAG',  'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'REGRESSION_BUG_FLAG']

#duplicates['Headline2'] = ""
#duplicates['ENCL-Description2'] = ""

ids = list(req_df['IDENTIFIER'].unique())

waste = []
headlines = []
encl_descriptions = []
c=0
new_duplicates = duplicates[duplicates['DUPLICATE_OF'].isin(ids)]

a = pd.DataFrame()
for i, row in new_duplicates.iterrows():
	c = c + 1
	print(c)
	identifier = row['DUPLICATE_OF']
	if identifier in ids:
		#duplicates.ix[i, 'Headline2'] = list(req_df[req_df['IDENTIFIER'] == identifier]['Headline'])[0]
		#duplicates.ix[i, 'ENCL-Description2'] = list(req_df[req_df['IDENTIFIER'] == identifier]['ENCL-Description'])[0]
		a = a.append(req_df[req_df['IDENTIFIER'] == identifier][feature_columns_to_use])
		#headlines.append(list(req_df[req_df['IDENTIFIER'] == identifier]['Headline'])[0])
		#encl_descriptions.append(list(req_df[req_df['IDENTIFIER'] == identifier]['ENCL-Description'])[0])
	else:
		waste.append(c)


#a['DUPLICATE_OF'] = a ['IDENTIFIER']
new_feature_columns = ['DUPLICATE_OF','Headline2', 'ENCL-Description2', 'DE_MANAGER_USERID2', 'SEVERITY_CODE2', 'LIFECYCLE_STATE_CODE2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'AGE2', 'FEATURE2', 'RELEASE_NOTE2', 'SA_ATTACHMENT_INDIC2', 'CR_ATTACHMENT_INDIC2', 'UT_ATTACHMENT_INDIC2', 'IMPACT2', 'ORIGIN2', 'IS_CUSTOMER_VISIBLE2', 'TICKETS_COUNT2', 'INCOMING_INDIC2', 'BACKLOG_INDIC2', 'DISPOSED_INDIC2', 'TS_INDIC2', 'SS_INDIC2', 'OIB_INDIC2', 'STATE_ASSIGN_INDIC2', 'STATE_CLOSE_INDIC2', 'STATE_DUPLICATE_INDIC2', 'STATE_FORWARD_INDIC2', 'STATE_HELD_INDIC2', 'STATE_INFO_INDIC2', 'STATE_JUNK_INDIC2', 'STATE_MORE_INDIC2', 'STATE_NEW_INDIC2', 'STATE_OPEN_INDIC2', 'STATE_POSTPONE_INDIC2', 'STATE_RESOLVE_INDIC2', 'STATE_SUBMIT_INDIC2', 'STATE_UNREP_INDIC2', 'STATE_VERIFY_INDIC2', 'STATE_WAIT_INDIC2', 'CFR_INDIC2', 'S12RD_INDIC2', 'S123RD_INDIC2', 'MISSING_SS_EVAL_INDIC2', 'S123_INDIC2', 'S12_INDIC2', 'RNE_INDIC2', 'UPDATED_BY2', 'DEV_ESCAPE_ACTIVITY2', 'RELEASED_CODE2', 'TEST_EDP_ACTIVITY2', 'TEST_EDP_PHASE2', 'RESOLVER_ANALYSIS_INDIC2', 'SUBMITTER_ANALYSIS_INDIC2', 'EDP_ANALYSIS_INDIC2', 'RETI_ANALYSIS_INDIC2', 'DESIGN_REVIEW_ESCAPE_INDIC2', 'STATIC_ANALYSIS_ESCAPE_INDIC2', 'FUNC_TEST_ESCAPE_INDIC2', 'SELECT_REG_ESCAPE_INDIC2', 'CODE_REVIEW_ESCAPE_INDIC2', 'UNIT_TEST_ESCAPE_INDIC2', 'DEV_ESCAPE_INDIC2', 'FEATURE_TEST_ESCAPE_INDIC2', 'REG_TEST_ESCAPE_INDIC2', 'SYSTEM_TEST_ESCAPE_INDIC2', 'SOLUTION_TEST_ESCAPE_INDIC2', 'INT_TEST_ESCAPE_INDIC2', 'GO_TEST_ESCAPE_INDIC2', 'COMPLETE_ESCAPE_INDIC2', 'SR_CNT2', 'PSIRT_INDIC2', 'BADCODEFLAG2',  'RISK_OWNER2', 'SIR2', 'PSIRT_FLAG2', 'URC_DISPOSED_INDIC2', 'CLOSED_DISPOSED_INDIC2', 'REGRESSION_BUG_FLAG2']
for i in range(0, len(new_feature_columns)):
	print(i)
	new_duplicates[new_feature_columns[i]] = list(a[feature_columns_to_use[i]])

#new_duplicates[new_feature_columns] = a[feature_columns_to_use]

new_duplicates['Headline2'] = headlines
new_duplicates['ENCL-Description2'] = encl_descriptions
new_duplicates['is_duplicate'] = 1
#new_duplicates = duplicates.drop(duplicates.index[waste])

#For non dups dataset...
#Fetch for 1st row and copy for next 3 indexes
#Then for Headline2 do the same as above...

identifiers = []
dup_identifiers = []
headlines1 = []
encl_descriptions1 = []
headlines2 = []
encl_descriptions2 = []

new_non_duplicates = pd.DataFrame()
non_a_1 = pd.DataFrame()
non_a_2 = pd.DataFrame()
a = pd.DataFrame()
for c in range(0, len(non_req_list)):
	tup_id = non_req_list[c]
	if(c%4 == 0):
		#a = df[df['IDENTIFIER'] == tup_id[0]]
		#h = list(a['Headline'])[0]
		#e = list(a['ENCL-Description'])[0]
		a = df[df['IDENTIFIER'] == tup_id[0]][feature_columns_to_use]
		non_a_1 = non_a_1.append(a)
		#for j in range(4):
		#identifiers.append(tup_id[0])
		#headlines1.append(h)
		#encl_descriptions1.append(e)
		#a1 = df[df['IDENTIFIER'] == tup_id[1]]
		non_a_2 = non_a_2.append(df[df['IDENTIFIER'] == tup_id[1]][feature_columns_to_use])
		#dup_identifiers.append(tup_id[1])
		#headlines2.append(list(a1['Headline'])[0])
		#encl_descriptions2.append(list(a1['ENCL-Description'])[0])
	else:
		#identifiers.append(tup_id[0])
		#headlines1.append(h)
		#encl_descriptions1.append(e)
		non_a_1 = non_a_1.append(a)
		#a1 = df[df['IDENTIFIER'] == tup_id[1]]
		non_a_2 = non_a_2.append(df[df['IDENTIFIER'] == tup_id[1]][feature_columns_to_use])
		#dup_identifiers.append(tup_id[1])
		#headlines2.append(list(a1['Headline'])[0])
		#encl_descriptions2.append(list(a1['ENCL-Description'])[0])
	print(c)

new_non_duplicates = non_a_1
#non_a_2['DUPLICATE_OF'] = non_a_2['IDENTIFIER']
print(new_non_duplicates.columns)
for i in range(0, len(new_feature_columns)):
	print(i)
	new_non_duplicates[new_feature_columns[i]] = list(non_a_2[feature_columns_to_use[i]])

new_non_duplicates['IDENTIFIER'] = identifiers
new_non_duplicates['DUPLICATE_OF'] = dup_identifiers
new_non_duplicates['Headline'] = headlines1
new_non_duplicates['Headline2'] = headlines2
new_non_duplicates['ENCL-Description'] = encl_descriptions1
new_non_duplicates['ENCL-Description2'] = encl_descriptions2
new_non_duplicates['is_duplicate'] = 0

new_duplicates_1 = new_duplicates[list(new_non_duplicates.columns)]
#new_duplicates_1 = new_duplicates[['IDENTIFIER', 'DUPLICATE_OF','Headline', 'Headline2', 'ENCL-Description', 'ENCL-Description2', 'is_duplicate']]
new_duplicates_1 = new_duplicates_1.append(new_non_duplicates)

new_duplicates_1.reset_index(drop = True, inplace = True)
records = json2.loads(new_duplicates_1.T.to_json(date_format='iso')).values()

collection = db['BugDupsTrainSet_436_1452_new']
collection.create_index([("IDENTIFIER", pymongo.ASCENDING), ("DUPLICATE_OF", pymongo.ASCENDING)], unique=True)
print(collection.index_information())
collection.insert(records)
print("Inserted data to results collection")



def fetch_bugs_list(db):
cluster_id = options.cluster
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

print(df['PROJECT'].unique())

duplicates = df[df['DUPLICATE_OF'].isnull() == False]
non_duplicates = df[df['DUPLICATE_OF'].isnull() == True]
print(duplicates[['IDENTIFIER', 'DUPLICATE_OF']])
#https://stackoverflow.com/questions/48220643/add-values-to-an-existing-dataframe-from-list-of-tuples
req_list = list(zip(duplicates['IDENTIFIER'], duplicates['DUPLICATE_OF']))
org_dup_ids = list(duplicates['DUPLICATE_OF'].unique())
dup_ids = list(duplicates['IDENTIFIER'])
non_dup_ids = list(non_duplicates['IDENTIFIER'])

non_req_list = []
for id in dup_ids:
	rand_items = random.sample(non_dup_ids, 2)
	for i in rand_items:
		req_list.append((id, i))
		non_req_list.append((id, i))

org_non_dups_ids = [j for i,j in req_list]
build_collection(df, duplicates, org_dup_ids)


def main():
	options = parse_options()
	if(options.env == "Prod"):
		key = "csap_prod_database"

	else:
		key = "csap_stage_database"

	db = get_db(settings, key)
	fetch_bugs_list(db)