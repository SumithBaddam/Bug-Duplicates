##########Populate Submitted Date for bug_dups _results collections from _complete coleltions#########
import pandas
import pymongo
import jsondatetime as json2
import sys
sys.path.insert(0, "/data/ingestion/")
from Utils import *
import configparser
import shutil
import argparse
import json
import pickle

#Setting up the config.ini file parameters
settings = configparser.ConfigParser()
settings.read('/data/ingestion/config.ini')

key = "csap_prod_database"
db = get_db(settings, key)

views = ['107', '384', '394', '480', '436', '493', '639', '673', 'Security', 'Security', 'Security', 'Security', 'Security']
queries = ['477', '1927', '1214', '1570', '1452', '1608', '1968', '2061', '1', '2', '4', '5', '6']

for j in range(len(views)):
	view = views[j]
	query = queries[j]
	print(view)
	coll1 = db['BugDupsTestSet_'+view+'_'+query+'_complete']
	coll2 = db['BugDupsTestSet_'+view+'_'+query+'_results']
	cursor = coll1.find({})
	df1 =  pd.DataFrame(list(cursor))
	cursor = coll2.find({})
	df2 =  pd.DataFrame(list(cursor))
	#df3 = df2.join(df1, on = 'IDENTIFIER', lsuffix='_left', rsuffix='_right')
	df3 = pd.merge(df1, df2[df2.columns], on='IDENTIFIER', suffixes = ['', 'y'])
	cols = list(df2.columns) + ['SUBMITTED_DATE']
	df3 = df3[cols]
	#df3 = df3.drop_duplicates(subset='IDENTIFIER', keep="last")
	records = json.loads(df3.T.to_json(default_handler=str)).values()
	#records = json2.loads(df3.T.to_json(date_format='iso')).values()
	coll2.drop()
	coll2.insert(records)