import pandas as pd
import numpy as	np
from keras.preprocessing.sequence import pad_sequences
import sys
sys.path.insert(0, "/data/ingestion/")
from Utils import *
import itertools
from sklearn.cluster import	KMeans
import pylab as	pl
from collections import defaultdict
import json
import argparse
import configparser

settings = configparser.ConfigParser()
settings.read('/data/ingestion/config.ini')

#Parser options
options = None

def parse_options():
	parser = argparse.ArgumentParser(description="""Script to predict Bug Duplicates""")
	parser.add_argument("--env", default="stage", help='Environment', type=str,	metavar='E')
	parser.add_argument("--cluster", default="", help ='clusetr	id', type=str, metavar='c')
	args = parser.parse_args()

	return args

def load_data(db, cluster_id):
	collection = db["BugDupsTrainSet_" + str(cluster_id)]
	cursor =collection.find({})
	df =  pd.DataFrame(list(cursor))
	return df

def load_data_complete(db, cluster_id):
	collection = db["BugDupsTrainSet_" + str(cluster_id) + "_complete"]
	cursor =collection.find({})
	df =  pd.DataFrame(list(cursor))
	return df

def populate_components_clusters(db, cluster_id):
	df = load_data(db, cluster_id)
	dups = df[df['is_duplicate'] ==	1]
	complete_df	= load_data_complete(db, cluster_id)

	d =	defaultdict(list)
	d['cluster_id']	= str(cluster_id)

	for index, row in dups.iterrows():
		#print(index)
		if(isinstance(row['COMPONENT'],	str) and isinstance(row['DUP_COMPONENT'], str)):
			if(row['DUP_COMPONENT']	not	in d[row['COMPONENT'].replace('.',	'-')]):
				d[row['COMPONENT'].replace('.',	'-')].append(row['DUP_COMPONENT'].replace('.',	'-'))

	unknown_comps =	list(set(complete_df.COMPONENT)	- set(df.COMPONENT))
	for comp in unknown_comps:
		if(isinstance(comp,	str) and not(comp == None)):
			d[comp.replace('.',	'-')] =	[comp.replace('.',	'-')]

	collection = db['BugDupsTrainSet_components']
	collection.insert(d)

	return


def test_components_cluster(db, cluster_id):
	collection = db['BugDupsTrainSet_components']
	cursor = collection.find({'cluster_id': str(cluster_id)})
	d =	list(cursor)[0]

	#original_comp = 'port'
	components = []
	components.append(original_comp)
	for mapped_comp in set(d[original_comp]):
		print(mapped_comp)
		components.append(mapped_comp)
		if mapped_comp in d.keys():
			for c in d[mapped_comp]:
				components.append(c)

	complete_df	= load_data_complete(db, cluster_id)
	complete_df[complete_df['COMPONENT'].isin(set(components))]

	return


if __name__	== "__main__":
	options	= parse_options()
	
	if(options.env.lower() == "prod"):
		key	= "csap_prod_database"

	elif(options.env.lower() ==	"stage"):
		key	= "csap_stage_database"
	
	db = get_db(settings, key)
	cluster_id = options.cluster
	populate_components_clusters(db, cluster_id)
