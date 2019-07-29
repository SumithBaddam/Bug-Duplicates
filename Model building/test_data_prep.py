#This is an old file. Updated working file is test_data_prer_1.py. This file works for finding the bugs complete1 values, so it takes time.
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
    parser = argparse.ArgumentParser(description="""Script to predict Bug Duplicates""")
    parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
    parser.add_argument("--cluster", default="", help ='Comma seperated cluster-ids', type=str, metavar='c')
    args = parser.parse_args()
    
    return args

def fetch_prep_data(df, row):
	#feature_columns_to_use = ['IDENTIFIER', 'Headline', 'ENCL-Description', 'DE_MANAGER_USERID', 'SEVERITY_CODE', 'LIFECYCLE_STATE_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'AGE',  'FEATURE', 'RELEASE_NOTE', 'SA_ATTACHMENT_INDIC', 'CR_ATTACHMENT_INDIC', 'UT_ATTACHMENT_INDIC', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'TICKETS_COUNT', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'TS_INDIC', 'SS_INDIC', 'OIB_INDIC', 'STATE_ASSIGN_INDIC', 'STATE_CLOSE_INDIC', 'STATE_DUPLICATE_INDIC', 'STATE_FORWARD_INDIC', 'STATE_HELD_INDIC', 'STATE_INFO_INDIC', 'STATE_JUNK_INDIC', 'STATE_MORE_INDIC', 'STATE_NEW_INDIC', 'STATE_OPEN_INDIC', 'STATE_POSTPONE_INDIC', 'STATE_RESOLVE_INDIC', 'STATE_SUBMIT_INDIC', 'STATE_UNREP_INDIC', 'STATE_VERIFY_INDIC', 'STATE_WAIT_INDIC', 'CFR_INDIC', 'S12RD_INDIC', 'S123RD_INDIC', 'MISSING_SS_EVAL_INDIC', 'S123_INDIC', 'S12_INDIC', 'RNE_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY',  'TEST_EDP_PHASE', 'RESOLVER_ANALYSIS_INDIC', 'SUBMITTER_ANALYSIS_INDIC', 'EDP_ANALYSIS_INDIC', 'RETI_ANALYSIS_INDIC', 'DESIGN_REVIEW_ESCAPE_INDIC', 'STATIC_ANALYSIS_ESCAPE_INDIC', 'FUNC_TEST_ESCAPE_INDIC', 'SELECT_REG_ESCAPE_INDIC', 'CODE_REVIEW_ESCAPE_INDIC', 'UNIT_TEST_ESCAPE_INDIC', 'DEV_ESCAPE_INDIC', 'FEATURE_TEST_ESCAPE_INDIC', 'REG_TEST_ESCAPE_INDIC', 'SYSTEM_TEST_ESCAPE_INDIC', 'SOLUTION_TEST_ESCAPE_INDIC', 'INT_TEST_ESCAPE_INDIC', 'GO_TEST_ESCAPE_INDIC', 'COMPLETE_ESCAPE_INDIC', 'SR_CNT', 'PSIRT_INDIC',  'BADCODEFLAG',   'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'URC_DISPOSED_INDIC', 'CLOSED_DISPOSED_INDIC', 'REGRESSION_BUG_FLAG', 'complete1']
	#new_feature_columns = ['DUPLICATE_OF','Headline2', 'ENCL-Description2', 'DE_MANAGER_USERID2', 'SEVERITY_CODE2', 'LIFECYCLE_STATE_CODE2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'AGE2', 'FEATURE2', 'RELEASE_NOTE2', 'SA_ATTACHMENT_INDIC2', 'CR_ATTACHMENT_INDIC2', 'UT_ATTACHMENT_INDIC2', 'IMPACT2', 'ORIGIN2', 'IS_CUSTOMER_VISIBLE2', 'TICKETS_COUNT2', 'INCOMING_INDIC2', 'BACKLOG_INDIC2', 'DISPOSED_INDIC2', 'TS_INDIC2', 'SS_INDIC2', 'OIB_INDIC2', 'STATE_ASSIGN_INDIC2', 'STATE_CLOSE_INDIC2', 'STATE_DUPLICATE_INDIC2', 'STATE_FORWARD_INDIC2', 'STATE_HELD_INDIC2', 'STATE_INFO_INDIC2', 'STATE_JUNK_INDIC2', 'STATE_MORE_INDIC2', 'STATE_NEW_INDIC2', 'STATE_OPEN_INDIC2', 'STATE_POSTPONE_INDIC2', 'STATE_RESOLVE_INDIC2', 'STATE_SUBMIT_INDIC2', 'STATE_UNREP_INDIC2', 'STATE_VERIFY_INDIC2', 'STATE_WAIT_INDIC2', 'CFR_INDIC2', 'S12RD_INDIC2', 'S123RD_INDIC2', 'MISSING_SS_EVAL_INDIC2', 'S123_INDIC2', 'S12_INDIC2', 'RNE_INDIC2', 'UPDATED_BY2', 'DEV_ESCAPE_ACTIVITY2', 'RELEASED_CODE2', 'TEST_EDP_ACTIVITY2', 'TEST_EDP_PHASE2', 'RESOLVER_ANALYSIS_INDIC2', 'SUBMITTER_ANALYSIS_INDIC2', 'EDP_ANALYSIS_INDIC2', 'RETI_ANALYSIS_INDIC2', 'DESIGN_REVIEW_ESCAPE_INDIC2', 'STATIC_ANALYSIS_ESCAPE_INDIC2', 'FUNC_TEST_ESCAPE_INDIC2', 'SELECT_REG_ESCAPE_INDIC2', 'CODE_REVIEW_ESCAPE_INDIC2', 'UNIT_TEST_ESCAPE_INDIC2', 'DEV_ESCAPE_INDIC2', 'FEATURE_TEST_ESCAPE_INDIC2', 'REG_TEST_ESCAPE_INDIC2', 'SYSTEM_TEST_ESCAPE_INDIC2', 'SOLUTION_TEST_ESCAPE_INDIC2', 'INT_TEST_ESCAPE_INDIC2', 'GO_TEST_ESCAPE_INDIC2', 'COMPLETE_ESCAPE_INDIC2', 'SR_CNT2', 'PSIRT_INDIC2', 'BADCODEFLAG2',  'RISK_OWNER2', 'SIR2', 'PSIRT_FLAG2', 'URC_DISPOSED_INDIC2', 'CLOSED_DISPOSED_INDIC2', 'REGRESSION_BUG_FLAG2', 'complete2']
	#new_feature_columns = ['DUPLICATE_OF', 'Headline2', 'ENCL-Description2', 'DE_MANAGER_USERID2', 'SEVERITY_CODE2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'FEATURE2', 'IMPACT2', 'ORIGIN2', 'RISK_OWNER2', 'complete2'] #Works the best
	new_feature_columns = ['DUPLICATE_OF', 'Headline2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'complete2']
	#PROduct, Projetc , component, headline
	#feature_columns_to_use = ['IDENTIFIER', 'Headline', 'ENCL-Description', 'DE_MANAGER_USERID', 'SEVERITY_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'IMPACT', 'ORIGIN', 'RISK_OWNER', 'complete1'] #Works the best
	feature_columns_to_use = ['IDENTIFIER', 'Headline', 'PROJECT', 'PRODUCT', 'COMPONENT', 'complete1']
	for i in range(0, len(feature_columns_to_use)):
		print(i)
		df[feature_columns_to_use[i]] = row[feature_columns_to_use[i]]
	return df

def get_test_data(df, test_df):
	#req_data = pd.DataFrame()
	#for index, row in test_df.iterrows():
		#print(row)
	df2 =  fetch_prep_data(df, test_df) #test_df)
		#print(df2)
		#req_data = req_data.append(df2)
	return df2 #req_data



def load_data(db, collection, is_train):
	#collection = db['BugDupsTrainSet_all_639_1968_new']
	cursor = collection.find({})
	df =  pd.DataFrame(list(cursor))
	df['complete1'] = df["Headline"].astype(str) #+ " " + df["ENCL-Description"].astype(str)
	df['complete2'] = df["Headline2"].astype(str) #+ " " + df["ENCL-Description2"].astype(str)
	if(is_train):
		return df
	else:
		a = df[df['SUBMITTED_DATE'] >= str('2018-01')]
		#print(a['SUBMITTED_DATE'])
		return df[df['SUBMITTED_DATE'] >= str('2018-01')]


cluster_id = 3
coll_name = "BugDupsTrainSet_" + str(cluster_id)
key = "csap_prod_database"
db = get_db(settings, key)
collection = db[coll_name]


embedding_vecor_length = 32
model1 = Sequential()
#print(X_test[0], len(X_test[0]))
model1.add(Embedding(3458230+1, embedding_vecor_length, input_length= 6))#len(X_test[0])))
model1.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model1.add(MaxPooling1D(pool_size=2))
model1.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())
#Model predictions
filename = '/data/ingestion/bugDuplicates/model3_lstm_cnn_cat_' + str(cluster_id) + '.h5'
model1 = load_model(filename) #model = load_model('/data/ingestion/bugDuplicates/model3_lstm_cnn_cat.h5')


filename = '/data/ingestion/bugDuplicates/w2vmodel_' + str(cluster_id) + '.bin'
w2vmodel = Word2Vec.load(filename)
f = '/data/ingestion/bugDuplicates/vocab_model_' + str(cluster_id) + '.json'
vocabulary = json.load(open(f, 'r'))
thefile = '/data/ingestion/bugDuplicates/inv_vocab_model_' + str(cluster_id) + '.json'
with open (thefile, 'rb') as fp:
	inverse_vocabulary = pickle.load(fp)


thefile = "/data/ingestion/bugDuplicates/embeddings_model_" + str(cluster_id) + '.json'
with open (thefile, 'rb') as fp:
	embeddings = pickle.load(fp)


n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 1 #25
def exponent_neg_manhattan_distance(left, right):
	return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')
embedding_layer = Embedding(len(embeddings), embedding_dim, weights = [embeddings], input_length=max_seq_length, trainable=False)
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
# load weights into new model
filename = '/data/ingestion/bugDuplicates/text_model_' + str(cluster_id) + '.h5'
malstm.load_weights(filename)


#We have a dataset of 20000+ bugs for ...
e = load_data(db, collection, False)
df = load_data(db, collection, False)
#print(df[['DUPLICATE_OF', 'PROJECT','PROJECT2', 'PRODUCT', 'PRODUCT2', 'COMPONENT', 'COMPONENT2', 'Headline', 'Headline2']])

w = 10
v = 10
cut_off = 95
final_df = pd.DataFrame()
duplicate_bugs_length = sum(df['is_duplicate']==1)

for j in range(0, duplicate_bugs_length, w):
	print(j, duplicate_bugs_length)
	df = e.copy()
	b1 = df.iloc[j:j+w, ] #26912-4, 26906-1, , 26928
	duplicate_of = list(b1['DUPLICATE_OF'])
	is_duplicates = list(b1['is_duplicate'])
	#print(b1[['IDENTIFIER', 'Headline', 'ENCL-Description', 'PROJECT', 'PRODUCT', 'COMPONENT']])
	c = get_test_data(df, b1)
	df1 = c
	#print(df1[['DUPLICATE_OF', 'PROJECT','PROJECT2', 'PRODUCT', 'PRODUCT2', 'COMPONENT', 'COMPONENT2', 'Headline', 'Headline2']])
	X_test, Y_test = build_data_cat(df1, False)
	cat_predictions = test_model_cat(model1, X_test, Y_test, cluster_id)
	vocabulary, w2vmodel, words, test_df = build_test_data_text(w2vmodel, vocabulary, inverse_vocabulary, df1, cluster_id)
	text_predictions = test_model_text(embeddings, vocabulary, w2vmodel, words, test_df, cluster_id)
		d = pd.DataFrame()
		d['IDENTIFIER'] = df1['IDENTIFIER']
		d['DUPLICATE_OF'] = df1['DUPLICATE_OF']
		p = []
		for i in cat_predictions:
			p.append(i[0]*100)
		d['pred_cat'] = p
		p = []
		for i in text_predictions:
			p.append(i[0]*100)
		d['pred_text'] = p
		a = d[d['pred_cat'] > cut_off]
		#for each unique identifier get these...
		unique_identifiers = list(a['IDENTIFIER'].unique())
		id_num = 0
		for identifier in unique_identifiers:
			result = a[a['IDENTIFIER'] == identifier]
			result = result.drop_duplicates(subset='DUPLICATE_OF', keep="last")
			result = result.sort_values(['pred_text'], ascending=[0])
			df2 = pd.DataFrame()
			l = []
			l.append(result.iloc[0,]['IDENTIFIER'])
			df2['IDENTIFIER'] = l
			#df2['PRODUCT'] = result.iloc[0,]['PRODUCT']
			#df2['PROJECT'] = b1['PROJECT']
			#df2['Headline'] = b1['Headline']
			df2['DUPLICATE_LIST'] = ' '.join(list(result.iloc[0:v]['DUPLICATE_OF']))
			df2['PROBABILITIES'] = ' '.join(str(x) for x in list(result.iloc[0:v]['pred_text']))
			if(is_duplicates[id_num] == 1):
				df2['actual_duplicate'] = duplicate_of[id_num]
			else:
				df2['actual_duplicate'] = ""
			final_df = final_df.append(df2)
			id_num = id_num + 1


#final_df.to_csv('Final_df_510_bugs.csv', encoding='utf-8')
res_coll_name = 'BugDupsTestSet_' + str(cluster_id) + '_headline_results'
collection = db[res_coll_name]

final_df = final_df.drop_duplicates(subset='IDENTIFIER', keep="last")
final_df.reset_index(drop = True, inplace = True)
records = json2.loads(final_df.T.to_json(date_format='iso')).values()

collection.create_index([("IDENTIFIER", pymongo.ASCENDING)], unique=True)
collection.insert(records)


#Verifying the accuracy of the model
dup_lists = list(final_df['DUPLICATE_LIST'].unique())
actuals_list list(final_df['actual_duplicate'])
i = 0
for dup_list in dup_lists:
	dup_list = dup_list.split(' ')
	if(actuals_list[i] in dup_list):
		acc = acc + 1
	i = i + 1


'''
view_id = "673"
query_id = "2061"
coll_name = "BugDupsTestSet_" + str(view_id) + "_" + str(query_id)
key = "csap_prod_database"
db = get_db(settings, key)
cluster_id = 3
collection = db[coll_name] #db['BugDupsTrainSet_all_639_1968_new'] #9, 18, 29, 15, 5

#We have a dataset of 20000+ bugs for ...
e = load_data(db, collection)
df = load_data(db, collection)
print(df[['DUPLICATE_OF', 'PROJECT','PROJECT2', 'PRODUCT', 'PRODUCT2', 'COMPONENT', 'COMPONENT2', 'Headline', 'Headline2']])

#Let us choose a bug, say...
b1 = df.iloc[9, :] #9, 18, 21, 15
duplicate_of = b1['DUPLICATE_OF']
print(b1[['IDENTIFIER', 'Headline', 'ENCL-Description', 'PROJECT', 'PRODUCT', 'COMPONENT']])

c = get_test_data(df, b1)
df1 = c
print(df1[['DUPLICATE_OF', 'PROJECT','PROJECT2', 'PRODUCT', 'PRODUCT2', 'COMPONENT', 'COMPONENT2', 'Headline', 'Headline2']])

#Now let's run the bug duplicates model for this bug against all the existing 20000+ bugs
X_test, Y_test = build_data_cat(df1, False)
cat_predictions = test_model_cat(X_test, Y_test, cluster_id)
vocabulary, w2vmodel, words, test_df = build_test_data_text(df1, cluster_id)
text_predictions = test_model_text(vocabulary, w2vmodel, words, test_df, cluster_id)
d = pd.DataFrame()
d['IDENTIFIER'] = df1['IDENTIFIER']
d['DUPLICATE_OF'] = df1['DUPLICATE_OF']
p = []
for i in cat_predictions:
	p.append(i[0]*100)

d['pred_cat'] = p
p = []
for i in text_predictions:
	p.append(i[0]*100)

d['pred_text'] = p
#print(d)
a = d[d['pred_cat'] > 98]
#print(a)

#We will sort the results based on probability
result = a.sort_values(['pred_text'], ascending=[0])
print(result.iloc[0:5,])






view_id = "436"
query_id = "1452"
coll_name = "BugDupsTestSet_" + str(view_id) + "_" + str(query_id)
key = "csap_prod_database"
db = get_db(settings, key)
cluster_id = 3
collection = db[coll_name] #db['BugDupsTrainSet_all_639_1968_new']
print(collection)
df = load_data(db, collection)
b = df

#We have a dataset of 900+ bugs for ...
print(df[['IDENTIFIER', 'DUPLICATE_OF', 'Headline', 'PROJECT', 'PRODUCT', 'COMPONENT', 'Headline2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2']])

#Let us choose a bug, say...
b1 = df.iloc[, :] #8, 9
print(b1[['IDENTIFIER', 'DUPLICATE_OF', 'Headline', 'PROJECT', 'PRODUCT', 'COMPONENT']])

a = get_test_data(b, b1)
df1 = a

#Now let's run the bug duplicates model for this bug against all the existing 900+ bugs
X_test, Y_test = build_data_cat(df1, False)
cat_predictions = test_model_cat(X_test, Y_test, cluster_id)
vocabulary, w2vmodel, words, test_df = build_test_data_text(df1, cluster_id)
text_predictions = test_model_text(vocabulary, w2vmodel, words, test_df, cluster_id)
d = pd.DataFrame()
d['IDENTIFIER'] = df1['IDENTIFIER']
d['DUPLICATE_OF'] = df1['DUPLICATE_OF']
p = []
for i in cat_predictions:
	p.append(i[0])

d['pred_cat'] = p
p = []
for i in text_predictions:
	p.append(i[0])

d['pred_text'] = p
#print(d)
a = d[d['pred_cat'] > 0.98]
#print(a)

#We will sort the results based on probability
result = a.sort_values(['pred_text'], ascending=[0])
print(result)




#ensemble_predictions = stacking_test(d, cluster_id)
#d['pred_ensemble'] = list(ensemble_predictions)
#d['Headline'] = df['Headline2']
#d['PRODUCT'] = df['PRODUCT2']
#d['COMPONENT'] = df['COMPONENT2']

#Subset the data based on the categorical model threshold and run the text classification only on those which are above certain threshold.
'''