#Works perfect for a group of bugs from the dataset only for 2018. Training is on entire dataset and testing on 2018. This uses precomputed complete1 values.
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


def get_test_data(df, test_df):
	c = pd.DataFrame()
	new_feature_columns = ['DUPLICATE_OF', 'Headline2', 'ENCL-Description2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'complete2', 'is_duplicate']
	feature_columns_to_use = ['IDENTIFIER', 'Headline', 'ENCL-Description', 'PROJECT', 'PRODUCT', 'COMPONENT']
	complete1 = [test_df['complete1']]*df.shape[0]
	for col in new_feature_columns:
		c[col] = df[col]
	for col in feature_columns_to_use:
		c[col] = test_df[col]
	c['complete1'] = complete1
	return c

def load_data(db, collection, is_train):
	#collection = db['BugDupsTrainSet_all_639_1968_new']
	cursor = collection.find({})
	df =  pd.DataFrame(list(cursor))
	df["SUBMITTED_DATE"] = pd.to_datetime(df['SUBMITTED_DATE'], unit='ms')
	#df['complete1'] = df["Headline"].astype(str) + " " + df["ENCL-Description"].astype(str)
	#df['complete2'] = df["Headline2"].astype(str) + " " + df["ENCL-Description2"].astype(str)
	if(is_train):
		return df
	else:
		#a = df[df['SUBMITTED_DATE'] >= str('2018-01')]
		return df[df['SUBMITTED_DATE'] >= str('2018-01')]


cluster_id = 3
coll_name = "BugDupsTrainSet_" + str(cluster_id) + "_complete"
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

words = list(w2vmodel.wv.vocab)

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
'''
a = -1
for index, row in e.iterrows():
	a = a + 1
	print(a)
	e['complete1'].iloc[a,] = list(eval(e['complete1'].iloc[a,])) #[int(s) for s in df1['complete1'].iloc[a,][1:-1].split(', ')]
	e['complete2'].iloc[a,] = list(eval(e['complete2'].iloc[a,])) #[int(s) for s in df1['complete2'].iloc[a,][1:-1].split(', ')]
'''

w = 1
v = 10
cut_off = 98
final_df = pd.DataFrame()
duplicate_bugs_length = sum(df['is_duplicate'] == 1)
j=0

for j in range(0, duplicate_bugs_length):
	print(j, duplicate_bugs_length)
	df = e.copy()
	b1 = df.iloc[j, ] #26912-4, 26906-1, , 26928
	duplicate_of = b1['DUPLICATE_OF']
	is_duplicates = b1['is_duplicate']
	#print(b1[['IDENTIFIER', 'Headline', 'ENCL-Description', 'PROJECT', 'PRODUCT', 'COMPONENT']])
	#df2 = fetch_prep_data(df, test_df, w2vmodel, vocabulary, inverse_vocabulary, cluster_id)
	c = get_test_data(df, b1)
	df1 = c
	#print(df1[['DUPLICATE_OF', 'PROJECT','PROJECT2', 'PRODUCT', 'PRODUCT2', 'COMPONENT', 'COMPONENT2', 'Headline', 'Headline2']])
	X_test, Y_test = build_test_data_cat(df1)
	cat_predictions = test_model_cat(model1, X_test, Y_test, cluster_id)
	#vocabulary, w2vmodel, words, test_df = build_test_data_text(w2vmodel, vocabulary, inverse_vocabulary, df1, cluster_id)
	text_predictions = test_model_text(embeddings, vocabulary, w2vmodel, words, df1, cluster_id)
	d = pd.DataFrame()
	d['IDENTIFIER'] = df1['IDENTIFIER']
	d['DUPLICATE_OF'] = df1['DUPLICATE_OF']
	d['Headline'] = df1['Headline']
	d['ENCL-Description'] = df1['ENCL-Description']
	p = []
	for i in cat_predictions:
		p.append(i[0]*100)
	d['pred_cat'] = p
	p = []
	for i in text_predictions:
		p.append(i[0]*100)
	d['pred_text'] = p
	result = d[d['pred_cat'] >= cut_off]
	result = result.drop_duplicates(subset='DUPLICATE_OF', keep="last")
	result = result.sort_values(['pred_text'], ascending=[0])
	if(result.shape[0] > 10):
		v = 10
	else:
		v = result.shape[0]
	if(v != 0):
		df2 = pd.DataFrame()
		l = []
		l.append(result.iloc[0,]['IDENTIFIER'])
		df2['IDENTIFIER'] = l
		df2['Headline'] = df1['Headline'].iloc[0]
		df2['ENCL-Description'] = df1['ENCL-Description'].iloc[0]
		#df2['PRODUCT'] = result.iloc[0,]['PRODUCT']
		#df2['PROJECT'] = b1['PROJECT']
		#df2['Headline'] = b1['Headline']
		df2['DUPLICATE_LIST'] = ' '.join(list(result.iloc[0:v]['DUPLICATE_OF']))
		df2['PROBABILITIES'] = ' '.join(str(x) for x in list(result.iloc[0:v]['pred_text']))
		if(is_duplicates == 1):
			df2['actual_duplicate'] = duplicate_of
		else:
			df2['actual_duplicate'] = ""
		final_df = final_df.append(df2)


res_coll_name = 'BugDupsTestSet_' + str(cluster_id) + '_results_2'
collection = db[res_coll_name]

final_df = final_df.drop_duplicates(subset='IDENTIFIER', keep="last")
final_df.reset_index(drop = True, inplace = True)
records = json2.loads(final_df.T.to_json(date_format='iso')).values()

collection.create_index([("IDENTIFIER", pymongo.ASCENDING)], unique=True)
collection.insert(records)


#Verifying the accuracy of the model
dup_lists = list(final_df['DUPLICATE_LIST'].unique())
actual_dups = list(final_df['actual_duplicate'])
i = 0
acc = 0
ids = []
for dup_list in dup_lists:
	a = dup_list.split(' ')
	if(actual_dups[i] in a):
		ids.append(i)
		acc = acc + 1
	i = i + 1

#Find bugs from res_coll_name which have more than 98% match. For those bugs run the test_data_prep_new. Now the dup_list for new bug contains original bug.

collection = db['BugDupsTestSet_3_results_d']
cursor = collection.find({})
results_df =  pd.DataFrame(list(cursor))
results_df = results_df[results_df['pred_cat'] > 0.98]
results_df = results_df.sort_values(['pred_text'], ascending=[0])

#[4, 18, 23, 27, 33, 34, 204, 290, 339, 443, 510, 651, 666, 683, 873, 905, 992, 1052, 1118, 1181, 1262, 1283, 1461, 1483, 1781, 1794, 1840]
#CSCvh40104, CSCvh44844, CSCvh44238, CSCvh53571, CSCvh83909, CSCvh98317, CSCvh44631-18, CSCvh44466-4
for index, row in results_df.iterrows():
	row['IDENTIFIER']


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