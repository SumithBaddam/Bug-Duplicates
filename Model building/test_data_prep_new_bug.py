#Trying for new bug on the fly model.
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

def build_test_data_text(sentence, w2vmodel, words, vocabulary, inverse_vocabulary, cluster_id):
	q1=[0]
	a = -1
	s = 0
	q2n = []  # q2n -> question numbers representation
	for word in text_to_word_list(sentence):
		# Check for unwanted words
		if word.lower() in stops and word.lower() not in words: #word2vec.vocab:
			continue
		if word not in vocabulary:
			s = 1
		else:
			q2n.append(vocabulary[word])
	q1[a] = q2n
	return q1


def get_test_data(df, test_df, w2vmodel, words, vocabulary, inverse_vocabulary, cluster_id):
	c = pd.DataFrame()
	new_feature_columns = ['DUPLICATE_OF', 'Headline2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'complete2', 'is_duplicate']
	feature_columns_to_use = ['IDENTIFIER', 'Headline', 'PROJECT', 'PRODUCT', 'COMPONENT']
	#complete1 = [test_df['complete1']]*df.shape[0]
	for col in new_feature_columns:
		c[col] = df[col]
	for col in feature_columns_to_use:
		c[col] = test_df[col]
	sentence = test_df["Headline"] + " " + test_df["ENCL-Description"]
	complete1_int = build_test_data_text(sentence, w2vmodel, words, vocabulary, inverse_vocabulary, cluster_id)
	complete1 = complete1_int*df.shape[0]
	c['complete1'] = complete1
	return c


def load_data(db, collection, is_train):
	#collection = db['BugDupsTrainSet_all_639_1968_new']
	cursor = collection.find({})
	df =  pd.DataFrame(list(cursor))
	df["SUBMITTED_DATE"] = pd.to_datetime(df['SUBMITTED_DATE'], unit='ms')
	if(is_train):
		return df
	else:
		a = df[df['SUBMITTED_DATE'] >= str('2018-01')]
		return df[df['SUBMITTED_DATE'] >= str('2018-01')]


cluster_id = 3
coll_name = "BugDupsTrainSet_3_complete"# + str(cluster_id) + "_2"
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
#df = load_data(db, collection, False)


cut_off = 98
final_df = pd.DataFrame()
duplicate_bugs_length = sum(df['is_duplicate'] == 1)
j=4
#for j in range(0, duplicate_bugs_length):
print(j, duplicate_bugs_length)
df = e.copy()
b1 = df.iloc[j, ]
print(b1[['IDENTIFIER', 'Headline', 'ENCL-Description', 'PRODUCT', 'PROJECT', 'COMPONENT']])
duplicate_of = b1['DUPLICATE_OF']
is_duplicates = b1['is_duplicate']
c = get_test_data(df, b1, w2vmodel, words, vocabulary, inverse_vocabulary, cluster_id)
df1 = c
X_test, Y_test = build_test_data_cat(df1)
cat_predictions = test_model_cat(model1, X_test, Y_test, cluster_id)
#vocabulary, w2vmodel, words, test_df = build_test_data_text(w2vmodel, vocabulary, inverse_vocabulary, df1, cluster_id)
text_predictions = test_model_text(embeddings, vocabulary, w2vmodel, words, df1, cluster_id)
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
	df2['DUPLICATE_LIST'] = ' '.join(list(result.iloc[0:v]['DUPLICATE_OF']))
	df2['PROBABILITIES'] = ' '.join(str(x) for x in list(result.iloc[0:v]['pred_text']))
	if(is_duplicates == 1):
		df2['actual_duplicate'] = duplicate_of
	else:
		df2['actual_duplicate'] = ""
	final_df = final_df.append(df2)














res_coll_name = 'BugDupsTestSet_' + str(cluster_id) + '_description_complete_results'
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
for dup_list in dup_lists:
	a = dup_list.split(' ')
	if(actual_dups[i] in a):
		acc = acc + 1
	i = i + 1

collection = db['BugDupsTestSet_3_results_d']
cursor = collection.find({})
results_df =  pd.DataFrame(list(cursor))
results_df = results_df[results_df['pred_cat'] > 0.98]
results_df = results_df.sort_values(['pred_text'], ascending=[0])

#CSCvh40104, CSCvh44844, CSCvh44238, CSCvh53571, CSCvh83909, CSCvh98317, CSCvh44631-18, CSCvh44466-4
for index, row in results_df.iterrows():
	row['IDENTIFIER']
