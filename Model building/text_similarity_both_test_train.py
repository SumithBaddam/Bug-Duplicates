#https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07
#https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb
from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import nltk
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import seaborn as sns
import itertools
import datetime
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
import sys
sys.path.insert(0, "/auto/vgapps-cstg02-vapps/analytics/csap/ingestion/scripts/")
from Utils import *
import datetime
import configparser
import shutil
import argparse
import json
import pickle

#Setting up the config.ini file parameters
settings = configparser.ConfigParser()
settings.read('/auto/vgapps-cstg02-vapps/analytics/csap/ingestion/scripts/config.ini')

#Parser options
options = None

stops = set(stopwords.words('english'))
max_seq_length = 150 
text_cols = ['complete1', 'complete2']
embedding_dim = 150 #300

chars = ["?", "'s", ">", "<", ",", ":", "'", "''", "--", "`", "``", "...", "", "!", "#", '"', '$', '%', '&', '(', ')', '*', '+', '-', '.', '/', ';', '=', '@', '[', '\\', ']', '^', '_', '{', '}', '|', '~', '\t', '\n', '']
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

def parse_options():
    parser = argparse.ArgumentParser(description="""Script to predict Potential CFD testing data""")
    parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
    parser.add_argument("--testdate", default="201706", help ='yyyy-mm', type=str, metavar='t')
    parser.add_argument("--cluster", default="", help ='Comma seperated cluster-ids', type=str, metavar='c')
    args = parser.parse_args()
    
    return args

#Loading the training data
def load_data(db, key, collection):
	key = "csap_prod_database"
	db = get_db(settings, key)
	#collection = db['BugDupsTrainSet_all_639_1968_new']
	cursor = collection.find({})
	df =  pd.DataFrame(list(cursor))
	df['complete1'] = df["Headline"].astype(str) + " " + df["ENCL-Description"].astype(str)
	df['complete2'] = df["Headline2"].astype(str) + " " + df["ENCL-Description2"].astype(str)
	return df


def build_data(train_df):
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
			print(index)
			if(type(row['complete1']) != float):
				sentences.append(text_to_word_list(row["complete1"]))
			if(type(row['complete2']) != float):
				sentences.append(text_to_word_list(row["complete2"]))

	# train model
	model = Word2Vec(sentences, min_count=1, size = 150)
	# summarize the loaded model
	print(model)
	# summarize vocabulary
	words = list(model.wv.vocab)
	#print(words)
	# access vector for one word
	print(model['sentence'])
	# save model
	model.save('/users/sumreddi/model_3.bin')

	c=0
	q1=[0]*(train_df.shape[0])
	q2=[0]*(train_df.shape[0])
	# Iterate over the questions only of both training and test datasets
	for dataset in [train_df]:#[train_df, test_df]:
		for index, row in dataset.iterrows():
			print(index)
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
	f = "/users/sumreddi/vocab_model_3.json" #str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/indexes_cluster_' +str(p)+'.json'
	f = open(f,'w')
	json1 = json.dumps(vocabulary, indent=4)
	f.write(json1)
	f.close()
	thefile = "/users/sumreddi/inv_vocab_model_3.json" #str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/top_words_cluster_' +str(p)+'.txt'
	with open(thefile, 'wb') as fp:
		pickle.dump(inverse_vocabulary, fp, protocol=2)

	return vocabulary, model, train_df


def build_model(vocabulary, model, validation_size, train_df):
	words = list(model.wv.vocab)
	embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
	embeddings[0] = 0  # So that the padding will be ignored

	# Build the embedding matrix
	for word, index in vocabulary.items():
		print(index)
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
		print(dataset)
		dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

	# Model variables
	n_hidden = 50
	gradient_clipping_norm = 1.25
	batch_size = 64
	n_epoch = 15

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

	print("Starting to train")

	malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
								validation_data=([X_validation['left'], X_validation['right']], Y_validation))
	print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))
	# serialize weights to HDF5
	malstm.save_weights("/users/sumreddi/model3_new.h5")
	print("Saved model to disk")

	return malstm


def build_test_data(test_df):
	model = Word2Vec.load('/users/sumreddi/model_3.bin')
	print("Loaded the W2V")
	words = list(model.wv.vocab)

	f = "/users/sumreddi/vocab_model_3.json" #str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/indexes_cluster_' +str(cluster_id)+'.json'
	vocabulary = json.load(open(f, 'r'))

	thefile = "/users/sumreddi/inv_vocab_model_3.json" #str(settings.get("Potential_CFD","temp_path_mod_potCFD")) + '/top_words_cluster_' +str(cluster_id)+'.txt'
	with open (thefile, 'rb') as fp:
		inverse_vocabulary = pickle.load(fp)

	c=0
	q1=[0]*(test_df.shape[0])
	q2=[0]*(test_df.shape[0])
	a = -1
	# Iterate over the questions only of both training and test datasets
	for dataset in [test_df]:#[train_df, test_df]:
		for index, row in dataset.iterrows():
			print(index)
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
						a = 1
					else:
						q2n.append(vocabulary[word])
				# Replace questions as word to question as number representation
				if(c%2 != 0):
					 q1[index] = q2n
				else:
					 q2[index] = q2n
				#dataset.set_value(index, question, q2n)

	test_df['complete1'] = q1
	test_df['complete2'] = q2
	return vocabulary, model, test_df 


def test_model(vocabulary, model, test_df):
	embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
	embeddings[0] = 0  # So that the padding will be ignored

	# Build the embedding matrix
	for word, index in vocabulary.items():
		print(index)
		if word in words:#word2vec.vocab:
			embeddings[index] = model[word]#word2vec.word_vec(word)

	del model

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
	n_hidden = 50
	gradient_clipping_norm = 1.25
	batch_size = 64
	n_epoch = 1 #25
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

	# load weights into new model
	malstm.load_weights("/users/sumreddi/model3_new.h5")
	print("Loaded model from disk")

	# evaluate loaded model on test data
	predictions = malstm.predict([X_test['left'], X_test['right']])

	return predictions



def plot_model(malstm_trained):
	# Plot accuracy
	plt.plot(malstm_trained.history['acc'])
	plt.plot(malstm_trained.history['val_acc'])
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper left')
	plt.show()

	# Plot loss
	plt.plot(malstm_trained.history['loss'])
	plt.plot(malstm_trained.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper right')
	plt.show()


def main():
    options = parse_options()
    if(options.env == "Prod"):
        key = "csap_prod_database"
    
    else:
        key = "csap_stage_database"

	db = get_db(settings, key)

	if(options.train == True):
		collection = db['BugDupsTrainSet_all_3']
		train_df = load_data(db, key, collection)
		vocabulary, w2vmodel, train_df = build_data(train_df)
		model = build_model(vocabulary, w2vmodel, 20000, train_df)
	else:
		collection = db['BugDupsTrainSet_all_639_1968_new']
		test_df = load_data(db, key, collection)
		vocabulary, w2vmodel, test_df = build_test_data(test_df)
		predictions = test_model(vocabulary, w2vmodel, test_df)


if __name__ == "__main__":
    main()