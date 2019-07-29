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
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import datetime
import numpy as np
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
import datetime
import configparser
import shutil
import argparse
import json
import pickle

#Setting up the config.ini file parameters
settings = configparser.ConfigParser()
settings.read('/data/ingestion/config.ini')

#Parser options
options = None


#Data transformation and imputation of missing values
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)        
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

def parse_options():
    parser = argparse.ArgumentParser(description="""Script to predict Potential CFD testing data""")
    parser.add_argument("--env", default="stage", help='Environment', type=str, metavar='E')
    parser.add_argument("--train", default=True, help ='True/False', type=str, metavar='t')
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
	return df


def build_data(df, is_train):
	new_feature_columns = ['DE_MANAGER_USERID2', 'SEVERITY_CODE2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'FEATURE2', 'RELEASE_NOTE2', 'SA_ATTACHMENT_INDIC2', 'CR_ATTACHMENT_INDIC2', 'UT_ATTACHMENT_INDIC2', 'IMPACT2', 'ORIGIN2', 'IS_CUSTOMER_VISIBLE2', 'TICKETS_COUNT2', 'INCOMING_INDIC2', 'BACKLOG_INDIC2', 'DISPOSED_INDIC2', 'TS_INDIC2', 'SS_INDIC2', 'OIB_INDIC2', 'STATE_ASSIGN_INDIC2', 'STATE_CLOSE_INDIC2', 'STATE_FORWARD_INDIC2', 'STATE_HELD_INDIC2', 'STATE_INFO_INDIC2', 'STATE_JUNK_INDIC2', 'STATE_MORE_INDIC2', 'STATE_NEW_INDIC2', 'STATE_OPEN_INDIC2', 'STATE_POSTPONE_INDIC2', 'STATE_RESOLVE_INDIC2', 'STATE_SUBMIT_INDIC2', 'STATE_UNREP_INDIC2', 'STATE_VERIFY_INDIC2', 'STATE_WAIT_INDIC2', 'CFR_INDIC2', 'S12RD_INDIC2', 'S123RD_INDIC2', 'MISSING_SS_EVAL_INDIC2', 'S123_INDIC2', 'S12_INDIC2', 'RNE_INDIC2', 'UPDATED_BY2', 'DEV_ESCAPE_ACTIVITY2', 'RELEASED_CODE2', 'TEST_EDP_ACTIVITY2', 'TEST_EDP_PHASE2', 'RESOLVER_ANALYSIS_INDIC2', 'SUBMITTER_ANALYSIS_INDIC2', 'EDP_ANALYSIS_INDIC2', 'RETI_ANALYSIS_INDIC2', 'DESIGN_REVIEW_ESCAPE_INDIC2', 'STATIC_ANALYSIS_ESCAPE_INDIC2', 'FUNC_TEST_ESCAPE_INDIC2', 'SELECT_REG_ESCAPE_INDIC2', 'CODE_REVIEW_ESCAPE_INDIC2', 'UNIT_TEST_ESCAPE_INDIC2', 'DEV_ESCAPE_INDIC2', 'FEATURE_TEST_ESCAPE_INDIC2', 'REG_TEST_ESCAPE_INDIC2', 'SYSTEM_TEST_ESCAPE_INDIC2', 'SOLUTION_TEST_ESCAPE_INDIC2', 'INT_TEST_ESCAPE_INDIC2', 'GO_TEST_ESCAPE_INDIC2', 'COMPLETE_ESCAPE_INDIC2', 'PSIRT_INDIC2', 'BADCODEFLAG2',  'RISK_OWNER2', 'SIR2', 'PSIRT_FLAG2', 'URC_DISPOSED_INDIC2', 'CLOSED_DISPOSED_INDIC2', 'REGRESSION_BUG_FLAG2']
	feature_columns_to_use = ['DE_MANAGER_USERID', 'SEVERITY_CODE', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'RELEASE_NOTE', 'SA_ATTACHMENT_INDIC', 'CR_ATTACHMENT_INDIC', 'UT_ATTACHMENT_INDIC', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'TICKETS_COUNT', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'TS_INDIC', 'SS_INDIC', 'OIB_INDIC', 'STATE_ASSIGN_INDIC', 'STATE_CLOSE_INDIC', 'STATE_FORWARD_INDIC', 'STATE_HELD_INDIC', 'STATE_INFO_INDIC', 'STATE_JUNK_INDIC', 'STATE_MORE_INDIC', 'STATE_NEW_INDIC', 'STATE_OPEN_INDIC', 'STATE_POSTPONE_INDIC', 'STATE_RESOLVE_INDIC', 'STATE_SUBMIT_INDIC', 'STATE_UNREP_INDIC', 'STATE_VERIFY_INDIC', 'STATE_WAIT_INDIC', 'CFR_INDIC', 'S12RD_INDIC', 'S123RD_INDIC', 'MISSING_SS_EVAL_INDIC', 'S123_INDIC', 'S12_INDIC', 'RNE_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY',  'TEST_EDP_PHASE', 'RESOLVER_ANALYSIS_INDIC', 'SUBMITTER_ANALYSIS_INDIC', 'EDP_ANALYSIS_INDIC', 'RETI_ANALYSIS_INDIC', 'DESIGN_REVIEW_ESCAPE_INDIC', 'STATIC_ANALYSIS_ESCAPE_INDIC', 'FUNC_TEST_ESCAPE_INDIC', 'SELECT_REG_ESCAPE_INDIC', 'CODE_REVIEW_ESCAPE_INDIC', 'UNIT_TEST_ESCAPE_INDIC', 'DEV_ESCAPE_INDIC', 'FEATURE_TEST_ESCAPE_INDIC', 'REG_TEST_ESCAPE_INDIC', 'SYSTEM_TEST_ESCAPE_INDIC', 'SOLUTION_TEST_ESCAPE_INDIC', 'INT_TEST_ESCAPE_INDIC', 'GO_TEST_ESCAPE_INDIC', 'COMPLETE_ESCAPE_INDIC', 'PSIRT_INDIC',  'BADCODEFLAG', 'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'URC_DISPOSED_INDIC', 'CLOSED_DISPOSED_INDIC', 'REGRESSION_BUG_FLAG']
	#AGE, SR_CNT, LIFECYCLE_STATE_CODE -- removed, (DE_MANAGER, COMPONENT, PROJECT, PRODUCT, FEATURE, IMPACT, ORIGIN, UPDATED_BY) -- for these just check if they are same or not 1/0.
	avoid_cols = ['DE_MANAGER_USERID', 'COMPONENT', 'PROJECT', 'PRODUCT', 'FEATURE', 'IMPACT', 'ORIGIN', 'UPDATED_BY', 'RISK_OWNER', 'FEATURE', 'DEV_ESCAPE_ACTIVITY']

	cols = [] #feature_columns_to_use + new_feature_columns + ['is_duplicate']
	for i in range(0, len(new_feature_columns)):
		cols.append(feature_columns_to_use[i])
		cols.append(new_feature_columns[i])

	k = 0
	avoid_cols_indexes = []
	for col in cols:
		if col in avoid_cols:
			avoid_cols_indexes.append(k)
		k = k + 1

	cols.append('is_duplicate')
	big_X = df[cols]

	nonnumeric_columns_1 = ['DE_MANAGER_USERID', 'PROJECT', 'PRODUCT', 'COMPONENT', 'FEATURE', 'RELEASE_NOTE', 'IMPACT', 'ORIGIN', 'IS_CUSTOMER_VISIBLE', 'TICKETS_COUNT', 'INCOMING_INDIC', 'BACKLOG_INDIC', 'DISPOSED_INDIC', 'UPDATED_BY', 'DEV_ESCAPE_ACTIVITY', 'RELEASED_CODE', 'TEST_EDP_ACTIVITY', 'TEST_EDP_PHASE', 'BADCODEFLAG',  'RISK_OWNER', 'SIR', 'PSIRT_FLAG', 'REGRESSION_BUG_FLAG']
	nonnumeric_columns_2 = ['DE_MANAGER_USERID2', 'PROJECT2', 'PRODUCT2', 'COMPONENT2', 'FEATURE2', 'RELEASE_NOTE2', 'IMPACT2', 'ORIGIN2', 'IS_CUSTOMER_VISIBLE2', 'TICKETS_COUNT2', 'INCOMING_INDIC2', 'BACKLOG_INDIC2', 'DISPOSED_INDIC2', 'UPDATED_BY2', 'DEV_ESCAPE_ACTIVITY2', 'RELEASED_CODE2', 'TEST_EDP_ACTIVITY2', 'TEST_EDP_PHASE2', 'BADCODEFLAG2', 'RISK_OWNER2', 'SIR2', 'PSIRT_FLAG2', 'REGRESSION_BUG_FLAG2']

	nonnumeric_columns = []
	for i in range(0, len(nonnumeric_columns_1)):
		nonnumeric_columns.append(nonnumeric_columns_1[i])
		nonnumeric_columns.append(nonnumeric_columns_2[i])

	big_X = big_X.replace(np.nan, '', regex=True)
	big_X_imputed = DataFrameImputer().fit_transform(big_X.iloc[:,:])

	le = LabelEncoder()
	for feature in nonnumeric_columns:
		big_X_imputed[feature] = big_X_imputed[feature].astype(str)

	#big_X_imputed = DataFrameImputer().fit_transform(big_X.iloc[:,:])
	#big_X_imputed = big_X_imputed.iloc[30050:30100, :]
	for i in range(0, len(nonnumeric_columns), 2):
		print(i)
		u1 = list(big_X_imputed[nonnumeric_columns[i]].unique())
		u2 = list(big_X_imputed[nonnumeric_columns[i + 1]].unique())
		u1 = u1 + u2
		myset = list(set(u1))
		d = dict()
		for j in range(0, len(myset)):
			d[myset[j]] = j
		big_X_imputed = big_X_imputed.replace({nonnumeric_columns[i]: d})
		big_X_imputed = big_X_imputed.replace({nonnumeric_columns[i + 1]: d})


	train_X = big_X_imputed.iloc[:, :146].as_matrix()
	Y = big_X['is_duplicate']
	X_train = train_X
	if(is_train == True):
		X_train, X_validation, Y_train, Y_validation = train_test_split(train_X, Y, test_size=20000)
		for i in range(0, len(X_train)):
			for k in avoid_cols_indexes:
				if(X_train[i][k] == X_train[i][k + 1]):
					X_train[i][k] = 1
					X_train[i][k + 1] = 1
				else:
					X_train[i][k] = 0
					X_train[i][k + 1] = 1

		for i in range(0, len(X_validation)):
			for k in avoid_cols_indexes:
				if(X_validation[i][k] == X_validation[i][k + 1]):
					X_validation[i][k] = 1
					X_validation[i][k + 1] = 1
				else:
					X_validation[i][k] = 0
					X_validation[i][k + 1] = 1
		Y_train = Y_train.values
		Y_validation = Y_validation.values
		return X_train, Y_train, X_validation, Y_validation
	else:
		X_train = train_X
		for i in range(0, len(X_train)):
			for k in avoid_cols_indexes:
				if(X_train[i][k] == X_train[i][k + 1]):
					X_train[i][k] = 1
					X_train[i][k + 1] = 1
				else:
					X_train[i][k] = 0
					X_train[i][k + 1] = 1

		Y_train = Y.values
		return X_train, Y_train

	'''
	train_X_left = []
	train_X_right = []
	for i in X_train:
		b = i[0:][::2]
		train_X_left.append(b[:-1])#train_X_left.append(i[0:77])
		train_X_right.append(i[1:][::2])#train_X_right.append(i[77:154])

	l = []
	for t in train_X_left:
		l.append(list(t))

	r = []
	for t in train_X_right:
		r.append(list(t))

	X_1 = pd.DataFrame()
	X_1['left'] = l
	X_1['right'] = r

	X_train_1 = {'left': X_1.left, 'right': X_1.right}
	for i in range(0, len(X_train_1['left'])):
		print(i)
		for k in avoid_cols_indexes:
			if(X_train_1['left'][i][k] == X_train_1['right'][i][k]):
				X_train_1['left'][i][k] = 1
				X_train_1['right'][i][k] = 1
			else:
				X_train_1['left'][i][k] = 0
				X_train_1['right'][i][k] = 1


	train_X_left = []
	train_X_right = []
	for i in X_validation:
		b = i[0:][::2]
		train_X_left.append(b[:-1])#train_X_left.append(i[0:77])
		train_X_right.append(i[1:][::2])#train_X_right.append(i[77:154])

	l = []
	for t in train_X_left:
		l.append(list(t))

	r = []
	for t in train_X_right:
		r.append(list(t))

	X_1 = pd.DataFrame()
	X_1['left'] = l
	X_1['right'] = r

	X_validation_1 = {'left': X_1.left, 'right': X_1.right}
	for i in range(0, len(X_validation_1['left'])):
		for k in avoid_cols_indexes:
			if(X_validation_1['left'][i][k] == X_validation_1['right'][i][k]):
				X_validation_1['left'][i][k] = 1
				X_validation_1['right'][i][k] = 1
			else:
				X_validation_1['left'][i][k] = 0
				X_validation_1['right'][i][k] = 1


	max_length = len(X_train_1['left'][0])
	for dataset, side in itertools.product([X_train_1, X_validation_1], ['left', 'right']):
		print(dataset)
		dataset[side] = pad_sequences(dataset[side], maxlen=max_length)


	model = Sequential()
	model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
					 activation='relu',
					 input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(Conv2D(64, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(1000, activation='relu'))
	model.add(Dense(1, activation='softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer=keras.optimizers.SGD(lr=0.01),
				  metrics=['accuracy'])
	model.fit(X_train, Y_train,
			  batch_size=batch_size,
			  epochs=3,
			  verbose=1,
			  validation_data=(X_validation, Y_validation))
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	'''

def build_model(X_train, Y_train, X_validation, Y_validation):
	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(3458230+1, embedding_vecor_length, input_length=146))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	#Fitting the data into the model
	model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=2, batch_size=64)
	model.save('/data/ingestion/potCFD/model3_lstm_cnn_cat.h5')
	return model

def test_model(X_test, Y_test):
	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(3458230+1, embedding_vecor_length, input_length=146))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	#Model predictions
	model = load_model('/data/ingestion/potCFD/model3_lstm_cnn_cat.h5')
	prediction = model.predict(X_test)
	return prediction


def main():
    options = parse_options()
    if(options.env == "Prod"):
        key = "csap_prod_database"
    
    else:
        key = "csap_stage_database"

	db = get_db(settings, key)

	if(options.train == True):
		collection = db['BugDupsTrainSet_all_3']
		df = load_data(db, key, collection)
		X_train, Y_train, X_validation, Y_validation = build_data(df, True)
		cat_model = build_model(X_train, Y_train, X_validation, Y_validation)
	else:
		collection = db['BugDupsTrainSet_all_639_1968_new']
		df = load_data(db, key, collection)
		X_test, Y_test = build_data(df, False)
		cat_predictions = test_model(X_test, Y_test)


if __name__ == "__main__":
    main()







'''

n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 3

print("Building the model")

embedding_dim = 1 #300
embeddings = 1 * np.random.randn(3458230 + 1, embedding_dim)  # This will be the embedding matrix
embeddings[0] = 0  # So that the padding will be ignored

# Build the embedding matrix
for i in range(0, 3458230):
	embeddings[i] = [i]


def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

# The visible layer
left_input = Input(shape=(max_length, ), dtype='int32')
right_input = Input(shape=(max_length, ), dtype='int32')

embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_length, trainable=False)

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
malstm3 = Model([left_input, right_input], [malstm_distance])

# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm3.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

#malstm3.load_weights("/users/sumreddi/model3_cat.h5")
#preds = malstm2.predict([X_test_1['left'], X_test_1['right']])

malstm_trained = malstm3.fit([X_train_1['left'], X_train_1['right']], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
                            validation_data=([X_validation_1['left'], X_validation_1['right']], Y_validation))

# serialize weights to HDF5
malstm3.save_weights("/users/sumreddi/model3_cat.h5")
print("Saved model to disk")





malstm3.load_weights("/users/sumreddi/model3_cat.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
predictions = malstm3.predict([X_train_1['left'], X_train_1['right']])


malstm2.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
predictions = malstm.predict([X_train_1['left'], X_train_1['right']])
preds = malstm.predict([X_train_1['left'][0:4], X_train_1['right'][0:4]])

'''