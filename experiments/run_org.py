from functions import *
import numpy as np
import pandas as pd
import pickle
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from keras import backend as K
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.models import Sequential, load_model, Model
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.metrics import Precision, Recall, AUC
from keras.optimizers import RMSprop

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #get rid of warnings

def create_y_matrix(y_data):
	y_matrix = np.zeros((len(y_data),2))
	for count,i in enumerate(y_data):
		if i == 1:
			y_matrix[count][1] = 1.0
		else:
			y_matrix[count][0] = 1.0
	return y_matrix

def create_X_matrix(dataset, w2v,word2vec_len=300, batch_size=25):
	dataset_size = len(dataset)
	x_matrix = np.zeros((dataset_size, batch_size, word2vec_len))    
	for i, line in enumerate(dataset):
		try:
			words = line.split()
			words = words[:batch_size] #cut off if too long
			for j, word in enumerate(words):
				if word in w2v:
					x_matrix[i, j, :] = w2v[word]
		except:
			pass
	return x_matrix


def build_model(batch_size=25, word2vec_len=300):
	model = None
	model = Sequential()
	model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(batch_size, word2vec_len)))
	model.add(Dropout(0.5))
	model.add(Bidirectional(LSTM(32, return_sequences=False)))
	model.add(Dropout(0.5))
	model.add(Dense(20, activation='relu'))

	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))              
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',Precision(),Recall(),AUC()] )
	return model

def evaluate_model(model, X_test, y_test):
	eval = model.evaluate(X_test, y_test)
	loss = eval[0]
	accuracy = eval[1]
	precision = eval[2]
	recall = eval[3]
	auc = eval[4]    
	f1_score = (2*precision*recall)/(precision+recall)

	return loss, accuracy, precision, recall, auc, f1_score

def run_original(dataset_name):
	# set the path folder and variations; for example here I load the original dataset with 1000 samples
	# and the word2vec pickle

	path = f'data/{dataset_name}/'
	path_test = f'{path}test.txt'	
	path_train_original = f'{path}train1000_sample.txt'
	test_data = load_data(path_test)
	train_original = load_data(path_train_original)
	X_test, y_test = test_data['text'].values, test_data['class'].values.astype(float)
	X_train_original, y_train_original = train_original['text'].values, train_original['class'].values.astype(float)
	# load wor2vec pickle
	path_w2v = f'{path}word2vec_1000_sample.p'
	w2v = pickle.load(open(path_w2v, 'rb'))
	X_train_original = create_X_matrix(X_train_original, w2v)
	X_test = create_X_matrix(X_test, w2v)
	model_original = build_model()
	callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
	model_original.fit(X_train_original, y_train_original, epochs=10000,
						callbacks=callbacks, validation_split=0.1,
						batch_size=32,shuffle=True, verbose=0)

	loss_org, accuracy_org, precision_org, recall_org, auc_org, f1_score_org = evaluate_model(model_original, X_test, y_test)
	return loss_org, accuracy_org, precision_org, recall_org, auc_org, f1_score_org

if __name__ == '__main__':
	dataset_name = 'pc'
	# run the model 3 times and take the average
	print('Running original model 3 times...')
	loss_org1, accuracy_org1, precision_org1, recall_org1, auc_org1, f1_score_org1 = run_original(dataset_name)
	print('first time done!')
	loss_org2, accuracy_org2, precision_org2, recall_org2, auc_org2, f1_score_org2 = run_original(dataset_name)
	print('second time done!')
	loss_org3, accuracy_org3, precision_org3, recall_org3, auc_org3, f1_score_org3 = run_original(dataset_name)
	print('third time done!')

	loss_org = (loss_org1 + loss_org2 + loss_org3)/3
	accuracy_org = (accuracy_org1 + accuracy_org2 + accuracy_org3)/3
	precision_org = (precision_org1 + precision_org2 + precision_org3)/3
	recall_org = (recall_org1 + recall_org2 + recall_org3)/3
	auc_org = (auc_org1 + auc_org2 + auc_org3)/3
	f1_score_org = (f1_score_org1 + f1_score_org2 + f1_score_org3)/3    

	print(f'\n\n Original Model Results base on 3 runs for {dataset_name } dataset with 1000 samples:  \n\n')
	print(f'loss_org: {loss_org:.4f}')
	print(f'accuracy_org: {accuracy_org:.4f}')
	print(f'precision_org: {precision_org:.4f}')
	print(f'recall_org: {recall_org:.4f}')
	print(f'auc_org: {auc_org:.4f}')
	print(f'f1_score_org: {f1_score_org:.4f}')