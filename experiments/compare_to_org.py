# Description: This script compares the performance of the original dataset to the augmented dataset
# here I used batch size of 32 
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
	path = f'data/{dataset_name}/'
	path_test = f'{path}test.txt'
	path_train_original = f'{path}train1000_sample.txt'
	test_data = load_data(path_test)
	train_original = load_data(path_train_original)
	X_test, y_test = test_data['text'].values, test_data['class'].values.astype(float)
	X_train_original, y_train_original = train_original['text'].values, train_original['class'].values.astype(float)
		# load word2vec pickle
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

def run(dataset_name,aug_method,n_sample,org):
		# Load data
		print('Loading data...')
		path = f'data/{dataset_name}/'
	 
		path_train_aug = f'{path}train_{aug_method}_{n_sample}_includeOrg_{org}.csv'
		path_test = f'{path}test.txt'
		train_aug = load_data(path_train_aug)

		test_data = load_data(path_test)
		X_train_aug, y_train_aug = train_aug['text'].values, train_aug['class'].values.astype(float)

		X_test, y_test = test_data['text'].values, test_data['class'].values.astype(float)

		# load word2vec pickle
		path_w2v = f'{path}word2vec_1000_sample.p'
		w2v = pickle.load(open(path_w2v, 'rb'))

		# load large glove
		# df = pd.read_csv('glove.840B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
		# w2v = {key: val.values for key, val in df.T.items()}

		# create matrices
		print('Creating matrices...')
		X_train_aug = create_X_matrix(X_train_aug, w2v)

		X_test = create_X_matrix(X_test, w2v)


		
		# Train model
		loss_aug, accuracy_aug, precision_aug, recall_aug, auc_aug, f1_score_aug	= [],[],[],[],[],[]
		for i in range(3):
			print('Training model...')
			model_aug = build_model()
			callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
			model_aug.fit(X_train_aug, y_train_aug, epochs=10000,
									callbacks=callbacks, validation_split=0.1,
										batch_size=32,shuffle=True, verbose=0)
			# Evaluate model
			print('Evaluating model...')
			
			loss_aug1, accuracy_aug1, precision_aug1, recall_aug1, auc_aug1, f1_score_aug1 = evaluate_model(model_aug, X_test, y_test)
			loss_aug.append(loss_aug1)
			accuracy_aug.append(accuracy_aug1)
			precision_aug.append(precision_aug1)
			recall_aug.append(recall_aug1)
			auc_aug.append(auc_aug1)
			f1_score_aug.append(f1_score_aug1)
		loss_aug = np.mean(loss_aug)
		accuracy_aug = np.mean(accuracy_aug)
		precision_aug = np.mean(precision_aug)
		recall_aug = np.mean(recall_aug)
		auc_aug = np.mean(auc_aug)
		f1_score_aug = np.mean(f1_score_aug)



		print(f'augmented model: \n acc: {accuracy_aug:.4f} \n f1: {f1_score_aug:.4f} \n precision: {precision_aug:.4f} \n recall: {recall_aug:.4f}, \n auc: {auc_aug:.4f}')

		print('finished')

		
		return loss_aug, accuracy_aug, precision_aug, recall_aug, auc_aug, f1_score_aug

#run the original model


def run_augmented_models(dataset,aug_method):

		loss_org, accuracy_org, precision_org, recall_org, auc_org, f1_score_org = run_original(dataset)

		n_samples = [1,2,4,8]

		loss_aug = {}
		accuracy_aug = {}
		precision_aug = {}
		recall_aug = {}
		auc_aug = {}
		f1_score_aug = {}
		#prepare the augmented models
		for n in n_samples:			
			loss, accuracy, precision, recall, auc, f1_score = run(dataset,aug_method,n,org=True)
			loss_aug.update({f'n_sample_{n}':loss})
			accuracy_aug.update({f'n_sample_{n}':accuracy})
			precision_aug.update({f'n_sample_{n}':precision})
			recall_aug.update({f'n_sample_{n}':recall})
			auc_aug.update({f'n_sample_{n}':auc})
			f1_score_aug.update({f'n_sample_{n}':f1_score}) 
			print(f'\n\nfinished {n} samples for {aug_method} on {dataset} dataset with org True \n')


			list_of_dicts = [loss_aug,accuracy_aug,precision_aug,recall_aug,auc_aug,f1_score_aug]
			df_aug = pd.DataFrame(list_of_dicts,index=['loss','accuracy','precision','recall','auc','f1_score'])
			df_org = pd.DataFrame([loss_org, accuracy_org, precision_org, recall_org, auc_org, f1_score_org],index=['loss','accuracy','precision','recall','auc','f1_score'],columns=['original'])
			df_compare = df_aug.join(df_org)
			df_compare.to_csv(f'results/1000sample/compare_1000_{dataset}_{aug_method}_org_True.csv')




if __name__ == '__main__':
		dataset = 'pc'
		aug_methods =	['eda_augmenter','wordnet_augmenter','aeda_augmenter','backtranslation_augmenter','clare_augmenter']
		for aug_method in aug_methods:
				run_augmented_models(dataset,aug_method)
				print(f'\n\n\nfinished {aug_method}\n\n\n')
