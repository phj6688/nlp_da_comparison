# =============================================================================
# Import libraries
# =============================================================================
from aug import *
from functions import *

from time import sleep
import numpy as np
import pickle

from sklearn.metrics import accuracy_score 
from sklearn.utils import shuffle

from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.models import Sequential
from keras.callbacks import EarlyStopping

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #get rid of warnings


# =============================================================================
# set the random seed
# =============================================================================
np.random.seed(1337)


# =============================================================================
# dictionarys
# =============================================================================
dict_of_train_datasets = {'pc':'data/pc/train.txt','cr':'data/cr/train.txt','subj':'data/subj/train.txt'}
dict_of_test_datasets = {'pc':'data/pc/test.txt','cr':'data/cr/test.txt','subj':'data/subj/test.txt'}

dict_of_word2vec_files = {'pc':pickle.load(open('data/pc/word2vec.p', 'rb')),
                        'cr':pickle.load(open('data/cr/word2vec.p', 'rb')),
                        'subj':pickle.load(open('data/subj/word2vec.p', 'rb'))}


dict_of_models = {'pc':'models/pc_model.h5','cr':'models/cr_model.h5','subj':'models/subj_model.h5'}

# ===============================================================================
# constants
# ===============================================================================
word2vec_len = 300
input_size = 25
num_classes = 2


# =============================================================================
# laod data
# =============================================================================
df_train_pc = load_data(dict_of_train_datasets['pc'])
df_train_cr = load_data(dict_of_train_datasets['cr'])
df_train_subj = load_data(dict_of_train_datasets['subj'])

df_test_pc = load_data(dict_of_test_datasets['pc'])
df_test_cr = load_data(dict_of_test_datasets['cr'])
df_test_subj = load_data(dict_of_test_datasets['subj'])

# =============================================================================
# Functions
# =============================================================================
def build_model(sentence_length, word2vec_len, num_classes):
    model = None
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(sentence_length, word2vec_len)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32, return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    return model



def get_x_y(train_txt, word2vec_len, input_size, word2vec, percent_dataset):

    #read in lines
    train_lines = open(train_txt, 'r').readlines()
    shuffle(train_lines)
    train_lines = train_lines[:int(percent_dataset*len(train_lines))]
    num_lines = len(train_lines)

    #initialize x and y matrix
    x_matrix = None
    y_matrix = None

    try:
        x_matrix = np.zeros((num_lines, input_size, word2vec_len))
    except:
        print("Error!", num_lines, input_size, word2vec_len)
    y_matrix = np.zeros((num_lines))

    #insert values
    for i, line in enumerate(train_lines):

        parts = line[:-1].split('\t')
        label = int(parts[0])
        sentence = parts[1]	

        #insert x
        words = sentence.split(' ')
        words = words[:x_matrix.shape[1]] #cut off if too long
        for j, word in enumerate(words):
            if word in word2vec:
                x_matrix[i, j, :] = word2vec[word]

        #insert y
        y_matrix[i] = label

    return x_matrix, y_matrix

#one hot to categorical
def one_hot_to_categorical(y):
    assert len(y.shape) == 2
    return np.argmax(y, axis=1)



def run_model(dataset_name):
	train_x, train_y = get_x_y(dict_of_train_datasets[dataset_name], num_classes, word2vec_len, input_size, dict_of_word2vec_files[dataset_name], 1)
	test_x, test_y = get_x_y(dict_of_test_datasets[dataset_name], num_classes, word2vec_len, input_size, dict_of_word2vec_files[dataset_name], 1)

	#build model
	model = build_model(input_size, word2vec_len, num_classes)

	callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

	#train model
	model.fit(	train_x, 
					train_y, 
					epochs=100000, 
					callbacks=callbacks,
					validation_split=0.1, 
					batch_size=1024, 
					shuffle=True, 
					verbose=0)
	#save the model
	model.save(dict_of_models[dataset_name])


	#evaluate model
	y_pred = model.predict(test_x)
	test_y_cat = one_hot_to_categorical(test_y)
	y_pred_cat = one_hot_to_categorical(y_pred)
	acc = accuracy_score(test_y_cat, y_pred_cat)

	#clean memory???
	train_x, train_y = None, None

	#return the accuracy
	#print("data with shape:", train_x.shape, train_y.shape, 'train=', train_file, 'test=', test_file, 'with fraction', percent_dataset, 'had acc', acc)
	return acc



if __name__ == '__main__':
    list_of_datasets = ['pc','cr','subj']
    accuracy_scores = []
    print("Running models...")
    for dataset_name in list_of_datasets:
        acc = run_model(dataset_name)
        accuracy_scores.append(acc)

    print('running models done')
    sleep(2)
    os.system('clear')
    for i,j in zip(list_of_datasets, accuracy_scores):
            
        print(f'the accuracy of {i} dataset with {input_size} input size and {word2vec_len} featurs is: {j:.4f}')
        print('=============================================')
        print('=============================================')