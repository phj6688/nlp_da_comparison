from functions import *
import numpy as np
import pandas as pd
import pickle
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
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
    # print("Loss: ", loss)
    # print("Accuracy: ", accuracy)
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("AUC: ", auc)
    # print ('f1 score: ', f1_score)
    return loss, accuracy, precision, recall, auc, f1_score


def run(dataset_name,aug_method,n_sample):
    # Load data
    print('Loading data...')
    path_train_original = f'data/{dataset_name}/train.txt'
    path_train_aug = f'data/{dataset_name}/train_{dataset_name}_{aug_method}_n_sample_{n_sample}.csv'
    path_test = f'data/{dataset_name}/test.txt'
    train_aug = load_data(path_train_aug)
    train_original = load_data(path_train_original)
    test_data = load_data(path_test)
    X_train_aug, y_train_aug = train_aug['text'].values, train_aug['class'].values.astype(float)
    X_train_original, y_train_original = train_original['text'].values, train_original['class'].values.astype(float)
    X_test, y_test = test_data['text'].values, test_data['class'].values.astype(float)

    # y_train_aug = to_categorical(y_train_aug)
    # y_train_original = to_categorical(y_train_original)
    # y_test = to_categorical(y_test)


    # load wor2vec pickle
    path_w2v = f'data/{dataset_name}/word2vec.p'
    w2v = pickle.load(open(path_w2v, 'rb'))


    # create matrices
    print('Creating matrices...')
    X_train_aug = create_X_matrix(X_train_aug, w2v)
    X_train_original = create_X_matrix(X_train_original, w2v)
    X_test = create_X_matrix(X_test, w2v)

    # Train model
    print('Training model...')
    model_aug = build_model()
    model_original = build_model()

    callbacks = [EarlyStopping(monitor='val_loss', patience=5)]

    model_aug.fit(X_train_aug, y_train_aug, epochs=1000,
                 callbacks=callbacks, validation_split=0.1,
                  batch_size=128,shuffle=True, verbose=0)
    model_original.fit(X_train_original, y_train_original, epochs=1000,
                    callbacks=callbacks, validation_split=0.1,
                    batch_size=128,shuffle=True, verbose=0)

    # Evaluate model
    print('Evaluating model...')

    loss_org, accuracy_org, precision_org, recall_org, auc_org, f1_score_org = evaluate_model(model_original, X_test, y_test)
    loss_aug, accuracy_aug, precision_aug, recall_aug, auc_aug, f1_score_aug = evaluate_model(model_aug, X_test, y_test)
    print(f'original model: \n acc: {accuracy_org:.4f} \n f1: {f1_score_org:.4f} \n precision: {precision_org:.4f} \n recall: {recall_org:.4f}, \n auc: {auc_org:.4f}')
    print(f'augmented model: \n acc: {accuracy_aug:.4f} \n f1: {f1_score_aug:.4f} \n precision: {precision_aug:.4f} \n recall: {recall_aug:.4f}, \n auc: {auc_aug:.4f}')

    print('finished')

if __name__ == '__main__':

    


    run('pc','wordnet_augmenter',2)