# =============================================================================
# Import libraries
# =============================================================================
from aug import *
from functions import *
from tqdm import tqdm

import plotly
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from scipy.spatial import distance

from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from keras import backend as K
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.models import Sequential, load_model, Model
from keras.callbacks import EarlyStopping

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #get rid of warnings




# =============================================================================
# Variables
# =============================================================================
# dictionarys
# =============================================================================
dict_of_train_datasets = {'pc':'data/pc/train.txt','cr':'data/cr/train.txt','subj':'data/subj/train.txt'}
dict_of_test_datasets = {'pc':'data/pc/test.txt','cr':'data/cr/test.txt','subj':'data/subj/test.txt'}

dict_of_30_samples = {'pc':'data/pc/30_samples.txt','cr':'data/cr/30_samples.txt','subj':'data/subj/30_samples.txt'}

dict_of_models = {'pc':'models/pc_model.h5','cr':'models/cr_model.h5','subj':'models/subj_model.h5'}

dict_of_word2vec_files = {'pc':pickle.load(open('data/pc/word2vec.p', 'rb')),
                        'cr':pickle.load(open('data/cr/word2vec.p', 'rb')),
                        'subj':pickle.load(open('data/subj/word2vec.p', 'rb'))}


dict_of_eda_augmented_datasets = {'pc':'data/pc/eda_augmenter_augmented.txt', 'cr':'data/cr/eda_augmenter_augmented.txt', 'subj':'data/subj/eda_augmenter_augmented.txt'}
dict_of_wordnet_augmented_datasets = {'pc':'data/pc/wordnet_augmenter_augmented.txt', 'cr':'data/cr/wordnet_augmenter_augmented.txt', 'subj':'data/subj/wordnet_augmenter_augmented.txt'}
dict_of_aeda_augmented_datasets = {'pc':'data/pc/aeda_augmenter_augmented.txt', 'cr':'data/cr/aeda_augmenter_augmented.txt', 'subj':'data/subj/aeda_augmenter_augmented.txt'}
dict_of_backtranslation_augmented_datasets = {'pc':'data/pc/backtranslation_augmenter_augmented.txt', 'cr':'data/cr/backtranslation_augmenter_augmented.txt', 'subj':'data/subj/backtranslation_augmenter_augmented.txt'}

dict_of_aug_methods = {'eda':dict_of_eda_augmented_datasets, 'wordnet':dict_of_wordnet_augmented_datasets, 'aeda':dict_of_aeda_augmented_datasets, 'backtranslation':dict_of_backtranslation_augmented_datasets}

# # =============================================================================
# # laod data
# df_train_cr = load_data('data/cr/train.txt')
# df_train_pc = load_data('data/pc/train.txt')
# df_train_subj = load_data('data/subj/train.txt')
# df_test_pc = load_data('data/pc/test.txt')
# df_test_cr = load_data('data/cr/test.txt')
# df_test_subj = load_data('data/subj/test.txt')

# ===============================================================================
# constants
# ===============================================================================
word2vec_len = 300
input_size = 25
num_classes = 2

# =============================================================================

# functions



def train_x(train_txt, word2vec_len, input_size, word2vec):

	#read in lines
	train_lines = open(train_txt, 'r').readlines()
	num_lines = len(train_lines)

	x_matrix = np.zeros((num_lines, input_size, word2vec_len))

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

	return x_matrix



def get_tsne_labels(file,num_aug):
	labels = []
	alphas = []
	lines = open(file, 'r').readlines()
	for i, line in enumerate(lines):
		parts = line[:-1].split('\t')
		_class = int(parts[0])
		alpha = i % (num_aug+1)
		if alpha == 0:
			labels.append(_class+100)
			alphas.append(alpha)
		else:
			labels.append(_class)
			alphas.append(alpha)
	return labels, alphas

def get_plot_vectors(layer_output,perplexity=30,n_iter=1000,random_state=0,method='barnes_hut',learning_rate=200):

	tsne = TSNE(n_components=2,perplexity=perplexity,n_iter=n_iter,random_state=random_state,method=method,learning_rate=learning_rate).fit_transform(layer_output)
	return tsne


def get_x_y(train_txt, num_classes, word2vec_len, input_size, word2vec, percent_dataset):

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
    y_matrix = np.zeros((num_lines, num_classes))

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
        y_matrix[i][label] = 1.0

    return x_matrix, y_matrix



#one hot to categorical
def one_hot_to_categorical(y):
    assert len(y.shape) == 2
    return np.argmax(y, axis=1)

def aug_samples(dataset_name):
    data = load_data(dict_of_30_samples['pc'])
    methods = ['eda_augmenter','wordnet_augmenter','aeda_augmenter','backtranslation_augmenter']
    for method in methods:
        augmented_data = augment_text(data, method,fraction=1,pct_words_to_swap=0.2 ,transformations_per_example=1,
                    label_column='class',target_column='text',include_original=True)
        augmented_data = augmented_data[['class','text']]
        np.savetxt(f'data/{dataset_name}/{method}_augmented.txt', augmented_data.values, fmt='%s', delimiter='\t')






def plotly_tsne(df, df_name, method):
    total_distance = df['normalized_distance'].sum()/2
    fig = px.scatter(df, x='standardized_x', y='standardized_y', color='color'
                                , size='size'
                                , symbol='symbol'
                                , title=f't-SNE plot of {df_name} dataset with {method} augmentation method, total distance: {total_distance:.4f}'
                                , custom_data=[df['text'],df['label'],df['distance']]
                                ).update_traces(hovertemplate='Text: %{customdata[0]} <br>' +
                                                                            'Label: %{customdata[1]} <br>' +
                                                                            'Distance: %{customdata[2]:.5f} <br>' )
    fig.update_layout(showlegend=False)
    fig.show()
    #fig.write_html(f'images/tsne_{df_name}_{method}.html')
    

def cal_distance(df,number_of_augmentation_per_sample):
    dists = []
    for i in range(0,len(df),number_of_augmentation_per_sample+1):
        point_a = df[['x','y']].iloc[i]
        point_b = df[['x','y']].iloc[i+1]
        dist = distance.euclidean(point_a, point_b)
        dists.append(dist)
    dists = np.repeat(dists, 2)
    return dists




def label_to_str_map(x):
    if x == 0:
        return 'Con (augmented)'
    elif x == 1:
        return 'Pro (augmented)'
    elif x == 100:
        return 'Con (original)'
    elif x == 101:
        return 'Pro (original)'

def label_to_color_map(x):
    if x == 100 or x == 0:
        return 'red'
    elif x == 1 or x == 101:
        return 'blue'

def label_to_size_map(x):
    if x >= 100:
        return 100
    elif x < 100:
        return 10

def label_to_symbol_map(x):
    if x == 0 or x == 100:
        return '^'
    elif x == 1 or x == 101:
        return 'o'


def run_tsne(dataset_name,method,layer_number):

    model = load_model(dict_of_models[dataset_name])
    augmented_file = dict_of_aug_methods[method][dataset_name]
    word2vec = dict_of_word2vec_files[dataset_name]
    labels, _ = get_tsne_labels(augmented_file,1)

    labels_str = list(map(label_to_str_map,labels))
    labels_color = list(map(label_to_color_map,labels))
    labels_size = list(map(label_to_size_map,labels))
    labels_symbol = list(map(label_to_symbol_map,labels))

    X = train_x(augmented_file, word2vec_len, input_size, word2vec)
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(model.layers[layer_number].name).output)
    layer_output = intermediate_layer_model.predict(X)
    t = get_plot_vectors(layer_output,perplexity=30,n_iter=5000,random_state=10,method='exact')

    df = pd.DataFrame(t, columns=['x','y'])
    df['standardized_x'] = df['x'].apply(lambda x: (x - df['x'].mean()) / df['x'].std())
    df['standardized_y'] = df['y'].apply(lambda x: (x - df['y'].mean()) / df['y'].std())

    df_text = load_data(augmented_file)


    df['ground_truth'] = [int(i) for i in df_text['class']]
    df['predicted_label'] = list(one_hot_to_categorical(model.predict(X)))

    df_text = df_text[['text']]

    df['distance'] = cal_distance(df,1)
    df['normalized_distance'] = df['distance'].apply(lambda x: (x - df['distance'].min()) / (df['distance'].max() - df['distance'].min()))

    df['text'] = df_text['text']
    df['label'] = labels_str
    df['color'] = labels_color
    df['size'] = labels_size
    df['symbol'] = labels_symbol

    plotly_tsne(df, dataset_name, method)


# =============================================================================
# Main
# =============================================================================


if __name__ == '__main__':
    # list_of_datasets = ['pc','cr','subj']
    # list_of_aug_methods = ['eda','wordnet','aeda','backtranslation']

    # for dataset_name in list_of_datasets:
    #     for method in list_of_aug_methods:
    #         run_tsne(dataset_name,method)
    run_tsne('cr','wordnet',4)
