# from tqdm import tqdm
# import numpy as np
# embedding_vector = {}
# f = open('eda_code/glove.840B.300d.txt')
# for line in tqdm(f):
#     value = line.split(' ')
#     word = value[0]
#     coef = np.array(value[1:],dtype = 'float32')
#     embedding_vector[word] = coef
# f.close()
# embedding_vector_list = list(embedding_vector.items())
# print(embedding_vector_list[5000][0])
# print(embedding_vector_list[5000][1])
# print(len(embedding_vector_list[5000][1]))
# print('len of list ==> ' + str(len(embedding_vector_list)))
# print('len of dict ==> ' + str(len(embedding_vector)))
# print('shape of list ==> ' + str(np.asarray(embedding_vector_list[5000],dtype=object).shape))
# print('shape of dict ==> ' + str(np.asarray(embedding_vector,dtype=object).shape))

# from sklearn.manifold import TSNE
# import plotly.express as px


# df = px.data.iris()

# features = df.loc[:, :'petal_width']

# tsne = TSNE(n_components=2, random_state=0)
# projections = tsne.fit_transform(features)

# fig = px.scatter(
#     projections, x=0, y=1,
#     color=df.species, labels={'color': 'species'}
# )
# print('len of projection: ',len(projections))
# print('shape of projection: ',projections.shape)
# print(projections)

import pickle
from functions import *
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.models import Sequential
from sklearn.utils import shuffle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #get rid of warnings


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



word2vec_len = 300
input_size = 25
num_classes = 2


dict_of_train_datasets = {'pc':'present/data/pc/train.txt','cr':'present/data/cr/train.txt','subj':'present/data/subj/train.txt'}
dict_of_test_datasets = {'pc':'present/data/pc/test.txt','cr':'present/data/cr/test.txt','subj':'present/data/subj/test.txt'}

dict_of_word2vec_files = {'pc':pickle.load(open('present/data/pc/word2vec.p', 'rb')),
                        'cr':pickle.load(open('present/data/cr/word2vec.p', 'rb')),
                        'subj':pickle.load(open('present/data/subj/word2vec.p', 'rb'))}
# laod data
df_train_pc = load_data('present/data/pc/train.txt')
df_train_cr = load_data('present/data/cr/train.txt')
df_train_subj = load_data('present/data/subj/train.txt')
df_test_pc = load_data('present/data/pc/test.txt')
df_test_cr = load_data('present/data/cr/test.txt')
df_test_subj = load_data('present/data/subj/test.txt')


matric = np.zeros((input_size, word2vec_len))
df_train_pc['XX'] = ''
for i in range(len(df_train_pc)):
    df_train_pc['XX'][i] = matric.tolist()


print(df_train_pc).head(20)
print('==================')
print(df_train_pc).tail(20)