from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
import numpy as np
from tqdm import tqdm
from functions import *
import pickle


def gen_word2vec_embeddings(dataset, word2vec_output, glove_file):
    """
    Generates word embeddings for a set of train test dataset.
    
    """

    embedding_vector = {}
    f = open(glove_file)
    for line in tqdm(f):
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:],dtype = 'float32')
        embedding_vector[word] = coef
    f.close()


    def try_stem(word):
        ps = PorterStemmer()
        ls = LancasterStemmer()
        wn = WordNetLemmatizer()
        if ps.stem(word) in embedding_vector:
            return ps.stem(word)
        elif ls.stem(word) in embedding_vector:
            return ls.stem(word)
        elif wn.lemmatize(word) in embedding_vector:
            return wn.lemmatize(word)
        else:
            return np.zeros(300)

    test_file_path = dataset_path(dataset, 'test')
    train_file_path = dataset_path(dataset, 'train')
    df_test = load_data(test_file_path[0]) #take the first item, because it is a list
    df_train = load_data(train_file_path[0])
    raw_text = list(df_train['text'].apply(lambda x: x.lower())) + list(df_test['text'].apply(lambda x: x.lower()))
    # bulid the vocab
    vocab = set()
    word2vec = {}

    for text in tqdm(raw_text):
        for word in word_tokenize(text):
            vocab.add(word)
            if word not in word2vec:
                try:
                    word2vec[word] = np.asarray(embedding_vector[word], dtype = 'float32')
                except:
                    word2vec[word] = try_stem(word)
            else:
                pass
    
    # save the vocab    
    pickle.dump(word2vec, open(word2vec_output, 'wb'))


if __name__ == "__main__":
    glove_file = 'glove.840B.300d.txt'
    # dataset should always be as list
    dataset = ['cr']


    #for test!
    output_dir = f'{os.getcwd()}/{dataset[0]}'
    if not os.path.exists(output_dir):
                os.makedirs(output_dir)
    word2vec_output = f'{output_dir}/word2vec.p'
    gen_word2vec_embeddings(dataset, word2vec_output, glove_file)