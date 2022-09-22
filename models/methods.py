import pickle
import os







huge_word2vec = 'glove.840B.300d.txt'
word2vec_len = 300 # don't want to load the huge pickle every time, so just save the words that are actually used into a smaller dictionary

#loading a pickle file
def load_pickle(file):
    return pickle.load(open(file, 'rb'))

#create an output folder if it does not already exist
def confirm_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)