import pickle
import os
import numpy as np







huge_word2vec = 'glove.840B.300d.txt'
word2vec_len = 300 # don't want to load the huge pickle every time, so just save the words that are actually used into a smaller dictionary

#loading a pickle file
def load_pickle(file):
    return pickle.load(open(file, 'rb'))

#create an output folder if it does not already exist
def confirm_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)



#get the pickle file for the word2vec so you don't have to load the entire huge file each time
def gen_vocab_dicts(folder, output_pickle_path, huge_word2vec):

    vocab = set()
    text_embeddings = open(huge_word2vec, 'r').readlines()
    word2vec = {}

    #get all the vocab
    all_txt_paths = get_all_txt_paths(folder)
    print(all_txt_paths)

    #loop through each text file
    for txt_path in all_txt_paths:

        # get all the words
        try:
            all_lines = open(txt_path, "r").readlines()
            for line in all_lines:
                words = line[:-1].split(' ')
                for word in words:
                    vocab.add(word)
        except:
            print(txt_path, "has an error")
    
    print(len(vocab), "unique words found")

    # load the word embeddings, and only add the word to the dictionary if we need it
    for line in text_embeddings:
        items = line.split(' ')
        word = items[0]
        if word in vocab:
            vec = items[1:]
            word2vec[word] = np.asarray(vec, dtype = 'float32')
    print(len(word2vec), "matches between unique words and word2vec dictionary")
        
    pickle.dump(word2vec, open(output_pickle_path, 'wb'))
    print("dictionaries outputted to", output_pickle_path)
