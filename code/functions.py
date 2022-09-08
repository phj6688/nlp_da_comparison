
import pandas as pd
from textattack.augmentation import *
from aeda import Aeda_Augmenter
import os



def dataset_path(names,target):
    """
    Returns the path to a dataset.
    target should be either 'train' or 'test'
    """
    dict_dataset_paths = {'cardio':f'../data/{target}/cardio/{target}.txt', 'cr':f'../data/{target}/cr/{target}.txt',
                        'kaggle_med':f'../data/{target}/kaggle_med/{target}.txt',
                        'pc':f'../data/{target}/pc/{target}.txt','sst2':f'../data/{target}/sst2/{target}.txt',
                        'subj':f'../data/{target}/subj/{target}.txt','trec':f'../data/{target}/trec/{target}.txt'}
    #list_dataset_paths = [dict_dataset_paths[value] for key,value in dict_dataset_paths.items() if key in name]
    list_dataset_paths = [dict_dataset_paths[i] for i in names]
    return list_dataset_paths


def load_data(path):
    """
    Loads data from a txt file.
    """
    # check file format
    if path.endswith('.txt'):
        df = pd.read_csv(path, sep='|', header=None, names=['text'])
        try:
            df['class'] = df['text'].apply(lambda x: x.split('\t')[0])
            df['text'] = df['text'].apply(lambda x: x.split('\t')[1])
        except:
            df['class'] = df['text'].apply(lambda x: x.split(' ',1)[0])
            df['text'] = df['text'].apply(lambda x: x.split(' ',1)[1])

        df = df[['class', 'text']]
        return df
    else:
        raise ValueError('File format not supported.')



def augment_text(df,aug_method,fraction,pct_words_to_swap,transformations_per_example,
                label_column='class',target_column='text',include_original=True):

    augmenter_dict = { 
    'eda_augmenter':EasyDataAugmenter(pct_words_to_swap=pct_words_to_swap,
                                    transformations_per_example=transformations_per_example)
                                    ,
    'wordnet_augmenter':WordNetAugmenter(pct_words_to_swap=pct_words_to_swap,
                                    transformations_per_example=transformations_per_example)
                                    ,
    'clare_augmenter' :CLAREAugmenter(pct_words_to_swap=pct_words_to_swap,
                                    transformations_per_example=transformations_per_example)
                                    ,
    'backtranslation_augmenter':BackTranslationAugmenter(pct_words_to_swap=pct_words_to_swap,
                                    transformations_per_example=transformations_per_example)
                                    ,
    # 'checklist_augmenter' :CheckListAugmenter(pct_words_to_swap=pct_words_to_swap,
    #                                     transformations_per_example=transformations_per_example)
    #                                     ,
    # 'embedding_augmenter':EmbeddingAugmenter(pct_words_to_swap=pct_words_to_swap,
    #                                 transformations_per_example=transformations_per_example)
    #                                 ,
    # 'deletion_augmenter':DeletionAugmenter(pct_words_to_swap=pct_words_to_swap,
    #                                 transformations_per_example=transformations_per_example)
    'aeda_augmenter':Aeda_Augmenter(pct_words_to_swap=pct_words_to_swap,
                            transformations_per_example=transformations_per_example),
    'charswap_augmenter':CharSwapAugmenter(pct_words_to_swap=pct_words_to_swap,
                                    transformations_per_example=transformations_per_example)
    }

    augmenter = augmenter_dict[aug_method]
    os.system('clear')
    df = df.sample(frac=fraction)
    text_list , class_list = [], []
    for c, txt in zip(df[label_column], df[target_column]):

        res = augmenter.augment(txt)
        if include_original:
            text_list.append(txt)
            class_list.append(c)
            for i in res:
                text_list.append(i)
                class_list.append(c)
        else:
            for i in range(len(res)):
                text_list.append(res[i])
                class_list.append(c)

    df_augmented = pd.DataFrame({target_column: text_list, label_column: class_list})

    return df_augmented