
import pandas as pd



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
        df['class'] = df['text'].apply(lambda x: x.split('\t')[0])
        df['text'] = df['text'].apply(lambda x: x.split('\t')[1])
        df = df[['class', 'text']]
        return df
    else:
        raise ValueError('File format not supported.')



def augment_text(df,aug_method,fraction=0.5,label_column='class',target_column='text',pct_words_to_swap=0.2,transformations_per_example=4,include_original=True):
    augmenter = augmenter_dict[aug_method.lower()+'_augmenter']
    augmenter.pct_words_to_swap = pct_words_to_swap
    augmenter.transformations_per_example = transformations_per_example
    # print('Augmenting with',str(augmenter))
    # print('percentage of words to swap:',augmenter.pct_words_to_swap)
    # print('number of transformations per example:',augmenter.transformations_per_example)
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