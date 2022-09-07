from xml.etree.ElementInclude import include
from textattack.augmentation import \
    EasyDataAugmenter, BackTranslationAugmenter, WordNetAugmenter, CLAREAugmenter, \
    CheckListAugmenter, EmbeddingAugmenter, DeletionAugmenter, CharSwapAugmenter
from functions import *
import os
import sys
import warnings
warnings.filterwarnings("ignore")



pct_words_to_swap = float(input("Enter percentage of words to swap (from 0.01 to 0.9): "))
assert 0.01<pct_words_to_swap<0.9,'enter number between 0.01 and 0.9'

transformations_per_example = int(input("Enter number of transformations per example(from 1 to 10): "))
assert 1<transformations_per_example<10,'enter number between 1 and 10'




dict_aug_methods = {1:'eda_augmenter', 2:'wordnet_augmenter', 3:'clare_augmenter', 4:'backtranslation_augmenter', 5:'checklist_augmenter', 6:'embedding_augmenter', 7:'deletion_augmenter', 8:'charswap_augmenter'}

for key, value in dict_aug_methods.items():
    print(value,' -->',key)
list_of_augs = [int(i) for i in input("Enter augmentation method numbers to use (separated by commas): \n ").split(',')]

# load dataset
dict_dataset_names = {'1':'cardio', '2':'cr', '3':'kaggle_med', '4':'pc', '5':'sst2', '6':'subj', '7':'trec'}
dict_target_names = {'1':'train', '2':'test'}
def load_all_data():
    for key, value in dict_dataset_names.items():
        print(value,' -->',key)
    list_dataset_numbers = [i for i in input("Enter dataset numbers to use (separated by commas): \n ").split(',')]
    list_of_dataset_names = [dict_dataset_names[i] for i in list_dataset_numbers]
    input_target = input("Enter target: 1 for train, 2 for test: ")
    target =dict_target_names[input_target]
    paths = dataset_path(list_of_dataset_names,target)
    df_list = [load_data(i) for i in paths]
    return df_list


# run it in the this folder in order to get access to relative pass!
df_list = load_all_data()
fraction = float(input('Enter fraction of data to use (from 0.1 to 0.9): '))
include_original = input('Include original data? (y/n): ').lower()
yes = ['yes','y', 'ye', '']
no = ['no','n']

if include_original in yes:
   include_original = True
elif include_original in no:
   include_original = False
else:
   print("Please respond with 'yes' or 'no'")
# pass the data to the augmenters

for aug in list_of_augs:
    for df in df_list:
        new_df = augment_text(df,dict_aug_methods[aug],fraction,pct_words_to_swap,transformations_per_example,include_original=include_original)
        new_df.to_csv(f'{dict_aug_methods[aug]}_{fraction}_{transformations_per_example}_{pct_words_to_swap}_{include_original}.csv',index=False)
        print(f'{dict_aug_methods[aug]}_{fraction}_{transformations_per_example}_{pct_words_to_swap}_{include_original}.csv was created successfully')


