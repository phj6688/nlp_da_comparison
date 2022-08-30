from textattack.augmentation import \
    EasyDataAugmenter, BackTranslationAugmenter, WordNetAugmenter, CLAREAugmenter, \
    CheckListAugmenter, EmbeddingAugmenter, DeletionAugmenter, CharSwapAugmenter
from functions import *
import os
import warnings
warnings.filterwarnings("ignore")


pct_words_to_swap = float(input("Enter percentage of words to swap (from 0.01 to 0.9): "))
assert 0.01<pct_words_to_swap<0.9,'enter number between 0.01 and 0.9'

transformations_per_example = int(input("Enter number of transformations per example(from 1 to 10): "))
assert 1<transformations_per_example<10,'enter number between 1 and 10'

eda_augmenter = EasyDataAugmenter(pct_words_to_swap=pct_words_to_swap,transformations_per_example=transformations_per_example)
wordnet_augmenter = WordNetAugmenter(pct_words_to_swap=pct_words_to_swap,transformations_per_example=transformations_per_example)
clare_augmenter = CLAREAugmenter(pct_words_to_swap=pct_words_to_swap,transformations_per_example=transformations_per_example)
backtranslation_augmenter = BackTranslationAugmenter(pct_words_to_swap=pct_words_to_swap,transformations_per_example=transformations_per_example)
checklist_augmenter = CheckListAugmenter(pct_words_to_swap=pct_words_to_swap,transformations_per_example=transformations_per_example)
embedding_augmenter = EmbeddingAugmenter(pct_words_to_swap=pct_words_to_swap,transformations_per_example=transformations_per_example)
deletion_augmenter = DeletionAugmenter(pct_words_to_swap=pct_words_to_swap,transformations_per_example=transformations_per_example)
charswap_augmenter = CharSwapAugmenter(pct_words_to_swap=pct_words_to_swap,transformations_per_example=transformations_per_example)

os.system('clear')
#list_of_augmenters = [eda_augmenter, wordnet_augmenter, clare_augmenter, backtranslation_augmenter, checklist_augmenter, embedding_augmenter, deletion_augmenter, charswap_augmenter]
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

# pass the data to the augmenters
for aug in list_of_augs:
    





# sentence = 'What I cannot create, I do not understand.'

# for augmenter in list_of_augmenters:
#     print(augmenter.__class__.__name__)
#     res = augmenter.augment(sentence)
#     print(res)