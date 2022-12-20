from aeda import *
from functions import *
from tqdm import tqdm

datasets = ['pc','cr','subj']
aug_methods = ['eda_augmenter','wordnet_augmenter','aeda_augmenter','backtranslation_augmenter','clare_augmenter']
#aug_methods = ['backtranslation_augmenter','clare_augmenter']
n_samples = [1,2,4,8]

# modify the path to your data (traub1000_sample.txt or train.txt or ...)

def aug_samples(dataset_name, aug_method, n_sample,org=True):
    path = f'data/{dataset_name}/train1000_sample.txt'
    data = load_data(path)
    
    
    augmented_data = augment_text(data, aug_method,fraction=1,pct_words_to_swap=0.2 ,transformations_per_example=n_sample,
                    label_column='class',target_column='text',include_original=org)

    augmented_data = augmented_data[['class','text']]
    augmented_data.to_csv(f'data/{dataset_name}/train_{aug_method}_{n_sample}_includeOrg_{org}.csv',index=False)

    np.savetxt(f'data/{dataset_name}/train_{aug_method}_{n_sample}_includeOrg_{org}.txt', augmented_data.values, fmt='%s', delimiter='\t')
    print(f'{dataset_name} {aug_method} {n_sample} include original {org} done')
    

if __name__ == '__main__':
    for dataset in tqdm(datasets):
        for aug_method in tqdm(aug_methods):
            for n_sample in tqdm(n_samples):
                aug_samples(dataset,aug_method,n_sample)