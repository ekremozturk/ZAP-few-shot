import os
import json

import numpy as np
import pandas as pd

from hpo.utils import get_augmented_train_set, get_n_aug

def extract(datasets_main_folder, savepath = "experiments/"):

    datasets = os.listdir(datasets_main_folder)
    datasets.sort()

    records = []
    for d in datasets:
        dataset_dir = os.path.join(datasets_main_folder, d)
        images, labels = get_augmented_train_set(dataset_dir)

        classes, counts = np.unique(labels, return_counts = True)
        num_classes = len(classes)
        n_aug = get_n_aug(images)+1
        _size = images.shape[0]
        final_size = min(30000, n_aug*_size)

        print(f'Batch images shape : {images.shape}')
        print(f'Batch labels shape : {labels.shape}')
        print(f"Unique labels : {num_classes}")
        print(f"Final dataset_size : {final_size}")

        records.append((_size, final_size, num_classes))

    metafeatures_df = pd.DataFrame(np.stack(records), index = datasets, columns = ['num_samples', 'num_aug_samples', 'num_classes'])
    metafeatures_df.to_csv(savepath)

if __name__ == '__main__':
    datasets_main_folder = "/work/dlclarge2/ozturk-experiments/few_shot_finalized"
    savepath = "experiments/02_22/simple_meta_features.csv"
    extract(datasets_main_folder, savepath)