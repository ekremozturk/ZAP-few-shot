import os
import json

import numpy as np
import pandas as pd

def create(experiments_main_folder, savepath = "experiments/"):

    datasets = os.listdir(experiments_main_folder)
    datasets.sort()
    print(datasets, len(datasets))

    result_dict = dict()
    for d in datasets:
        dataset_path = os.path.join(experiments_main_folder, d)
        configs_x3 = os.listdir(dataset_path)
        configs_x3.sort()

        result_dict[d] = dict()

        for c in configs_x3:
            result_path = os.path.join(dataset_path, c, 'result.json')

            try:
                with open(result_path, 'r') as f:
                    result = json.load(f)
            except:
                print(f"No result produced for dataset {d} configuration {c[:-2]} iteration {c[-1]}")
            
            c = c[:-2]

            if c in result_dict[d]:
                result_dict[d][c].append(result['accuracy'])
            else:
                result_dict[d][c] = [result['accuracy']]

    records = []
    for d, ca in result_dict.items():
        print(d)
        d_rec = []
        for c, a in ca.items():
            d_rec.append(np.mean(a))
        print(len(d_rec))
        records.append(d_rec)


    records = np.stack(records).T
    print(records.shape)

    perf_df = pd.DataFrame(records, index = datasets, columns = datasets)
    perf_df.to_csv(savepath, index = True)

if __name__ == '__main__':
    experiments_main_folder = "/work/dlclarge2/ozturk-experiments/per_few_shot_augmentation_datasets_x_configs_v2"
    savepath = "experiments/02_22/perf_matrix.csv"
    #create(experiments_main_folder, savepath)

    datasets = os.listdir(experiments_main_folder)
    datasets.sort()
    print(datasets, len(datasets))

    cores = [d.split("-")[-1] for d in datasets]
    import numpy as np
    cores = np.unique(cores)
    from random import shuffle
    shuffle(cores)
    fold_size = len(cores)//5+1
    folds = [float(i) for i in range(5)]*fold_size
    print(cores, len(cores))
    print(folds, len(folds))
    per_d_folds = []
    for d in datasets:
        for c, k in zip(cores, folds):
            if c in d:
                per_d_folds.append(k)
    folds_df = pd.DataFrame(per_d_folds, index = datasets, columns = ["fold"])
    folds_df.to_csv("experiments/02_22/cv_folds.csv", index = True)
