import os
import sys
sys.path.append(os.getcwd())
import pickle

import pandas as pd

from hpo.utils import *
from hpo.helpers import *

from hpo.task2vec.task2vec import Task2Vec
from hpo.task2vec.models import get_model
import hpo.task2vec.task_similarity as task_similarity

def calculate_dataset_x_augmentation_embeddings(
    datasets_main_dir: str, 
    dataset_names: list, 
    probe: str,
    skip_layers: int,
    method: str, 
    max_samples: int):

    embeddings = []
    meta_features = dict()
    dataset_dirs = [os.path.join(datasets_main_dir, name) for name in dataset_names]
    
    for name, dataset_dir in zip(dataset_names, dataset_dirs):
        
        logger.info(f"Embedding {name}")
        
        images, labels = get_augmented_train_set(dataset_dir)

        classes, counts = np.unique(labels, return_counts = True)
        num_classes = len(classes)

        logger.info(f'Batch images shape : {images.shape}')
        logger.info(f'Batch labels shape : {labels.shape}')
        logger.info(f"Unique labels : {num_classes}")

        n_aug = get_n_aug(images)
        logger.info(f'Augmentating for {n_aug} times...')
        images, labels = augment(images, labels, n_aug = n_aug)
        logger.info(f'Augmentating completed. Processing...')
        images = process_images(images, 224)
        labels = torch.from_numpy(labels).long()
        logger.info(f'Augmented images shape : {images.size()}')
        logger.info(f'Augmented labels shape : {labels.size()}')

        dataset = torch.utils.data.TensorDataset(images, labels)
        del images
        del labels

        probe_network = get_model(probe, pretrained=True, num_classes=num_classes).cuda()
        task2vec = Task2Vec(probe_network, max_samples=max_samples, skip_layers=skip_layers, method = method, loader_opts = {'batch_size': 100})
        embedding, metrics = task2vec.embed(dataset)
        embeddings.append(embedding)
        
        meta_features[name] = {i: embedding.hessian[i] for i in range(len(embedding.hessian))}

        logger.info(f"Embedding {name} completed!")
        logger.info(metrics.avg)

        del dataset
        del probe_network
        del task2vec
    
    return embeddings, meta_features

def dump_embeddings(embeddings, task_names, output_dir):
    pickle.dump(embeddings, open(os.path.join(output_dir, 'embeddings.pkl'), 'wb'))
    pickle.dump(task_names, open(os.path.join(output_dir, 'task_names.pkl'), 'wb'))

def convert_metadata_to_df(metadata):
    k, v = list(metadata.items())[0]
    columns = sorted(v.keys())
    columns_edited = False

    features_lists = []
    indices = []

    for key, values_dict in sorted(metadata.items()):
        indices.append(key)
        feature_list = [values_dict[k] for k in sorted(values_dict.keys())]

        # below loop flattens feature list since there are tuples in it &
        # it extends columns list accordingly
        for i, element in enumerate(feature_list):
            if type(element) is tuple:
                # convert tuple to single list elements
                slce = slice(i, i + len(element) - 1)

                feature_list[slce] = list(element)

                if not columns_edited:
                    columns_that_are_tuples = columns[i]
                    new_columns = [
                        columns_that_are_tuples + "_" + str(i) for i in range(len(element))
                    ]
                    columns[slce] = new_columns
                    columns_edited = True

        features_lists.append(feature_list)

    return pd.DataFrame(features_lists, columns=columns, index=indices)


def dump_meta_features_df_and_csv(meta_features, output_path, file_name="metafeatures", samples_along_rows=False, n_samples=None):

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        
    if not isinstance(meta_features, pd.DataFrame):
        df = convert_metadata_to_df(meta_features)
    else:
        df = meta_features

    df.to_csv(os.path.join(output_path, file_name+".csv"))
    logger.info("meta features data dumped to: {}".format(output_path))


def main(args):
    ### ALL DATASETS ALL AUGMENTATIONS ###

    dataset_names = os.listdir(args.dataset_dir)

    embeddings, meta_features = \
    calculate_dataset_x_augmentation_embeddings(args.dataset_dir, 
                                                dataset_names, 
                                                args.probe_network, 
                                                args.skip_layers,
                                                args.method,
                                                args.max_samples)
    
    task_names = dataset_names

    if args.plot_dist_mat:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        task_similarity.plot_distance_matrix(embeddings, task_names, savepath = os.path.join(args.output_dir, 'dist_mat.png'))

    return embeddings, task_names, meta_features

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser("Task2VecPipeline")
    parser.add_argument("--dataset_dir", type=str, default = "/work/dlclarge2/ozturk-experiments/few_shot_finalized/")
    parser.add_argument("--dataset_group", type=str, default = 'all') # all/training/validation
    parser.add_argument("--output_dir", type=str, default = 'experiments/02_22/task2vec_montecarlo/')
    parser.add_argument("--plot_dist_mat", type=bool, default = True)
    parser.add_argument("--probe_network", type=str, default = 'resnet34')
    parser.add_argument("--skip_layers", type=int, default = 0)
    parser.add_argument("--method", type=str, default = 'montecarlo')
    parser.add_argument("--max_samples", type=int, default = 10000)
    args, _ = parser.parse_known_args()

    verbosity_level = "INFO"
    logger = get_logger(verbosity_level)

    #### ORIGINAL RUN SCRIPT ####
    embeddings, task_names, meta_features = main(args)
    dump_embeddings(embeddings, task_names, args.output_dir)
    dump_meta_features_df_and_csv(meta_features=meta_features, output_path=args.output_dir)




