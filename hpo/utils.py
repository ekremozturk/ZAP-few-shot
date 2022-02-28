import os
import sys
import logging
import torch
import numpy as np

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def get_logger(verbosity_level, use_error_log=False):
    """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO score.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(filename)s: %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


def get_augmented_numpy_dataset(dataset_dir):
    train_x = np.load(os.path.join(dataset_dir,'train_x.npy'), allow_pickle = True)
    train_y = np.load(os.path.join(dataset_dir,'train_y.npy'), allow_pickle = True)
    valid_episodes = np.load(os.path.join(dataset_dir,'valid_episodes.npy'), allow_pickle = True)
    test_episodes = np.load(os.path.join(dataset_dir,'test_episodes.npy'), allow_pickle = True)

    return (train_x, train_y), valid_episodes, test_episodes

def get_augmented_train_set(dataset_dir):
    train_x = np.load(os.path.join(dataset_dir,'train_x.npy'), allow_pickle = True)
    train_y = np.load(os.path.join(dataset_dir,'train_y.npy'), allow_pickle = True)

    return train_x, train_y

def get_augmented_evaluation_set(dataset_dir):
    valid_episodes = np.load(os.path.join(dataset_dir,'valid_episodes.npy'), allow_pickle = True)
    test_episodes = np.load(os.path.join(dataset_dir,'test_episodes.npy'), allow_pickle = True)

    return valid_episodes, test_episodes


def measure_acc(logits, targets):
    predictions = torch.max(torch.from_numpy(logits), 1).indices
    a = torch.eq(torch.from_numpy(targets), predictions)
    a = sum(a)/len(a)

    return a.item()


def get_n_aug(images):
    n_aug = int(8e4)//images.shape[0]
    return min(n_aug, 9)
    

def remap_labels(labels):
    new_label = 0
    new_label_dict = dict()
    temp_labels = []
    for l in labels:
        if l in new_label_dict:
            temp_labels.append(new_label_dict[l])
        else:
            new_label_dict[l] = new_label
            new_label += 1
            temp_labels.append(new_label_dict[l])

    return np.array(temp_labels)


class AverageMeter(object):

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    '''
    root_dir = "/work/dlclarge2/ozturk-experiments/few_shot_finalized"
    meta_datasets = os.listdir(root_dir)
    print(meta_datasets)
    print(len(meta_datasets))

    f = open('hpo/hpo.args', 'w+')
    for d in meta_datasets:
        f.write('--dataset_name '+d)
        f.write('\n')
    f.close()

    
    evaluation_args = []
    for i in range(3):
        for c in meta_datasets:
            for d in meta_datasets:
                evaluation_args.append(' '.join(['--dataset_name', d, '--config_name', c, '--iter_n', str(i)]))

    import random
    random.seed(42)
    random.shuffle(evaluation_args)
    f = open('hpo/evaluation.args', 'w+')
    for args in evaluation_args:
        f.write(args)
        f.write('\n')
    f.close()

    '''
    '''
    from hpo.get_incumbent import get_incumbent_config

    hpo_result_dir = "/work/dlclarge2/ozturk-experiments/per_few_shot_augmentation_hpo_v2"
    completed_eval_n = get_incumbent_config(hpo_result_dir, None)

    root_dir = "/work/dlclarge2/ozturk-experiments/few_shot_finalized"
    meta_datasets = os.listdir(root_dir)

    f = open('hpo/hpo_warmstart.args', 'w+')
    for d in meta_datasets:
        n_iter = 60-completed_eval_n[d]
        if n_iter == 0:
            continue
        prev_dir = os.path.join(hpo_result_dir, d)
        f.write(' '.join(['--dataset_name', d, '--n_iterations', str(n_iter), '--previous_run_dir', prev_dir]))
        f.write('\n')
    f.close()

    '''
    f = open('hpo/evaluation.args', 'r+')
    jobs = f.readlines()
    f.close()

    f_suc = open('hpo/evaluation_success.args', 'r+')
    jobs_suc = f_suc.readlines()
    f_suc.close()
    
    print(len(jobs), len(jobs_suc))

    f_fail = open('hpo/evaluation_failure.args', 'w+')
    for j in jobs:
        if j not in jobs_suc:
            f_fail.write(j)
    
    


