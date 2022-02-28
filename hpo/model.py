""" This is a dummy baseline. It is just supposed to check if ingestion and 
scoring are called properly.
"""
import os
import logging
import csv 
import time
import copy
import json
import random
import psutil

from metadl.api.api import MetaLearner, Learner, Predictor

import numpy as np

import gin

import torch
import torch.nn
import torch.optim as optim

from hpo.helpers import *
from hpo.utils import *

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class MyMetaLearner(MetaLearner):

    def __init__(self, dataset, validation_episodes, num_classes, class_weights, logger = None, val_interval = 30):
        super().__init__()

        self.dataset = dataset
        self.validation_episodes = validation_episodes
        self.num_classes = num_classes
        self.class_weights = class_weights

        self.device = torch.device('cuda:0')
        self.logger = logger
        self.val_interval = val_interval

    def meta_fit(self, config, budget) -> Learner:
        """
        Args:
            None
        Returns:
            MyLearner object : a Learner that stores the meta-learner's 
                learning object. (e.g. a neural network trained on meta-train
                episodes)
        """

        # Initialize the model with the chosen configuration
        model = initialize_model(config['architecture'], self.num_classes).to(self.device)

        criterion = torch.nn.CrossEntropyLoss(weight = self.class_weights.to(self.device))
        
        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay = config['weight_decay'])
        elif config['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay = config['weight_decay'], momentum=config['momentum'])
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, budget/config['sched_decay_interval'])

        train_dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=config['batch_size'], num_workers = 2, pin_memory = True, shuffle=True, generator=torch.Generator().manual_seed(42))

        # INFO ON USAGE
        self.logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        # Train the model on the batch and transfer the weights by returning MyLearner(trained_learner)
        
        running_loss = AverageMeter("loss")
        best_accuracy = 0
        best_params = copy.deepcopy(model.state_dict())

        time_elapsed = 0
        latest_sched_update = 0
        latest_val_update = 0
        
        self.logger.info("Beginning to the meta-model training...")
        model.train()
        while True:

            for imgs, labels in train_dataloader:
                
                start = time.time()

                optimizer.zero_grad()
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                output = model(imgs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                end = time.time()

                time_elapsed += end-start
                latest_sched_update += end-start
                latest_val_update += end-start

                running_loss.update(loss.item())

                update_scheduler = latest_sched_update > config['sched_decay_interval']
                update_val = (latest_val_update > self.val_interval) or (time_elapsed >= budget and running_loss.count > 1)
                
                if update_scheduler:
                    lr_scheduler.step()
                    self.logger.info(f'Learning rate is set to: {lr_scheduler.get_last_lr()[0]}')
                    latest_sched_update = 0

                if update_val:
                    self.logger.info(f"Beginning validation on {round(time_elapsed, 2)} sec")
                    val_start = time.time()
                    val_accuracy = self.evaluate(MyLearner(model, config['architecture']))
                    val_end = time.time()
                    val_time_taken = val_end-val_start

                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        a = time.time()
                        best_params = copy.deepcopy(model.state_dict())
                        self.logger.info("Best model has changed!")

                    self.logger.info(f"Training time elapsed: {round(time_elapsed, 2)} sec. Validation time elapsed: {round(val_time_taken, 2)} sec. {round(budget-time_elapsed, 2)} sec left on budget.")
                    self.logger.info(f"Train loss: {round(running_loss.avg, 4)} steps: {running_loss.count} | Val accuracy: {round(val_accuracy, 4)} best: {round(best_accuracy, 4)}")
                    
                    running_loss.reset()

                    latest_val_update = 0

                if time_elapsed > budget:
                    break

            if time_elapsed > budget:
                    break
            else:
                self.logger.info("Completed a full iteration over the meta-training set")

        model.load_state_dict(best_params)
        return MyLearner(model, config['architecture'])

    def evaluate(self, learner):
        accuracy = AverageMeter("validation_accuracy")
        for ep in self.validation_episodes:
            support_images, support_labels = ep[0], ep[1]
            query_images, query_targets = ep[3], ep[4]
            predictor = learner.fit(support_images, support_labels)
            query_logits = predictor.predict(query_images)
            a = measure_acc(query_logits, query_targets)
            accuracy.update(a)

        return accuracy.avg

class MyLearner(Learner):

    def __init__(self, model, architecture, n_iters = 5, num_ways = 5):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.architecture = architecture
        self.n_iters = n_iters
        self.num_ways = num_ways

        self.device = torch.device('cuda:0')

        self.model = initialize_finetune(self.model, self.architecture, self.num_ways).to(self.device)
        self.meta_params = copy.deepcopy(self.model.state_dict())

    def process_task(self, images, labels):
        # [nbr_imgs, H, W, C] -> [nbr_imgs, C, H, W]
        images = process_images(images, 224)
        labels = torch.from_numpy(labels).long()
        
        return images, labels

    def fit(self, support_images, support_labels) -> Predictor:
        """
        Args: 
            support_images: numpy tensor [nbr_imgs, H, W, C]
            support_labels: numpy array 1-D
        Returns:
            ModelPredictor : a Predictor.
        """

        # HERE SUPPORT SET IS GIVEN 5-way 1 shot -> 5 examples
        ''''''
        self.model.load_state_dict(self.meta_params)
        support_images, support_labels = augment(support_images, support_labels, n_aug = 9)
        support_images, support_labels = self.process_task(support_images, support_labels)
        dataset = torch.utils.data.TensorDataset(support_images, support_labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=25, pin_memory = True, shuffle=True, generator=torch.Generator().manual_seed(42))

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 0.001)
        
        for i in range(self.n_iters):
            for imgs, labels in dataloader:
                optimizer.zero_grad()
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                output = self.model(imgs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        
        return MyPredictor(self.model)
    

class MyPredictor(Predictor):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

        self.device = torch.device('cuda:0')

    def predict(self, query_images):
        """ Predicts the label of the examples in the query set which is the 
        query_images in this case. The prototypes are already computed by
        the Learner.

        Args:
            query_images: 
        Returns: 
            preds : tensors, shape (num_examples, N_ways). We are using the 
                Categorical Accuracy to evaluate the predictions.

        Case 1 : The i-th prediction row contains the i-th example logits.


        Since in both cases the SparseCategoricalAccuracy behaves the same way,
        i.e. taking the argmax of the row inputs, both forms are valid.
        Note : In the challenge N_ways = 5 at meta-test time.
        """
        query_images = do_PIL(query_images)
        query_images = process_images(query_images, 224)

        with torch.no_grad():
            query_images = query_images.to(self.device)
            pred = self.model(query_images)

        return pred.detach().cpu().numpy()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Evaluation')

    # Experimentwise
    parser.add_argument("--experiment_root", default="/work/dlclarge2/ozturk-experiments/per_few_shot_augmentation_datasets_x_configs_v2")
    parser.add_argument("--datasets_main_dir", default="/work/dlclarge2/ozturk-experiments/few_shot_finalized")
    parser.add_argument("--configs_main_dir", default="experiments/02_22/configs")
    parser.add_argument("--dataset_name", help="dataset_name", default = "omniglot")
    parser.add_argument("--config_name", help="config_name", default = "omniglot")
    parser.add_argument("--iter_n", type = int, help="the number of iteration", default = 0)
    # Evaluationwise
    parser.add_argument('--budget', type=float, help='Budget used during the evaluation.', default=600)
    parser.add_argument('--task_id', help = 'SLURM ARRAY TASK ID', default = 0)
    
    args=parser.parse_args()

    verbosity_level = "INFO"
    logger = get_logger(verbosity_level)

    dataset_dir = os.path.join(args.datasets_main_dir, args.dataset_name)
    config_dir = os.path.join(args.configs_main_dir, args.config_name+'.json')
    output_dir = os.path.join(args.experiment_root, args.config_name, args.dataset_name+'_'+str(args.iter_n))
    print(output_dir)
    os.makedirs(output_dir, exist_ok = True)

    with open(config_dir, 'r') as f:
        config = json.load(f)
    config_str = '\n'.join([k+': '+str(v) for k, v in config.items()])
    logger.info(f"Configuration:\n{config_str}")

    ############################################################################################################################
    # Processes on the same node gets killed when the program tries to resize images at the same time
    # To get less out-of-memory error 
    #wait = int(args.task_id) % 4
    #logger.info(f'Waiting for {wait*120} seconds before processing...')
    #time.sleep(wait*120)
    '''
    cpu_usage = psutil.cpu_percent(5)
    total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    ram_usage = round((used_memory/total_memory) * 100, 2)
    while ram_usage > 65 or cpu_usage > 65:
        time.sleep(5)
        cpu_usage = psutil.cpu_percent(5)
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        ram_usage = round((used_memory/total_memory) * 100, 2)
    logger.info(f"RAM memory used: {ram_usage} The CPU usage is: {cpu_usage}")
    '''
    ############################################################################################################################
    
    images, labels = get_augmented_train_set(dataset_dir)
    labels = remap_labels(labels)

    classes, counts = np.unique(labels, return_counts = True)
    num_classes = len(classes)
    class_weights = torch.tensor(1/counts).float()

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

    validation_episodes, test_episodes = get_augmented_evaluation_set(dataset_dir)
    total_start = time.time()

    meta_learner = MyMetaLearner(dataset, validation_episodes, num_classes, class_weights, logger)
    learner = meta_learner.meta_fit(config, args.budget)

    del dataset
    del validation_episodes
    del meta_learner
    
    test_start = time.time()

    accuracy = AverageMeter("test_accuracy")
    accuracy_per_episode = []
    for ep in test_episodes:
        support_images, support_labels = ep[0], ep[1]
        query_images, query_targets = ep[3], ep[4]
        predictor = learner.fit(support_images, support_labels)
        query_logits = predictor.predict(query_images)
        a = measure_acc(query_logits, query_targets)
        accuracy.update(a)
        accuracy_per_episode.append(a)

    total_end = time.time()

    del test_episodes
    del learner
    del predictor

    result = {"accuracy": accuracy.avg, "time_elapsed": total_end-total_start, "test_time_elapsed": total_end-test_start, "accuracy_per_episode": accuracy_per_episode}

    logger.info(f"Time elapsed: {result['time_elapsed']}")
    logger.info(f"Time elapsed for test: {result['test_time_elapsed']}")
    logger.info(f"Mean accuracy {result['accuracy']}")

    with open(os.path.join(output_dir, "result.json"), 'w') as f:
        json.dump(result, f)

    with open("hpo/evaluation_success.args", "a") as success:
        success.write(' '.join(['--dataset_name', args.dataset_name, '--config_name', args.config_name, '--iter_n', str(args.iter_n)]))
        success.write('\n')
    