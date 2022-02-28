import time

import torch
import numpy as np

from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpo.model import MyMetaLearner, MyLearner, MyPredictor
from hpo.helpers import *
from hpo.utils import *

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class FewShotWorker(Worker):
    def __init__(self, dataset_dir, n_repeat, logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
        self.dataset_dir = dataset_dir
        (images, labels), self.validation_episodes, self.test_episodes = get_augmented_numpy_dataset(dataset_dir)
        labels = remap_labels(labels)
        classes, counts = np.unique(labels, return_counts = True)
        self.num_classes = len(classes)
        self.class_weights = torch.tensor(1/counts).float()

        self.logger.info(f'Batch images shape : {images.shape}')
        self.logger.info(f'Batch labels shape : {labels.shape}')
        self.logger.info(f"Unique labels : {self.num_classes}")
        
        n_aug = get_n_aug(images)
        images, labels = augment(images, labels, n_aug = n_aug)
        
        self.logger.info(f'Augmented images shape : {images.shape}')
        self.logger.info(f'Augmented labels shape : {labels.shape}')

        images = process_images(images, 224)
        labels = torch.from_numpy(labels).long()
        self.train_dataset = torch.utils.data.TensorDataset(images, labels)

        self.meta_learner = MyMetaLearner(self.train_dataset, self.validation_episodes, self.num_classes, self.class_weights, self.logger)
        self.n_repeat = n_repeat

    def compute(self, config, budget, working_directory, *args, **kwargs):

        self.logger.info(config)

        try:
            overall_accuracy = AverageMeter("overall_test_accuracy")
            for _ in range(self.n_repeat):
                learner = self.meta_learner.meta_fit(config, budget)

                start = time.time()
                accuracy = AverageMeter("test_accuracy")

                for ep in self.test_episodes:
                    support_images, support_labels = ep[0], ep[1]
                    query_images, query_targets = ep[3], ep[4]
                    predictor = learner.fit(support_images, support_labels)
                    query_logits = predictor.predict(query_images)
                    a = measure_acc(query_logits, query_targets)
                    accuracy.update(a)

                overall_accuracy.update(accuracy.avg)
                
                end = time.time()

                self.logger.info(f"Time elapsed for evaluation: {end-start}")
                self.logger.info(f"Mean accuracy {accuracy.avg}")
                self.logger.info(f"Memory allocated: {torch.cuda.memory_allocated()/1024**2}M reserved: {torch.cuda.memory_reserved()/1024**2}M")

            return ({
                'loss': 1-overall_accuracy.avg, # remember: HpBandSter always minimizes!
                'info': {}
                            
            })
        except:
            self.logger.info("Probably CUDA OOM")
            return ({
                'loss': 1.0, # remember: HpBandSter always minimizes!
                'info': {}
                            
            })
        

def run_BOHB(args):

    verbosity_level = "INFO"
    logger = get_logger(verbosity_level)
    logging.basicConfig(level=logging.DEBUG)

    if args.previous_run_dir is not None:
        previous_result = hpres.logged_results_to_HBS_result(args.previous_run_dir)
        args.bohb_root_path = args.bohb_root_path + '_warmstart'
    else:
        previous_result = None

    NS = hpns.NameServer(run_id=args.job_id, nic_name='eth0', working_directory=args.bohb_root_path)
    ns_host, ns_port = NS.start()

    time.sleep(5)
    
    worker = FewShotWorker(args.dataset_dir, 
                           args.n_repeat, 
                           logger,
                           run_id=args.job_id,
                           host=ns_host,
                           nameserver=ns_host,
                           nameserver_port=ns_port
                           )
    
    worker.run(background=True)

    result_logger = hpres.json_result_logger(directory=args.bohb_root_path, overwrite=True)

    optimizer = BOHB(
        configspace=get_configspace(),
        run_id=args.job_id,
        host=ns_host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        min_budget=args.budget,
        max_budget=args.budget,
        result_logger=result_logger,
        logger=logger,
        previous_result=previous_result
    )

    res = optimizer.run(n_iterations=args.n_iterations)

    # Shutdown
    optimizer.shutdown(shutdown_workers=True)
    NS.shutdown()

    return res

def print_res(res):
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/args.budget))

if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser(description='BOHB')

    # Experimentwise
    parser.add_argument("--experiment_root", default="/work/dlclarge2/ozturk-experiments/per_few_shot_augmentation_hpo_v2")
    parser.add_argument("--datasets_main_dir", default="/work/dlclarge2/ozturk-experiments/few_shot_finalized")
    parser.add_argument("--dataset_name", help="e.g 0-cifar100") # MUST
    parser.add_argument("--job_id", default=None)

    parser.add_argument('--budget',   type=float, help='Budget used during the optimization.', default=300)
    parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=60)
    parser.add_argument("--n_repeat", type=int, help="Number of runs per iteration", default=3)
    parser.add_argument("--previous_run_dir", default=None, help="Path to a previous run to warmstart from")

    args=parser.parse_args()

    args.bohb_root_path = os.path.join(args.experiment_root, args.dataset_name)
    args.dataset_dir = os.path.join(args.datasets_main_dir, args.dataset_name)
    args.job_id = args.dataset_name if args.job_id is None else args.job_id

    test_start = time.time()
    res = run_BOHB(args)
    test_end = time.time()
    print(f" Time elapsed for BOHB iterations {test_end-test_start}")
    print_res(res)


