from __future__ import absolute_import

import argparse
import os
import sys
sys.path.append(os.getcwd())
import warnings
import logging
import pickle
import time

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("AutoFolio"))
from AutoFolio.autofolio.autofolio import AutoFolio

warnings.simplefilter(action="ignore", category=FutureWarning)

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

verbosity_level = "INFO"
logger = get_logger(verbosity_level)


class AutoFolioPipeline(object):
    """
    A wrapper class for the execution of AutoFolio code.
    """

    def __init__(self, args):
        self.wallclock_limit = args.wallclock_limit
        self.maximize = args.maximize
        self.autofolio_model_path = args.autofolio_model_path
        self.verbose = args.verbose
        self.tune = args.tune
        self.runcount_limit = args.runcount_limit

        self.perf_csv_path = args.perf_csv
        self.feat_csv_path = args.feat_csv

        self.cv_csv = args.cv_csv

        self.autofolio = None  # AutoFolio_old instance

    def create_autofolio_function_call(self, arguments, perf_data_path, feature_data_path, budget=60, subprocess_call=True):
        if not subprocess_call:
            function_call = os.path.join(arguments["autofolio_dir"], "scripts", "autofolio")
            function_call += " --performance_csv=" + perf_data_path
            function_call += " --feature_csv=" + feature_data_path
            function_call += " --wallclock_limit=" + str(budget)
            function_call += "--maximize"
            return function_call
        else:
            autofolio_program = os.path.join(arguments["autofolio_dir"], "scripts", "autofolio")
            return [
                autofolio_program,
                "--performance_csv",
                perf_data_path,
                "--feature_csv",
                feature_data_path,
                "wallclock_limit",
                str(budget),
                "--maximize",
            ]

    def train_and_save_autofolio_model(self):
        self.autofolio = AutoFolio()
        af_args = {
            "performance_csv": self.perf_csv_path,
            "feature_csv": self.feat_csv_path,
            "wallclock_limit": self.wallclock_limit,
            "maximize": self.maximize,
            "save": self.autofolio_model_path,
            "verbose": self.verbose,
            "tune": self.tune,
            "runcount_limit": self.runcount_limit,
            "cv_csv": self.cv_csv
        }

        return self.autofolio.run_cli(af_args)  # 10 fold cv result


if __name__ == "__main__":
    """ Pipeline arguments"""
    parser = argparse.ArgumentParser("AutoFolioPipeline")

    parser.add_argument("--perf_csv", type=str, default="experiments/02_22/perf_matrix.csv")
    parser.add_argument("--cv_csv", type=str, default="experiments/02_22/cv_folds.csv")
    parser.add_argument("--feat_csv", type=str, default="experiments/02_22/simple_meta_features.csv")
    parser.add_argument("--exp_suffix", type=str, default="simple") # name of the spesific run 

    """ AutoFolio_old arguments """
    parser.add_argument("--maximize", type=bool, default=True, 
        help = "Whether to maximize/minimize the objective.")
    parser.add_argument("--autofolio_model_path", type=str, default=None, 
        help = "If None: Performs inner-CV and reports the performance, else: Trains an AF model over whole meta-dataset")
    parser.add_argument("--verbose", type=str, default="INFO", 
        help = "INFO/WARNING/ERROR etc.")
    parser.add_argument("--tune", action = "store_true", default = False, 
        help = "Whether to optimize the AS pipeline comfiguration")
    parser.add_argument("--wallclock_limit", type=str, default=str(4*3600),
        help = "Time budget for the execution of meta-configuration optimization")
    parser.add_argument("--runcount_limit", type=str, default=str(1000),
        help = "Number of configurations to try")

    args, _ = parser.parse_known_args()

    if args.autofolio_model_path is not None:
        os.makedirs(args.autofolio_model_path, exist_ok = True)
        args.autofolio_model_path = os.path.join(args.autofolio_model_path, args.exp_suffix)

    autofolio_pipeline = AutoFolioPipeline(args=args)
    autofolio_pipeline.train_and_save_autofolio_model()

    
