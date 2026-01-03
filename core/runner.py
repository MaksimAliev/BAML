import itertools
import logging
import pprint
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union, Optional, List, cast, final
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from loguru import logger
from sklearn.base import BaseEstimator

from core.automl import H2O, AutoML, AutoGluon
from data.domain import Dataset, Task
from data.repository import DatasetRepository, BinaryImbalancedDatasetRepository
from utils.helpers import infer_positive_class_label, split_data_on_train_and_test


# TODO: support presets and leaderboard.
class BAML:
    def __init__(
        self,
        repository = 'binary-imbalanced',
        automl = 'ag',
        validation_metric = 'f1',
        timeout = None,
        test_metrics: Optional[List[str]] = None,
        verbosity = 1,
        *args,
        **kwargs
    ):
        self._automl: AutoML
        self._validation_metric: str
        self._fitted_model = None

        self.repository = repository
        self.automl = (automl, args, kwargs)
        self.validation_metric = validation_metric
        self._timeout = timeout
        self._test_metrics = test_metrics
        self._verbosity = verbosity

        self._configure_environment(self._verbosity)

    def _configure_environment(self, logging_level: int) -> None:
        if logging_level < 2:
            logger.remove()
            logger.add(sys.stdout, level='INFO')

    @logger.catch
    def run(self) -> None:
        self.repository.load_datasets()
        
        for dataset in self.repository.datasets:
            self._run_on_dataset(dataset)

    def _run_on_dataset(self, dataset: Dataset, x_and_y = False) -> None:
        logger.info(f"Running on the dataset: Dataset(name={dataset.name}).")

        if not x_and_y:
            y_label = dataset.x.columns[-1]
            y = dataset.x[y_label]
            x = dataset.x.drop([y_label], axis=1)
        else:
            x = dataset.x
            y = dataset.y
            y_label = y.name
        
        x_train, x_test, y_train, y_test = split_data_on_train_and_test(x, y)
        y_train = y_train.astype(object)

        pos_class_label = infer_positive_class_label(y_train)

        if x_and_y:
            training_dataset = Dataset(
                name=dataset.name,
                x=x_train,
                y=y_train,
            )
        else:
            df = pd.concat((x_train, y_train),axis=1)
            training_dataset = Dataset(
                name=dataset.name,
                x=df
            )

        training_dataset.size = int(x_train.memory_usage(deep=True).sum() / (1024 ** 2))
        logger.debug(f"Train sample size < {training_dataset.size + 1}mb.")

        task  = Task(
            dataset=training_dataset,
            metric=self.validation_metric,
            timeout=self._timeout
        )

        start_time = time.time()
        self._automl.fit(task)

        time_passed = time.time() - start_time
        logger.info(f"Training took {time_passed // 60} min.")

        y_predicted = self._automl.predict(x_test)

        metrics = {self.validation_metric}
        if self._test_metrics is not None:
            for metric in self._test_metrics:
                metrics.add(metric)
        self._automl.score(metrics, y_test, y_predicted, pos_class_label)

    @property
    def repository(self) -> DatasetRepository:
        return self._repository
    
    @repository.setter
    def repository(self, value: str):
        if value == 'binary-imbalanced':
            verbosity = self._verbosity >= 2
            self._repository = BinaryImbalancedDatasetRepository(verbosity)
        else:
            raise ValueError(
                f"""
                Invalid value of repository parameter:{value}.
                Options available: ['binary-imbalanced'].
                """
            )
    
    @property
    def validation_metric(self) -> str:
        return self._validation_metric
    
    @validation_metric.setter
    def validation_metric(self, value: str):
        if value not in [
            'f1',
            'precision',
            'recall',
            'roc_auc',
            'average_precision',
            'balanced_accuracy',
            'mcc',
            'accuracy'
        ]:
            raise ValueError(
                f"""
                Invalid value of metric parameter: {value}.
                Options available: [
                    'f1',
                    'precision',
                    'recall',
                    'roc_auc',
                    'average_precision',
                    'balanced_accuracy',
                    'mcc',
                    'accuracy'].
                """)
        self._validation_metric = value
    
    @property
    def automl(self) -> AutoML:
        return self._automl

    @automl.setter
    def automl(self, value: Tuple[str, Tuple[Any, ...], Dict[str, Any]]):
        if value[0] == 'ag':
            self._automl = AutoGluon(*value[1], **value[2])
        elif value[0] == 'h2o':
            self._automl = H2O(*value[1], **value[2])
        else:
            raise ValueError(
                f"""
                Invalid value of automl parameter: {value[0]}.
                Options available: ['ag', 'h2o'].
                """)
