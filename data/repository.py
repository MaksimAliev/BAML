import itertools
import logging
import multiprocessing
import os
import pprint
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Any, TypeVar
import numpy as np
import pandas as pd
import openml
from imblearn.datasets import fetch_datasets
from loguru import logger

from data.domain import Dataset


class DatasetRepository(ABC):
    def __init__(self, *args, **kwargs):
        self._datasets: List[Dataset] = []

    @abstractmethod
    def load_datasets(self, ids: Optional[Union[List[int], range]] = None, X_and_y = False) -> List[Dataset]:
        raise NotImplementedError()

    @abstractmethod
    def load_dataset(self, id: int, X_and_y = False) -> Optional[Dataset]:
        raise NotImplementedError()

    @property
    def datasets(self):
        return self._datasets

class BinaryImbalancedDatasetRepository(DatasetRepository):
    def __init__(self, verbose = False):
        super().__init__()

        # TODO: Should it be async?
        self._raw_datasets = fetch_datasets(data_home='datasets/imbalanced', verbose=verbose)

    @logger.catch
    def load_dataset(self, id: int, X_and_y = False) -> Optional[Dataset]:
        for i, (dataset_name, dataset_data) in enumerate(self._raw_datasets.items(), 1):
            if i == id:
                x = dataset_data.get("data")
                y = dataset_data.get("target")[:, np.newaxis]

                if not X_and_y:
                    x = pd.DataFrame(np.concatenate((x, y), axis=1))
                    y = None
                else:
                    # TODO: make it work.
                    x = pd.DataFrame(x)
                    y = pd.Series(y.T[0], dtype=str)
                
                return Dataset(
                    name=dataset_name,
                    x=x,
                    y=y
                )
            elif i > id:
                raise ValueError(f"Id {id}) is out of range.")

    def load_datasets(self, ids: Optional[Union[List[int], range]] = None, X_and_y = False) -> List[Dataset]:
        if ids is None:
            range_start = 1
            range_end = len(self._raw_datasets.keys()) + 1
            ids = range(range_start, range_end)
        
        logger.debug(f"Chosen dataset identifiers for a benchmark: {ids}.")
        for id in ids:
            dataset = self.load_dataset(id, X_and_y)
            if dataset is not None:
                self._datasets.append(dataset)
        
        return self.datasets


# REFACTOR THIS SHIT.
class OpenMLRepository(DatasetRepository):
    def __init__(self, suite_id=271):
        super().__init__()
        self._suite_id = suite_id
        openml.config.set_root_cache_directory("datasets/openml")

    # TODO: parallelize.
    def load_dataset(self, id: Optional[int] = None, X_and_y = False) -> Optional[Dataset]:
        task = openml.tasks.get_task(id)
        dataset = task.get_dataset()
        x, y, categorical_indicator, feature_names = dataset.get_data(
            target=dataset.default_target_attribute)

        if not X_and_y:
            x = pd.DataFrame(np.concatenate((x, y), axis=1))
            y = None
        else:
            x = pd.DataFrame(x)
            y = pd.Series(y, name=dataset.default_target_attribute, dtype="category")
        
        return Dataset(
            name=dataset.name,
            x=x,
            y=y
        )

    @logger.catch
    def load_datasets(self, ids: Optional[Union[List[int], range]] = None, X_and_y = False) -> List[Dataset]:
        benchmark_suite = openml.study.get_suite(suite_id=self._suite_id)
        for i, id in enumerate(benchmark_suite.tasks):
            if ids is not None and i not in ids:
                raise ValueError(f"Id {id}) is out of range.")
            self._datasets.append(self.load_dataset(id))
        
        return self.datasets
