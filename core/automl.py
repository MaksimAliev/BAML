import logging
import os
import pprint
import re
from abc import ABC, abstractmethod
from io import StringIO
from typing import Optional, Set, Union, final, List, Dict
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import fbeta_score, balanced_accuracy_score, matthews_corrcoef, recall_score, precision_score, average_precision_score, roc_auc_score, accuracy_score
from autogluon.tabular import TabularDataset as AutoGluonTabularDataset, TabularPredictor as AutoGluonTabularPredictor
from autogluon.core.metrics import make_scorer
from loguru import logger
import ray

from data.domain import Dataset, Task


class AutoML(ABC):
    def __init__(self, verbosity=1):
        if verbosity > 2:
            raise ValueError()
        self._verbosity = verbosity
    
    @abstractmethod
    def fit(
        self,
        task: Task,
    ) -> None:
        raise NotImplementedError()

    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> Union[pd.DataFrame, np.ndarray]:
        if self._fitted_model is None:
            raise NotFittedError()
        predictions = self._fitted_model.predict(X_test)
        
        return predictions

    @final
    def score(
        self,
        metrics: Set[str],
        y_test: Union[pd.DataFrame, np.ndarray],
        y_pred: Union[pd.DataFrame, np.ndarray],
        pos_label: Optional[int] = None,
    ) -> None:

        calculate_metric_score_kwargs = {
            'y_test': y_test,
            'y_pred': y_pred,
            'pos_label': pos_label
        }

        for metric in metrics:
            self._calculate_metric_score(
                metric,
                **calculate_metric_score_kwargs)

    @final
    def _log_val_loss_alongside_fitted_model(self, losses: Dict[str, np.float64]) -> None:
        for m, l in losses.items():
            # TODO: different output for leaderboard.
            logger.info(f"Validation loss: {abs(l):.3f}")

            model_log = pprint.pformat(f"ML model: {m}", compact=True)
            logger.info(model_log)

    def _configure_environment(self, seed=42) -> None:
        np.random.seed(seed)
        logger.debug(f"Seed = {seed}.")
        
        self._seed = seed

    @final
    def _calculate_metric_score(self, metric: str, *args, **kwargs) -> None:
        y_test = kwargs.get("y_test")
        y_pred = kwargs.get("y_pred")
        pos_label = kwargs.get("pos_label")

        if metric == 'f1':
            score = fbeta_score(y_test, y_pred, beta=1, pos_label=pos_label)
            logger.info(f"F1 score: {score:.3f}.")
        elif metric == 'precision':
            score = precision_score(y_test, y_pred, pos_label=pos_label)
            logger.info(f"Precision score: {score:.3f}.")
        elif metric == 'recall':
            score = recall_score(y_test, y_pred, pos_label=pos_label)
            logger.info(f"Recall score: {score:.3f}.")
        elif metric == 'roc_auc':
            score = roc_auc_score(y_test, y_pred)
            logger.info(f"ROC AUC score: {score:.3f}.")
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_test, y_pred)
            logger.info(f"Balanced accuracy score: {score:.3f}.")
        elif metric == 'average_precision':
            score = average_precision_score(y_test, y_pred, pos_label=pos_label)
            logger.info(f"Average precision score: {score:.3f}.")
        elif metric == 'mcc':
            score = matthews_corrcoef(y_test, y_pred)
            logger.info(f"MCC score: {score:.3f}.")
        elif metric == 'accuracy':
            score = accuracy_score(y_test, y_pred)
            logger.info(f"Balanced accuracy score: {score:.3f}.")
        else:
            raise ValueError(
                f"""
                Invalid value encountered among values of test_metrics parameter:{metric}.
                Metrics available: [
                'f1',
                'precision',
                'recall',
                'roc_auc',
                'average_precision',
                'balanced_accuracy',
                'mcc',
                'accuracy'].        
                """)

    def __str__(self):
        return self.__class__.__name__


class AutoGluon(AutoML):
    def __init__(
        self,
        preset='medium',
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._preset = preset
        self._fitted_model: Optional[AutoGluonTabularPredictor] = None

    @property
    def preset(self):
        return self._preset

    @preset.setter
    def preset(self, preset):
        if preset not in ['medium', 'good', 'high', 'best', 'extreme']:
            raise ValueError(
                f"""
                Invalid value of preset parameter: {preset}.
                Options available: [
                    'medium',
                    'good',
                    'high',
                    'best',
                    'extreme'
                ].
                """)
        self._preset = preset

    @logger.catch
    def predict(self, X_test: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        if self._fitted_model is None:
            raise NotFittedError()
        dataset_test = AutoGluonTabularDataset(X_test)
        predictions = self._fitted_model.predict(dataset_test)

        return predictions

    @logger.catch
    def fit(
        self,
        task: Task,
    ) -> None:
        dataset = task.dataset
        metric = task.metric
        y_label = dataset.y_label
        
        if metric not in [
            'f1',
            'precision',
            'recall',
            'roc_auc',
            'average_precision',
            'balanced_accuracy',
            'mcc',
            'accuracy'
        ]:
            raise ValueError(f"Metric {metric} is not supported by AutoGluon.")
        
        if isinstance(dataset.X, np.ndarray):
            dataframe = pd.DataFrame(data=np.column_stack([dataset.X, dataset.y]))
        elif isinstance(dataset.X, pd.DataFrame):
            dataframe = pd.DataFrame(
                data=np.column_stack([dataset.X, dataset.y]),
                columns=[*dataset.X.columns, y_label])
        else:
            raise TypeError()

        ag_dataset = AutoGluonTabularDataset(dataframe)

        predictor = AutoGluonTabularPredictor(
            problem_type='binary',
            label=y_label,
            eval_metric=metric,
            verbosity=self._verbosity
        ).fit(ag_dataset)

        val_scores = predictor.leaderboard().get('score_val')
        if val_scores is None or len(val_scores) == 0:
            logger.error("No model found.")
            return

        best_model = predictor.model_best

        logger.info(f"Best model found: {best_model}")

        predictor.delete_models(models_to_keep=best_model, dry_run=False)

        self._fitted_model = predictor
