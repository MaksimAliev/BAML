import uuid
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    id: int
    name: str
    x: pd.DataFrame
    y: Optional[pd.Series] = None
    y_label: Optional[str] = None
    size: Optional[int] = None

@dataclass(frozen=True)
class Task:
    dataset: Dataset
    metric: str
    timeout: Optional[int] = None
    seed: int = 42
