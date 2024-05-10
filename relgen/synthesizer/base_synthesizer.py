from typing import Dict
import logging
from logging import getLogger
import pandas as pd

from relgen.data.dataset import Dataset
from relgen.model.base_model import BaseModel


class BaseSynthesizer:
    """Synthesizer Class is used to manage the training and sampling processes of models.
    BaseSynthesizer is an abstract class in which the fit() method should be implemented according
    to different training strategies.
    """

    def __init__(self, dataset: Dataset):
        logging.basicConfig(format='%(message)s', level=logging.DEBUG, handlers=[logging.StreamHandler()])
        self.logger = getLogger()
        self.dataset = dataset
        if not dataset.is_fitted:
            raise ValueError("Dataset is not fitted")

    def fit(self, data: Dict[str, pd.DataFrame], condition=None):
        raise NotImplementedError

    def sample(self, condition=None):
        raise NotImplementedError

    def save(self, save_path: str):
        raise NotImplementedError

    def load(self, load_path: str):
        raise NotImplementedError
