from typing import Dict
import pandas as pd

from relgen.data.dataset import Dataset
from relgen.model.diffusionmodel import DiffusionModel
from relgen.synthesizer import BaseSynthesizer


class DiffusionSynthesizer(BaseSynthesizer):
    def __init__(self, dataset: Dataset):
        super(DiffusionSynthesizer, self).__init__(dataset)

    def fit(self, data: Dict[str, pd.DataFrame], condition=None):
        raise NotImplementedError

    def sample(self, condition=None):
        raise NotImplementedError
