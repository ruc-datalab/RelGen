from typing import Dict
import pandas as pd

from relgen.data.dataset import Dataset
from relgen.model.armodel import ARModel
from relgen.synthesizer import BaseSynthesizer


class ARSynthesizer(BaseSynthesizer):
    def __init__(self, dataset: Dataset):
        super(ARSynthesizer, self).__init__(dataset)

    def fit(self, data: Dict[str, pd.DataFrame], condition=None):
        raise NotImplementedError

    def sample(self, condition=None):
        raise NotImplementedError
