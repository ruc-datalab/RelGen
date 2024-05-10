import pandas as pd
import numpy as np
import sklearn
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler

def exists(x):
	return False if x is None else True


class RelQuantileTransformer(QuantileTransformer):
    def __init__(self, seed=0, subsample=1e9, output_distribution="normal", bins=None):
        super().__init__(random_state=seed, subsample=int(subsample), output_distribution=output_distribution)
        self.bins = bins
        self.seed = seed
    
    def fit(self, data_instance):
        data_instance = data_instance.reshape(-1, 1)
        if self.bins is not None:
            self.n_quantiles = max(min(len(data_instance) // 30, 1000), 10)
        super().fit(data_instance)
    
    def fit_transform(self, data_instance):
        self.fit(data_instance)
        return super().transform(data_instance)

class ScaleTransformer:
    def __init__(self, scaler="MinMax", feature_range=(0, 1)):
        super().__init__()
        self.scaler = scaler 
        if scaler == "MinMax":
            self.Transformer = MinMaxScaler(feature_range=feature_range)
        elif scaler == "Standard":
            self.Transformer = StandardScaler()  
        else:
            raise ValueError(f"Unknown scaler {scaler}. Now have 'MinMax', 'Standard'")

    def fit(self, data_instance):
        data_instance = data_instance.reshape(-1, 1)
        self.Transformer.fit(data_instance)
    
    def transform(self, data_instance):
        return self.Transformer.transform(data_instance)

    def fit_transform(self, data_instance):
        self.fit(data_instance)
        return self.transform(data_instance)

    def inverse_transform(self, data_instance):
        return self.Transformer.inverse_transform(data_instance)

class GaussianMixtureTransformer:
    def __init__(self, n_component = 5):
        self.n_component = n_component
        self.Transformer = GaussianMixture(self.n_component)

    def fit(self, data_instance):
        data_instance = data_instance.reshape(-1, 1)
        self.Transformer.fit(data_instance)
    
    def transform(self, data_instance):
        data_instance = data_instance.reshape(-1, 1)
        trans_data = np.zeros([data_instance.shape[0], self.n_component+1])

        weights = self.Transformer.weights_
        means = self.Transformer.means_.reshape(1, self.n_component)[0]
        stds = np.sqrt(self.Transformer.covariances_).reshape(1, self.n_component)[0]

        features = (data_instance - means) / (2 * stds)
        probs = self.Transformer.predict_proba(data_instance)
        argmax = np.argmax(probs, axis=1)
        idx = np.arange(len(features))
        features = features[idx, argmax].reshape(-1, 1)
        features = np.concatenate((features, probs), axis=1)
        return features
    
    def fit_transform(self, data_instance):
        self.fit(data_instance)
        return self.transform(data_instance)

    def inverse_transform(self, data_instance):
        assert data_instance.shape[1] == self.n_component + 1
        v = data_instance[:, 0]
        u = data_instance[:, 1:]

        argmax = np.argmax(u, axis=1)
        means = self.Transformer.means_.reshape(1, self.n_component)[0]
        stds = np.sqrt(self.Transformer.covariances_).reshape(1, self.n_component)[0]

        mean = means[argmax]
        std = stds[argmax]
        inverse_data = v * 2 * std + mean 
        inverse_data = inverse_data.reshape(-1, 1)
        return inverse_data
    
    
