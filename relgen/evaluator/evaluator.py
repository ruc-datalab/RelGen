import math
from typing import List
import pandas as pd
import sklearn.metrics as metrics
import numpy as np
import sklearn.tree as tree
import json
import sklearn.ensemble._forest as forest
import sklearn
import os
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
import dython.nominal as dm
import multiprocessing
import tqdm


class Evaluator:
    def __init__(
            self,
            real_data: pd.DataFrame,
            fake_data: pd.DataFrame,
            test_data: pd.DataFrame = None,
    ):
        assert len(real_data.columns) == len(fake_data.columns)
        if test_data is not None:
            assert len(test_data.columns) == len(real_data.columns)
        self.real_data = real_data
        self.fake_data = fake_data
        self.test_data = test_data

    def eval_fidelity(self, save_path: str = None):
        dist = self._distance()
        diff_corr = self._diff_corr()
        disc_meas = self._dm()
        ret_dict = {**dist, **diff_corr, **disc_meas}
        if save_path:
            json_str = json.dumps(ret_dict)
            with open(save_path, 'w') as json_file:
                json_file.write(json_str)
        return ret_dict

    def eval_privacy(self, precision=False, save_path: str = None):
        dcr = self._dcr_nndr(is_dcr=True, precision=precision)
        nndr = self._dcr_nndr(is_dcr=False, precision=precision)
        ret_dict = {**dcr, **nndr}
        if save_path:
            json_str = json.dumps(ret_dict)
            with open(save_path, 'w') as json_file:
                json_file.write(json_str)
        return ret_dict

    def eval_diversity(self, precision=False, save_path: str = None):
        ret_dict = self._sampling_diversity(precision=precision)
        if save_path:
            json_str = json.dumps(ret_dict)
            with open(save_path, 'w') as json_file:
                json_file.write(json_str)
        return ret_dict

    def eval_histogram(self, columns: List[str] = [], save_path: str = None):
        if len(columns) == 0:
            real_data = self.real_data
            fake_data = self.fake_data
        else:
            real_data = self.real_data[columns]
            fake_data = self.fake_data[columns]
        num_columns = len(real_data.columns)
        ncols = 3
        nrows = math.ceil(num_columns / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 4 * nrows))
        if nrows == 1:
            axes = axes.reshape((1, -1))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        for i, col in enumerate(real_data.columns):
            row_idx = i // 3
            col_idx = i % 3

            axes[row_idx, col_idx].hist(real_data[col], bins=10, alpha=0.5, label='real')
            axes[row_idx, col_idx].hist(fake_data[col], bins=10, alpha=0.5, label='fake')

            axes[row_idx, col_idx].set_xticks([])

            axes[row_idx, col_idx].set_title(col)
            axes[row_idx, col_idx].legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def eval_tsne(self, save_path: str = None):
        fake_X = self.fake_data.iloc[:, :-1]
        real_X = self.real_data.iloc[:, :-1]
        real_X, fake_X = self._min_max_eu_dummies(real_X, fake_X)
        # sample 1000 rows from real data
        if real_X.shape[0] > 10000:
            real_X = real_X.sample(n=10000, random_state=200)
        # sample 1000 rows from synthetic data
        if fake_X.shape[0] > 10000:
            fake_X = fake_X.sample(n=10000, random_state=200)
        real_embedded = TSNE(
            n_components=2, learning_rate="auto", init='pca', metric='euclidean'
        ).fit_transform(real_X)
        synthetic_embedded = TSNE(
            n_components=2, learning_rate="auto", init='pca', metric='euclidean'
        ).fit_transform(fake_X)
        r_min, r_max = np.min(real_embedded, 0), np.max(real_embedded, 0)
        real_embedded = (real_embedded - r_min) / (r_max - r_min)
        s_min, s_max = np.min(synthetic_embedded, 0), np.max(synthetic_embedded, 0)
        synthetic_embedded = (synthetic_embedded - s_min) / (s_max - s_min)
        plt.figure(figsize=(6, 6))
        plt.scatter(
            real_embedded[:, 0],
            real_embedded[:, 1],
            label="real",
        )
        plt.scatter(
            synthetic_embedded[:, 0],
            synthetic_embedded[:, 1],
            label="fake",
        )
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    # a get dummy function for min max euclidean distance
    def _min_max_eu_dummies(self, train_X, test_X):
        numerical_columns = []
        for column in train_X.columns:
            if train_X[column].dtype == "object":
                train_X = pd.get_dummies(train_X, columns=[column])
                test_X = pd.get_dummies(test_X, columns=[column])
            else:
                numerical_columns.append(column)
                # min max normalization
                min_v = min(train_X[column].min(), test_X[column].min())
                max_v = max(train_X[column].max(), test_X[column].max())
                if min_v == max_v:
                    print("Warning.Mode Collapse on ", column)
                    continue
                train_X[column] = (train_X[column] - min_v) / (max_v - min_v)
                test_X[column] = (test_X[column] - min_v) / (max_v - min_v)
        columns = set(train_X.columns)
        columns = columns.union(set(test_X.columns))
        columns = list(columns)
        test_X = test_X.reindex(columns=columns, fill_value=0)
        train_X = train_X.reindex(columns=columns, fill_value=0)
        for column in columns:
            if column not in numerical_columns:
                train_X[column] = train_X[column].astype("int64")
                test_X[column] = test_X[column].astype("int64")
                train_X[column] = train_X[column] * np.sqrt(0.5)
                test_X[column] = test_X[column] * np.sqrt(0.5)
        return train_X, test_X

    # calculate the decision tree metrics, for categorical data,
    # we use one hot encoding with maintaining the max distance=1
    def _dt(self):
        # check if the last column is string
        if self.fake_data.iloc[:, -1].dtype != "object" or self.test_data is None:
            return {}
        # encode the string to int
        train_X = self.real_data.iloc[:, :-1]
        train_Y = self.real_data.iloc[:, -1]
        test_X = self.test_data.iloc[:, :-1]
        test_Y = self.test_data.iloc[:, -1]
        train_X, test_X = self._min_max_eu_dummies(train_X, test_X)
        # As a decision tree model, no need to normalize the data
        tree_model = tree.DecisionTreeClassifier()
        # check if it is a binary classification
        if len(train_Y.unique()) == 2:
            tree_model.fit(train_X, train_Y)
            tree_predict = tree_model.predict(test_X)
            tree_predict_proba = tree_model.predict_proba(test_X)
            tree_real = test_Y
            return {
                "DT Accuracy": metrics.accuracy_score(tree_real, tree_predict),
                "DT Precision": metrics.precision_score(tree_real, tree_predict, pos_label=test_Y.unique()[0]),
                "DT Recall": metrics.recall_score(tree_real, tree_predict, pos_label=test_Y.unique()[0]),
                "DT F1": metrics.f1_score(tree_real, tree_predict, pos_label=test_Y.unique()[0]),
                "DT AUC": metrics.roc_auc_score(tree_real, tree_predict_proba[:, 1])
            }
        else:
            tree_model.fit(train_X, train_Y)
            tree_predict = tree_model.predict(test_X)
            tree_real = test_Y
            return {"DT Accuracy": metrics.accuracy_score(tree_real, tree_predict),
                    "DT Macro Precision": metrics.precision_score(tree_real, tree_predict, average='macro'),
                    "DT Macro Recall": metrics.recall_score(tree_real, tree_predict, average='macro'),
                    "DT Macro F1": metrics.f1_score(tree_real, tree_predict, average='macro'),
                    "DT Micro Precision": metrics.precision_score(tree_real, tree_predict, average='micro'),
                    "DT Micro Recall": metrics.recall_score(tree_real, tree_predict, average='micro'),
                    "DT Micro F1": metrics.f1_score(tree_real, tree_predict, average='micro')
                    }

    # calculate the linear regression metrics, for categorical data, we use one hot encoding with maintaining the max
    # distance=1,for numerical data, we use min max normalization based on min,max of all data
    def _lr(self):
        if self.test_data is None:
            return {}

        train_X = self.fake_data.iloc[:, :-1]
        train_Y = self.fake_data.iloc[:, -1]
        test_X = self.test_data.iloc[:, :-1]
        test_Y = self.test_data.iloc[:, -1]
        regression_model = sklearn.linear_model.LinearRegression()
        # check if the last column is numerical
        if (
                test_Y.dtype == "int64"
                or test_Y.dtype == "float64"
        ):
            # use pd.get_dummies to encode the string to int
            train_X, test_X = self._min_max_eu_dummies(train_X, test_X)
            regression_model.fit(train_X, train_Y)
            regression_predict = regression_model.predict(test_X)
            regression_real = test_Y
            return {
                "LR Mean Absolute Error": metrics.mean_absolute_error(
                    regression_real, regression_predict
                ),
                "LR Mean Squared Error": metrics.mean_squared_error(
                    regression_real, regression_predict
                ),
                "LR Root Mean Squared Error": np.sqrt(
                    metrics.mean_squared_error(regression_real, regression_predict)
                ),
                "LR R2 Score": metrics.r2_score(regression_real, regression_predict),
            }
        else:
            return {}

    # calculate the distance between real and fake data, for categorical data, we use JS distance,
    # for numerical data, we use WD distance
    def _distance(self):
        fake_tmp = self.fake_data.copy()
        real_tmp = self.real_data.copy()
        # data preprocessing
        list_probability_real = []
        list_probability_fake = []
        list_unique_keys = []
        JS_distance_list = []
        WD_distance_list = []
        for column in real_tmp.columns:
            if real_tmp[column].dtype == "object":
                for key in real_tmp[column].unique():
                    if key not in list_unique_keys:
                        list_unique_keys.append(key)
                for key in fake_tmp[column].unique():
                    if key not in list_unique_keys:
                        list_unique_keys.append(key)
                for key in list_unique_keys:
                    list_probability_real.append(
                        real_tmp[real_tmp[column] == key].shape[0]
                        / real_tmp.shape[0]
                    )
                    list_probability_fake.append(
                        fake_tmp[fake_tmp[column] == key].shape[0]
                        / fake_tmp.shape[0]
                    )
                # calculate the JS distance
                JS_distance = distance.jensenshannon(
                    list_probability_real, list_probability_fake
                )
                JS_distance_list.append(JS_distance)
                list_probability_fake = []
                list_probability_real = []
                list_unique_keys = []
            else:
                WD_distance = wasserstein_distance(
                    real_tmp[column], fake_tmp[column]
                )
                WD_distance_list.append(WD_distance)
        dicta = {}
        if len(JS_distance_list) != 0:
            dicta["JS_distance Mean"] = np.mean(JS_distance_list)
            dicta["JS_distance Std"] = np.std(JS_distance_list)
            dicta["JS_distance Max"] = np.max(JS_distance_list)
            dicta["JS_distance Min"] = np.min(JS_distance_list)
        if len(WD_distance_list) != 0:
            dicta["WD_distance Mean"] = np.mean(WD_distance_list)
            dicta["WD_distance Std"] = np.std(WD_distance_list)
            dicta["WD_distance Max"] = np.max(WD_distance_list)
            dicta["WD_distance Min"] = np.min(WD_distance_list)
        return dicta

    def _diff_corr(self):
        continuous_columns = []
        categorical_columns = []
        for column in self.real_data.columns:
            if self.real_data[column].dtype != "object":
                continuous_columns.append(column)
            else:
                categorical_columns.append(column)
        real_matrix = \
        dm.associations(self.real_data, nominal_columns=categorical_columns, nom_nom_assoc="theil", plot=False,
                        compute_only=True)["corr"]
        fake_matrix = \
        dm.associations(self.fake_data, nominal_columns=categorical_columns, nom_nom_assoc="theil", plot=False,
                        compute_only=True)["corr"]
        diff_matrix = np.abs(real_matrix - fake_matrix)
        return {"Diff.Corr Mean": np.mean(diff_matrix).mean()}

    def _dcr_nndr(self, is_dcr=True, precision=False):
        Real_data = self.real_data.copy()
        Synthetic_data = self.fake_data.copy()
        discrete_columns = []
        continuous_columns = []
        Real_data, Synthetic_data = self._min_max_eu_dummies(Real_data, Synthetic_data)

        # calculate the weighted distance
        def _multi_get_distance(x: pd.DataFrame, y: pd.DataFrame):
            return _single_get_distance(x, y)
            # # split x data into 40 pd.DataFrame
            # x_num = x.shape[0]
            # if x_num < 40:
            #     return _single_get_distance(x, y)
            # pool = multiprocessing.Pool(processes=40)
            # lists = []
            # for j in range(40):
            #     lists.append([j, x, y])
            # result = pool.map(func=_get_x, iterable=lists)
            # pool.close()
            # pool.join()
            # distance_list = []
            # for i in result:
            #     distance_list += i
            #
            # return distance_list

        def _multi_get_NNDR_distance(x: pd.DataFrame, y: pd.DataFrame):
            return _single_get_NNDR_distance(x, y)
            # # split y data into 40 pd.DataFrame
            # y_num = y.shape[0]
            # if y_num < 40:
            #     return _single_get_NNDR_distance(x, y)
            # pool = multiprocessing.Pool(processes=40)
            # lists = []
            # for j in range(40):
            #     lists.append([j, x, y])
            # result = pool.map(func=_get_x_2, iterable=lists)
            # pool.close()
            # pool.join()
            # distance_list = []
            # for i in result:
            #     distance_list += i
            # return distance_list

        if is_dcr:
            if not precision:
                # sample 2000 rows from synthetic data
                if Synthetic_data.shape[0] > 100:
                    Synthetic_data = Synthetic_data.sample(n=100, random_state=200)
                if Real_data.shape[0] > 100:
                    Real_data = Real_data.sample(n=100, random_state=200)
            # calculate the DCR
            DCR_list = _multi_get_distance(Real_data, Synthetic_data)
            # return the 5th percentile of DCR
            # compute the 5% percentile of DCR
            return {"DCR 5 percentile": np.percentile(DCR_list, 0.05)}

        else:
            if not precision:
                # sample 2000 rows from synthetic data
                if Synthetic_data.shape[0] > 100:
                    Synthetic_data = Synthetic_data.sample(n=100, random_state=200)
                if Real_data.shape[0] > 100:
                    Real_data = Real_data.sample(n=100, random_state=200)
            # calculate the NNDR
            NNDR_list = _multi_get_NNDR_distance(Real_data, Synthetic_data)
            # return the 5th percentile of NNDR
            return {"NNDR 5 percentile": np.percentile(NNDR_list, 0.05)}

    def _dm(self):
        # add label real to the last column of real data
        data_1 = self.real_data.copy()
        data_1["label"] = 1
        # add label fake to the last column of fake data
        data_2 = self.fake_data.copy()
        data_2["label"] = 0
        # combine real and fake data
        data = pd.concat([data_1, data_2])
        # split data into train and test with 80% and 20% respectively
        train_data = data.sample(frac=0.8, random_state=200)
        test_data = data.drop(train_data.index)
        train_x = train_data.iloc[:, :-1]
        train_y = train_data.iloc[:, -1]
        test_x = test_data.iloc[:, :-1]
        test_y = test_data.iloc[:, -1]
        train_x, test_x = self._min_max_eu_dummies(train_x, test_x)
        # use decision tree to train the model
        tree_model = tree.DecisionTreeClassifier()
        tree_model.fit(train_x, train_y)
        tree_predict = tree_model.predict(test_x)
        accuracy = metrics.accuracy_score(test_y, tree_predict)
        return {"Discriminator Measure": accuracy}

    def _sampling_diversity(self, precision=False):
        Real_data = self.real_data.copy()
        Synthetic_data = self.fake_data.copy()
        discrete_columns = []
        continuous_columns = []
        Real_data, Synthetic_data = self._min_max_eu_dummies(Real_data, Synthetic_data)

        # 目前有bug，暂时没有启用多线程逻辑
        def _multi_get_distance_dict(x: pd.DataFrame):
            # split x data into 40 pd.DataFrame
            x_num = x.shape[0]
            if x_num < 40:
                return _single_get_distance_dict(x, x, k=2, start=0)
            pool = multiprocessing.Pool(processes=40)
            lists = []
            for j in range(40):
                lists.append([j, x, x])
            result = pool.map(func=_get_dict_3, iterable=lists)
            pool.close()
            pool.join()
            distance_dict = {}
            for i in result:
                distance_dict += i
            return distance_dict

        # we now sample 200 rows from real data
        if not precision:
            if Real_data.shape[0] > 200:
                Real_data = Real_data.sample(n=200, random_state=200)
        return {"Sampling Diversity": _single_get_sampling_diversity(Real_data, Synthetic_data,
                                                                     _single_get_distance_dict(Real_data))}


def _get_x(i):  # DCR
    if i[0] == 39:
        return _single_get_distance(i[1], i[2].iloc[int(i[2].shape[0] * i[0] / 40):])
    else:
        return _single_get_distance(i[1],
                                    i[2].iloc[int(i[2].shape[0] * i[0] / 40):int(i[2].shape[0] * (i[0] + 1) / 40)])


def _get_x_2(i):  # NNDR
    if i[0] == 39:
        return _single_get_NNDR_distance(i[1], i[2].iloc[int(i[2].shape[0] * i[0] / 40):])
    else:
        return _single_get_NNDR_distance(i[1],
                                         i[2].iloc[int(i[2].shape[0] * i[0] / 40):int(i[2].shape[0] * (i[0] + 1) / 40)])


def _get_dict_3(i):  # Sampling Diversity
    if i[0] == 39:
        return _single_get_distance_dict(whole_data=i[1], part_data=i[2].iloc[int(i[2].shape[0] * i[0] / 40):], k=2,
                                         start=int(i[2].shape[0] * i[0] / 40))


def _get_sampling_diversity_4(i):  # Sampling Diversity
    if i[0] == 39:
        return _single_get_sampling_diversity(i[1], i[2].iloc[int(i[2].shape[0] * i[0] / 40):], i[3])
    else:
        return _single_get_sampling_diversity(i[1], i[2].iloc[int(i[2].shape[0] * i[0] / 40):int(
            i[2].shape[0] * (i[0] + 1) / 40)], i[3])


def _get_pairwise_weighted_distance(x, y):  # DCR NNDR
    distance = 0
    for column in x.index:
        distance += np.sqrt(np.square(x[column] - y[column]))
    return np.sqrt(distance / len(x.index))


def _single_get_distance(x: pd.DataFrame, y: pd.DataFrame):  # DCR
    min_distance_list = []
    for row_id in range(y.shape[0]):
        min_distance = 10000
        for row_id_2 in range(x.shape[0]):
            distance = _get_pairwise_weighted_distance(x.iloc[row_id_2], y.iloc[row_id])
            if distance < min_distance:
                min_distance = distance
        min_distance_list.append(min_distance)
    return min_distance_list


def _single_get_distance_dict(whole_data: pd.DataFrame, part_data: pd.DataFrame = None, k=2,
                              start=0):  # Sampling Diversity
    if part_data is None:
        part_data = whole_data
    distance_dict = {}
    for row_id in range(part_data.shape[0]):
        distance_list = []
        for row_id_2 in range(whole_data.shape[0]):
            distance = _get_pairwise_weighted_distance(part_data.iloc[row_id], whole_data.iloc[row_id_2])
            distance_list.append(distance)
        distance_list.sort()
        distance_dict[row_id + start] = distance_list[k - 1]
    return distance_dict


def _single_get_sampling_diversity(x: pd.DataFrame, y: pd.DataFrame, z: dict):  # Sampling Diversity
    sampling_diversity = 0
    num = x.shape[0]
    for row_id in range(x.shape[0]):
        for row_id_2 in range(y.shape[0]):
            distance = _get_pairwise_weighted_distance(x.iloc[row_id], y.iloc[row_id_2])
            if distance < z[row_id]:
                sampling_diversity += 1
                break
    return sampling_diversity / num


def _single_get_NNDR_distance(x: pd.DataFrame, y: pd.DataFrame):  # NNDR
    distance_ratio_list = []
    for row_id in range(y.shape[0]):
        min_distance = 10000
        second_min_distance = 10000
        for row_id_2 in range(x.shape[0]):
            distance = _get_pairwise_weighted_distance(x.iloc[row_id_2], y.iloc[row_id])
            if distance < min_distance:
                second_min_distance = min_distance
                min_distance = distance
            elif distance < second_min_distance:
                second_min_distance = distance
        distance_ratio_list.append(min_distance / second_min_distance)
    return distance_ratio_list


# metrics userfriendly process , do not modify
def _load_metrics_dict(config_json):
    metrics_json = config_json
    metrics_dict = {}
    for group in metrics_json["metrics_grouped"]:
        metrics_dict[group["type"]] = []
        for metric in group["name"]:
            metrics_dict[group["type"]].append(metric)
    return metrics_dict


def _load_metrics_help_dict(config_json):
    metrics_json = config_json
    metrics_dict = {}
    for metric in metrics_json["metrics_help"]:
        metrics_dict[metric["name"]] = metric["description"]
    return metrics_dict


def _metrics_transfer(user_metrics_list, metrics_dict):
    metrics_list = []
    for metric in user_metrics_list:
        if metric in metrics_dict.keys():
            for metric in metrics_dict[metric]:
                metrics_list.append(metric)
        else:
            in_list = 0
            for key in metrics_dict.keys():
                if metric in metrics_dict[key]:
                    metrics_list.append(metric)
                    in_list = 1
                    break
                else:
                    continue
            if in_list == 0:
                raise ValueError("The metric you input is not in the metrics list")
    return metrics_list
