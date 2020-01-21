"""
Copyright (C) 2019  F.Hoffmann-La Roche Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import os
import json
import numpy as np
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.base import MetaEstimatorMixin, ClassifierMixin
from precision_fda_brain_biomarker.apps.util import is_bit_set_at_index
from precision_fda_brain_biomarker.models.baselines.base_model import HyperparamMixin, BaseModel


class SwitchModel(_BaseComposition, ClassifierMixin, MetaEstimatorMixin, HyperparamMixin):
    def __init__(self, feature_names=[], switch_features=["CANCER_TYPE_GBM"], base_model="LogisticRegression"):
        from precision_fda_brain_biomarker.apps.main import MainApplication

        self.models = [(str(i), MainApplication.get_model_type_for_method_name(base_model)())
                       for i in range(2**len(switch_features))]
        self.base_model = base_model
        self.feature_names = feature_names
        self.switch_features = switch_features
        super(SwitchModel, self).__init__()

    @staticmethod
    def get_hyperparameter_ranges():
        ranges = {

        }
        return ranges

    def _split_data(self, x):
        x_split_i_actives = []
        for i in range(len(self.switch_features)):
            idx = self.feature_names.index(self.switch_features[i])
            x_split_i_active = np.where(np.isclose(x[:, idx], 1.0))[0]
            x_split_i_actives.append(set(x_split_i_active))

        x_split, split_indices = [], []
        for i in range(2**len(self.switch_features)):
            all_indices = set(np.arange(len(x)))
            for j in range(len(self.switch_features)):
                if is_bit_set_at_index(i, j):
                    all_indices = all_indices.intersection(x_split_i_actives[j])
                else:
                    all_indices -= x_split_i_actives[j]
            cur_indices = sorted(all_indices)
            split_indices.append(cur_indices)
            x_split.append([x[idx] for idx in cur_indices])
        assert sum(map(len, x_split)) == len(x)
        return x_split, split_indices

    def fit(self, x, y, validation_data=None):
        x_split, split_indices = self._split_data(x)
        y_split = [[y[idx] for idx in indices] for indices in split_indices]
        for model, x_i, y_i in zip(self.models, x_split, y_split):
            if len(x_i) != 0:
                model[1].fit(x_i, y_i)
        return None

    def _predict(self, x, try_predict_proba=False):
        if self.models is None:
            raise AssertionError("Model must be fit before calling predict.")

        all_y_pred = []
        x_split, split_indices = self._split_data(x)
        for model, x_i in zip(self.models, x_split):
            if len(x_i) == 0:
                y_pred = []
            elif hasattr(model[1], "predict_proba") and try_predict_proba:
                y_pred = model[1].predict_proba(x_i)
            else:
                y_pred = model[1].predict(x_i)
            all_y_pred.append(y_pred)
        assert len(all_y_pred) == len(split_indices)

        first_valid_index = 0
        for first_valid_index in range(len(all_y_pred)):
            if isinstance(all_y_pred[first_valid_index], np.ndarray):
                break

        y_pred = np.zeros((sum(map(len, all_y_pred)),) + all_y_pred[first_valid_index].shape[1:])
        for cur_y_pred, indices in zip(all_y_pred, split_indices):
            if len(indices) != 0:
                y_pred[np.array(indices)] = cur_y_pred
        return y_pred

    def predict(self, x):
        return self._predict(x, try_predict_proba=False)

    def predict_proba(self, x):
        return self._predict(x, try_predict_proba=True)

    @staticmethod
    def get_config_file_name():
        return "switch_config.json"

    def get_config(self):
        config = {
            "feature_names": self.feature_names,
            "switch_features": self.switch_features,
            "base_model": self.base_model,
        }
        return config

    @staticmethod
    def load(file_path):
        from precision_fda_brain_biomarker.apps.main import MainApplication

        directory_path = os.path.dirname(os.path.abspath(file_path))
        config_file_name = SwitchModel.get_config_file_name()
        config_file_path = os.path.join(directory_path, config_file_name)
        with open(config_file_path, "r") as fp:
            config = json.load(fp)

        feature_names = config["feature_names"]
        switch_features = config["switch_features"]
        base_model = config["base_model"]

        models = []
        filename, file_extension = os.path.splitext(file_path)
        for i in range(2**len(switch_features)):
            cls = MainApplication.get_model_type_for_method_name(base_model)
            models.append((str(i), cls.load(filename + str(i) + file_extension)))
        instance = SwitchModel(feature_names=feature_names, switch_features=switch_features, base_model=base_model)
        instance.models = models
        return instance

    def save(self, file_path, overwrite=True):
        BaseModel.save_config(file_path, self.get_config(), self.get_config_file_name(), overwrite, SwitchModel)
        filename, file_extension = os.path.splitext(file_path)
        for i, model in enumerate(self.models):
            model[1].save(filename + str(i) + file_extension)

    @staticmethod
    def get_save_file_type():
        return ".pickle"

    def set_params(self, **params):
        super(SwitchModel, self)._set_params('models', **params)
        return self

    def get_params(self, deep=True):
        return super(SwitchModel, self)._get_params('models', deep=deep)