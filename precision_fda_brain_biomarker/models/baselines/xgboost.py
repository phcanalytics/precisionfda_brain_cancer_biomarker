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
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from precision_fda_brain_biomarker.models.baselines.base_model import BaseModel, HyperparamMixin


class XGBoost(BaseModel, BaseEstimator, ClassifierMixin, HyperparamMixin):
    def __init__(self, max_depth=6, learning_rate=0.3, objective="binary:logistic", eval_metric="auc",
                 l2_weight=1, l1_weight=0, num_boost_rounds=10, early_stopping_rounds=10, min_split_loss=0):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.eval_metric = eval_metric
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight
        self.num_boost_rounds = num_boost_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.min_split_loss = min_split_loss
        self.model = None

    @staticmethod
    def get_hyperparameter_ranges():
        ranges = {
            "subsample": (0.25, 0.5, 0.75, 1.0),
            "max_depth": (2, 3, 4, 5, 6, 7, 8),
            "min_split_loss": (0.0, 0.1, 1.0, 10.0),
            "learning_rate": (0.5, 0.3, 0.03, 0.003),
            "l2_weight": (1.0, 0.1, 0.001, 0.0),
            "l1_weight": (1.0, 0.1, 0.001, 0.0),
            "num_boost_rounds": (5, 10, 15, 20)
        }
        return ranges

    def _build_model_params(self, y=None):
        if y is None:
            scale_pos_weight = 1
        else:
            scale_pos_weight = sum(y) / float(len(y))

        model_params = {
            'max_depth': self.max_depth,
            'eta': self.learning_rate,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            "reg_alpha": self.l1_weight,
            "reg_lambda": self.l2_weight,
            "min_split_loss": self.min_split_loss,
            "scale_pos_weight": scale_pos_weight,
        }
        model_params.update(self.get_params())
        return model_params

    def fit(self, x, y, validation_data=None):
        model_params = self._build_model_params(y)
        train_set = xgb.DMatrix(x, label=y)
        eval_list = [(train_set, 'train')]
        if validation_data is not None:
            test_set = xgb.DMatrix(validation_data[0], label=validation_data[1])
            eval_list.append((test_set, 'eval'))
        self.model = xgb.train(model_params, train_set, self.num_boost_rounds, eval_list,
                               early_stopping_rounds=self.early_stopping_rounds)

    def predict(self, x):
        if self.model is None:
            raise AssertionError("Model must be fit before calling predict.")
        x_dmatrix = xgb.DMatrix(x)
        if hasattr(self.model, "best_ntree_limit"):
            best_ntree_limit = self.model.best_ntree_limit
        else:
            best_ntree_limit = 0
        return self.model.predict(x_dmatrix, ntree_limit=best_ntree_limit)

    def predict_proba(self, x):
        y_pred = self.predict(x)
        return np.column_stack([1-y_pred, y_pred])

    def get_config(self):
        config = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            "l1_weight": self.l1_weight,
            "l2_weight": self.l2_weight,
            "num_boost_rounds": self.num_boost_rounds,
            "early_stopping_rounds": self.early_stopping_rounds,
            "min_split_loss": self.min_split_loss,
        }
        return config

    @staticmethod
    def get_config_file_name():
        return "xgboost_config.json"

    @staticmethod
    def load(file_path):
        directory_path = os.path.dirname(os.path.abspath(file_path))
        config_file_name = XGBoost.get_config_file_name()
        config_file_path = os.path.join(directory_path, config_file_name)
        with open(config_file_path, "r") as fp:
            config = json.load(fp)

        max_depth = config["max_depth"]
        learning_rate = config["learning_rate"]
        objective = config["objective"]
        eval_metric = config["eval_metric"]
        l1_weight = config["l1_weight"]
        l2_weight = config["l2_weight"]
        num_boost_rounds = config["num_boost_rounds"]
        early_stopping_rounds = config["early_stopping_rounds"]
        min_split_loss = config["min_split_loss"]

        instance = XGBoost(max_depth=max_depth,
                           learning_rate=learning_rate,
                           objective=objective,
                           eval_metric=eval_metric,
                           l1_weight=l1_weight,
                           l2_weight=l2_weight,
                           num_boost_rounds=num_boost_rounds,
                           early_stopping_rounds=early_stopping_rounds,
                           min_split_loss=min_split_loss)

        model_params = instance._build_model_params()
        model = xgb.Booster(model_params)
        model.load_model(file_path)
        instance.model = model
        return instance

    def save(self, file_path, overwrite=True):
        directory_path = os.path.dirname(os.path.abspath(file_path))

        already_exists_exception_message = "__directory_path__ already contains a saved XGBoost instance and" \
                                           " __overwrite__ was set to __False__. Conflicting file: {}"

        config_file_name = XGBoost.get_config_file_name()
        config_file_path = os.path.join(directory_path, config_file_name)
        if os.path.exists(config_file_path) and not overwrite:
            raise ValueError(already_exists_exception_message.format(config_file_path))
        else:
            with open(config_file_path, "w") as fp:
                json.dump(self.get_config(), fp)
        self.model.save_model(file_path)

    @staticmethod
    def get_save_file_type():
        return ".model"
