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
import sys
import glob
import pickle
import numpy as np
from os.path import join
from collections import defaultdict
from sklearn.ensemble import VotingClassifier
from precision_fda_brain_biomarker.models.baselines.base_model import HyperparamMixin
from precision_fda_brain_biomarker.models.baselines.logistic_regression import LogisticRegression


class Ensemble(VotingClassifier, HyperparamMixin):
    def __init__(self, estimators=list([]), voting="soft", weights=None):
        super(Ensemble, self).__init__(estimators=estimators, voting=voting, weights=weights)
        self.stack_model = None

    @staticmethod
    def preprocess_y_pred(y_pred):
        y_pred = y_pred.swapaxes(0, 1)
        if len(y_pred.shape) == 3:
            y_pred = y_pred[..., -1]
        return y_pred

    def fit(self, X, y, sample_weight=None):
        if self.voting == "stacked":
            y_pred = self._collect_probas(X)
            y_pred = Ensemble.preprocess_y_pred(y_pred)
            self.stack_model = LogisticRegression()
            self.stack_model.fit(y_pred, y)

    def predict(self, X):
        if self.voting == "stacked":
            y_pred = self._collect_probas(X)
            y_pred = Ensemble.preprocess_y_pred(y_pred)
            y_pred_stacked = self.stack_model.predict(y_pred)
            return y_pred_stacked
        else:
            return super(Ensemble, self).predict(X)

    def predict_proba(self, X):
        if self.voting == "stacked":
            y_pred = self._collect_probas(X)
            y_pred = Ensemble.preprocess_y_pred(y_pred)
            y_pred_stacked = self.stack_model.predict_proba(y_pred)
            return y_pred_stacked
        else:
            return super(Ensemble, self).predict_proba(X)

    @staticmethod
    def read_from_pickle_path(path):
        with open(path, "rb") as fp:
            return pickle.load(fp)

    @staticmethod
    def get_estimator_at_path(path):
        from precision_fda_brain_biomarker.apps.main import MainApplication

        program_args_path = join(path, MainApplication.get_config_file_name())
        program_args = Ensemble.read_from_pickle_path(program_args_path)
        cls = MainApplication.get_model_type_for_method_name(program_args["method"])
        extension = cls.get_save_file_type()
        model_path = join(path, "model" + extension)

        program_args["load_existing"] = model_path
        program_args["do_train"] = False

        sub_app = MainApplication(program_args, do_setup=False, do_print=False)
        model = sub_app.get_model()
        return model

    @staticmethod
    def get_topk_models_in_path(search_directory, subchallenge, outer_index, inner_index, k=10, sort_key="f1",
                                pickle_file_name="eval_score.pickle"):
        results = Ensemble.get_all_models_in_path(search_directory, subchallenge, pickle_file_name)
        sorted_results = list(sorted(results.items(), key=lambda x: x[1][sort_key]))[::-1][:k]
        model_names = list(map(lambda x: os.path.join(search_directory, x[0],
                                                      "outer_{}".format(outer_index),
                                                      "inner_{}".format(inner_index)), sorted_results))
        return model_names

    @staticmethod
    def get_all_models_in_path(search_directory, subchallenge, pickle_file_name="eval_score.pickle"):
        from precision_fda_brain_biomarker.apps.results_to_csv import ResultsToCSVApplication

        results = defaultdict(dict)

        files = glob.glob(os.path.join(search_directory, "*"))
        for file_candidate in files:
            if os.path.isdir(file_candidate):
                pickle_file = os.path.join(file_candidate, pickle_file_name)
                if os.path.isfile(pickle_file):
                    with open(pickle_file, "rb") as fp:
                        results_dict = pickle.load(fp)
                        base_file_name = os.path.basename(file_candidate)
                        current_subchallenge = ResultsToCSVApplication.get_subchallenge_for_file_name(
                            base_file_name
                        )
                        if current_subchallenge != subchallenge:
                            continue

                        results[base_file_name] = results_dict
                else:
                    print("WARN:", pickle_file, "was not present.", file=sys.stderr)
        return results

    @staticmethod
    def get_weight_at_path(path, weight_key="auroc"):
        test_score_path = join(path, "eval_score.pickle")
        test_score = Ensemble.read_from_pickle_path(test_score_path)
        weight = test_score[weight_key]
        return weight

    @staticmethod
    def get_estimators_from_paths(paths, weight_key=None):
        estimators, weights = [], []
        for path in paths:
            outers = sorted(glob.glob(join(path, "outer_*")), key=lambda name: int(name.split("outer_")[-1]))
            if len(outers) == 0:
                outers = [path]
            for outer_dir in outers:
                inners = sorted(glob.glob(join(outer_dir, "inner_*")), key=lambda name: int(name.split("inner_")[-1]))
                if len(inners) == 0 and path != "":
                    inners = [path]
                for inner_dir in inners:
                    model = Ensemble.get_estimator_at_path(inner_dir)
                    if weight_key is None:
                        weight = 1.0
                    else:
                        weight = Ensemble.get_weight_at_path(inner_dir, weight_key=weight_key)
                    estimators.append(model)
                    weights.append(weight)
        weights = np.array(weights)
        weights /= np.sum(weights)
        estimators = [("model_{:d}".format(idx), model) for idx, model in enumerate(estimators)]
        return estimators, weights

    @staticmethod
    def get_hyperparameter_ranges():
        ranges = {

        }
        return ranges

    @staticmethod
    def get_save_file_type():
        return ".tar"

    @staticmethod
    def load(file_path):
        return None  # TODO

    def save(self, file_path, overwrite=True):
        pass  # TODO
