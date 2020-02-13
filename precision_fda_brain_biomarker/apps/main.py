#!/usr/bin/env python
"""
Copyright (C) 2019  F.Hoffmann-La Roche Ltd
Copyright (C) 2019  Patrick Schwab, ETH Zurich

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
from __future__ import print_function

import os
import sys
import glob
import pickle
import inspect
import importlib
import numpy as np
import pandas as pd
from os.path import join
from sklearn.pipeline import Pipeline
from keras.callbacks import TensorBoard
from precision_fda_brain_biomarker.apps.util import time_function
from precision_fda_brain_biomarker.data_access.data_loader import DataLoader
from precision_fda_brain_biomarker.models.baselines.ensemble import Ensemble
from precision_fda_brain_biomarker.apps.evaluate import EvaluationApplication
from precision_fda_brain_biomarker.models.model_evaluation import ModelEvaluation
from precision_fda_brain_biomarker.models.baselines.switch_model import SwitchModel
from precision_fda_brain_biomarker.apps.parameters import clip_percentage, parse_parameters
from precision_fda_brain_biomarker.data_access.generator import make_generator, get_last_row_id
from precision_fda_brain_biomarker.models.feature_selection.pipeline import Pipeline as FeatureSelectionPipeline


class MainApplication(EvaluationApplication):
    def __init__(self, args, do_setup=True, do_print=True):
        super(MainApplication, self).__init__(args, do_setup=do_setup, do_print=do_print)

    def save_config(self):
        with open(self.get_config_path(), "wb") as fp:
            pickle.dump(self.args, fp, pickle.HIGHEST_PROTOCOL)

    def load_data(self):
        resample_with_replacement = self.args["resample_with_replacement"]
        seed = int(np.rint(self.args["seed"]))

        self.training_set, self.validation_set, self.test_set, self.input_dim, self.output_dim, self.feature_names = \
            self.get_data(seed=seed, resample=resample_with_replacement, resample_seed=seed)

        self.args["feature_names"] = self.feature_names

    def setup(self):
        super(MainApplication, self).setup()

    def get_data(self, seed=0, resample=False, resample_seed=0):
        dataset = self.args["dataset"].lower()

        phase = 1
        if "ph2" in dataset:
            phase = 2

        if phase == 1:
            if "sc1" in dataset:
                print("INFO: Loading Sub Challenge 1 (SC1) data.", file=sys.stderr)
                return DataLoader.get_data_ph1_sc1(self.args, seed=seed,
                                                   do_resample=resample, resample_seed=resample_seed)
            elif "sc2" in dataset:
                print("INFO: Loading Sub Challenge 2 (SC2) data.", file=sys.stderr)
                return DataLoader.get_data_ph1_sc2(self.args, seed=seed,
                                                   do_resample=resample, resample_seed=resample_seed)
            elif "sc3" in dataset:
                print("INFO: Loading Sub Challenge 3 (SC3) data.", file=sys.stderr)
                return DataLoader.get_data_ph1_sc3(self.args, seed=seed,
                                                   do_resample=resample, resample_seed=resample_seed)
        else:
            if "sc1" in dataset:
                print("INFO: Loading Sub Challenge 1 (SC1) data (Ph2).", file=sys.stderr)
                return DataLoader.get_data_ph2_sc1(self.args, seed=seed,
                                                   do_resample=resample, resample_seed=resample_seed)
            elif "sc2" in dataset:
                print("INFO: Loading Sub Challenge 2 (SC2) data (Ph2).", file=sys.stderr)
                return DataLoader.get_data_ph2_sc2(self.args, seed=seed,
                                                   do_resample=resample, resample_seed=resample_seed)
            elif "sc3" in dataset:
                print("INFO: Loading Sub Challenge 3 (SC3) data (Ph2).", file=sys.stderr)
                return DataLoader.get_data_ph2_sc3(self.args, seed=seed,
                                                   do_resample=resample, resample_seed=resample_seed)

    def get_num_losses(self):
        return 1

    def make_train_generator(self, randomise=True, stratify=True):
        batch_size = int(np.rint(self.args["batch_size"]))
        seed = int(np.rint(self.args["seed"]))
        num_losses = self.get_num_losses()

        train_generator, train_steps = make_generator(self.args,
                                                      dataset=self.training_set,
                                                      batch_size=batch_size,
                                                      num_losses=num_losses,
                                                      shuffle=randomise,
                                                      seed=seed)

        return train_generator, train_steps

    def make_validation_generator(self, randomise=False):
        batch_size = int(np.rint(self.args["batch_size"]))
        num_losses = self.get_num_losses()

        val_generator, val_steps = make_generator(self.args,
                                                  dataset=self.validation_set,
                                                  batch_size=batch_size,
                                                  num_losses=num_losses,
                                                  shuffle=randomise)
        return val_generator, val_steps

    def make_test_generator(self, randomise=False, do_not_sample_equalised=False):
        batch_size = int(np.rint(self.args["batch_size"]))
        num_losses = self.get_num_losses()

        test_generator, test_steps = make_generator(self.args,
                                                    dataset=self.test_set,
                                                    batch_size=batch_size,
                                                    num_losses=num_losses,
                                                    shuffle=randomise)
        return test_generator, test_steps

    @staticmethod
    def get_config_file_name():
        return "program_args.pickle"

    def get_config_path(self):
        return join(self.args["output_directory"], MainApplication.get_config_file_name())

    def get_model_path(self):
        cls = MainApplication.get_model_type_for_method_name(self.args["method"])
        if cls is not None:
            extension = cls.get_save_file_type()
            model_path = join(self.args["output_directory"], "model" + extension)
        else:
            print("WARN: Unable to retrieve class for provided method name [", self.args["method"], "].",
                  file=sys.stderr)
            model_path = join(self.args["output_directory"], "model.pickle")
        return model_path

    def get_feature_selection_path(self):
        cls = MainApplication.get_feature_selection_type_for_name(self.args["feature_selection"])
        if cls is not None:
            extension = cls.get_save_file_type()
            model_path = join(self.args["output_directory"], "feature_selection" + extension)
        else:
            print("WARN: Unable to retrieve class for provided feature selection name [",
                  self.args["feature_selection"], "].",
                  file=sys.stderr)
            model_path = join(self.args["output_directory"], "feature_selection.pickle")
        return model_path

    def get_prediction_path(self, set_name):
        return join(self.args["output_directory"], set_name + "_predictions.tsv")

    def get_attribution_path(self, prefix=""):
        return join(self.args["output_directory"], prefix + "attributions.tsv")

    def get_thresholded_prediction_path(self, set_name):
        return join(self.args["output_directory"], set_name + "_predictions.thresholded.tsv")

    def get_hyperopt_parameters(self):
        hyper_params = {}

        resample_with_replacement = self.args["resample_with_replacement"]
        if resample_with_replacement:
            base_params = {
                "seed": [0, 2 ** 32 - 1],
            }
        else:
            base_params = {}

            # Collect hyperparameter ranges from method and feature selection stages.
            for name, converter in zip([self.args["method"], self.args["feature_selection"]],
                                       [MainApplication.get_model_type_for_method_name,
                                        MainApplication.get_feature_selection_type_for_name]):
                cls = converter(name)
                if cls == SwitchModel:
                    # Substitute for base model hyperparams when SwitchModel is selected.
                    class_names = [self.args["base_model"]]
                    classes = [MainApplication.get_model_type_for_method_name(class_names[0])]
                elif cls == FeatureSelectionPipeline:
                    class_names = self.args["feature_selection"].split(",")
                    classes = FeatureSelectionPipeline.get_feature_selection_stages(class_names)
                else:
                    classes, class_names = [cls], [name]

                class_params = {}
                for cls, class_name in zip(classes, class_names):
                    if cls is not None:
                        if hasattr(cls, "get_hyperparameter_ranges"):
                            class_params = cls.get_hyperparameter_ranges()
                        else:
                            print("WARN: Unable to retrieve hyperparameter ranges for provided name "
                                  "[", class_name, "].",
                                  file=sys.stderr)
                    else:
                        print("WARN: Unable to retrieve class for provided name [", class_name, "].",
                              file=sys.stderr)
                base_params.update(class_params)

        hyper_params.update(base_params)
        return hyper_params

    @staticmethod
    def get_model_type_for_method_name(method):
        from precision_fda_brain_biomarker.models.baselines.base_model import BaseModel
        return MainApplication.get_type_for_name_in_class_module(method, BaseModel, ensemble_class=Ensemble)

    @staticmethod
    def get_feature_selection_type_for_name(name):
        from precision_fda_brain_biomarker.models.feature_selection.pca import PCA
        return MainApplication.get_type_for_name_in_class_module(name, PCA, ensemble_class=FeatureSelectionPipeline)

    @staticmethod
    def get_type_for_name_in_class_module(method, module_clazz, ensemble_class):
        is_ensemble = "," in method
        if is_ensemble:
            return ensemble_class
        else:
            baseline_package_path = os.path.dirname(inspect.getfile(module_clazz))

            for module_path in glob.glob(baseline_package_path + "/*.py"):
                modname = os.path.basename(module_path)[:-3]
                fully_qualified_name = module_clazz.__module__
                fully_qualified_name = fully_qualified_name[:fully_qualified_name.rfind(".")] + "." + modname
                mod = importlib.import_module(fully_qualified_name)
                if hasattr(mod, method):
                    cls = getattr(mod, method)
                    return cls
            return None

    def get_model_for_method_name(self, model_params):
        cls = MainApplication.get_model_type_for_method_name(self.args["method"])
        if cls is not None:
            if cls == Ensemble:
                paths = self.args["method"].split(",")
                is_topk_ensemble = len(paths) == 2 and paths[1].isdigit()
                if is_topk_ensemble:
                    paths = Ensemble.get_topk_models_in_path(paths[0],
                                                             outer_index=self.args["split_index_outer"],
                                                             inner_index=self.args["split_index_inner"],
                                                             subchallenge=self.args["dataset"],
                                                             k=int(paths[1]))
                estimators, weights = Ensemble.get_estimators_from_paths(paths)
                model_params["estimators"] = estimators
                model_params["weights"] = weights
                instance = Ensemble(estimators=estimators, weights=weights)
                # Estimators_ must be set to enable predicting without re-fitting.
                instance.estimators_ = [estimator[1] for estimator in estimators]
                available_model_params = {k: model_params[k] if k in model_params else instance.get_params()[k]
                                          for k in instance.get_params().keys()}
                instance = instance.set_params(**available_model_params)
            else:
                # from sklearn.utils.estimator_checks import check_estimator
                if cls == SwitchModel:
                    instance = cls(base_model=model_params["base_model"])
                else:
                    instance = cls()
                # check_estimator(instance)
                available_model_params = {k: model_params[k] if k in model_params else instance.get_params()[k]
                                          for k in instance.get_params().keys()}
                instance = instance.set_params(**available_model_params)
            return instance
        else:
            return None

    def get_feature_selection_for_name(self, selection_params):
        cls = MainApplication.get_feature_selection_type_for_name(self.args["feature_selection"])
        if cls is not None:
            if cls == FeatureSelectionPipeline:
                feature_selection_stage_names = self.args["feature_selection"].split(",")
                instances = FeatureSelectionPipeline.get_feature_selection_stages(feature_selection_stage_names,
                                                                                  selection_params)
                steps = [("feature_select_{}".format(i), ins) for i, ins in zip(range(len(instances)), instances)]
                pipes = FeatureSelectionPipeline(steps=steps)
                return pipes
            else:
                instances = cls()
                available_model_params = {k: selection_params[k] if k in selection_params else instances.get_params()[k]
                                          for k in instances.get_params().keys()}
                instances = instances.set_params(**available_model_params)
                return instances
        else:
            return None

    def get_model(self, val_generator=None, val_steps=None):
        with_tensorboard = self.args["with_tensorboard"]
        output_directory = self.args["output_directory"]
        n_jobs = int(np.rint(self.args["n_jobs"]))
        num_epochs = int(np.rint(self.args["num_epochs"]))
        learning_rate = float(self.args["learning_rate"])
        l1_weight = float(self.args["l1_weight"])
        l2_weight = float(self.args["l2_weight"])
        batch_size = int(np.rint(self.args["batch_size"]))
        early_stopping_patience = int(np.rint(self.args["early_stopping_patience"]))
        num_layers = int(np.rint(self.args["num_layers"]))
        num_units = int(np.rint(self.args["num_units"]))
        svm_c = float(self.args["svm_c"])
        dropout = float(self.args["dropout"])
        n_estimators = int(np.rint(self.args["n_estimators"]))
        max_depth = int(np.rint(self.args["max_depth"]))
        best_model_path = self.get_model_path()
        feature_selection = self.args["feature_selection"]
        seed = int(np.rint(self.args["seed"]))
        kernel = self.args["kernel"]
        probability = self.args["probability"]
        criterion = self.args["criterion"]
        activation = self.args["activation"]
        base_model = self.args["base_model"]
        voting = self.args["voting"]
        if "feature_names" in self.args:
            feature_names = self.args["feature_names"]
        else:
            feature_names = []
        min_split_loss = float(self.args["min_split_loss"])
        subsample = float(self.args["subsample"])

        input_dim = self.input_dim if hasattr(self, "input_dim") else 0
        output_dim = self.output_dim if hasattr(self, "output_dim") else 0

        model_params = {
            "output_directory": output_directory,
            "early_stopping_patience": early_stopping_patience,
            "num_layers": num_layers,
            "num_units": num_units,
            "p_dropout": dropout,
            "input_dim": input_dim,
            "feature_names": feature_names,
            "output_dim": output_dim,
            "batch_size": batch_size,
            "best_model_path": best_model_path,
            "l1_weight": l1_weight,
            "l2_weight": l2_weight,
            "learning_rate": learning_rate,
            "activation": activation,
            "with_tensorboard": with_tensorboard,
            "n_jobs": n_jobs,
            "num_epochs": num_epochs,
            "C": svm_c,
            "kernel": kernel,
            "probability": probability,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "criterion": criterion,
            "min_split_loss": min_split_loss,
            "base_model": base_model,
            "voting": voting,
            "subsample": subsample,
            "seed": seed,
            "random_state": seed,
        }

        if val_generator is None or val_steps is None:
            # Disable Tensorboard if __get_model__ is called without a __val_generator__.
            model_params["with_tensorboard"] = False
        else:
            if with_tensorboard:
                tb_folder = join(self.args["output_directory"], "tensorboard")
                tmp_generator, tmp_steps = val_generator, val_steps
                tb = [MainApplication.build_tensorboard(tmp_generator, tb_folder)]
            else:
                tb = []

            model_params["tensorboard_callback"] = tb

        model = self.get_model_for_method_name(model_params)

        if self.args["load_existing"]:
            print("INFO: Loading existing model from", self.args["load_existing"], file=sys.stderr)
            model = model.load(self.args["load_existing"])

        if feature_selection:
            pca_n_components = self.args["pca_n_components"]

            whiten_pca = self.args["whiten_pca"]
            n_neighbors = self.args['n_neighbors'] if "n_neighbors" in self.args else 10

            feature_selection_params = {
                "n_components": pca_n_components,
                "whiten": whiten_pca,
                "random_state": seed,
                "tier": self.args['tier'],
                "feature_names": feature_names,
                "n_neighbors": n_neighbors
            }

            feature_selection = self.get_feature_selection_for_name(feature_selection_params)

            if self.args["load_existing"]:
                feature_selection_path = self.get_feature_selection_path()
                feature_selection_load_path = join(os.path.dirname(self.args["load_existing"]),
                                                   os.path.basename(feature_selection_path))
                feature_selection = feature_selection.load(feature_selection_load_path)

            model = Pipeline([('feature_selection', feature_selection),
                              ('model', model)])
        return model

    @time_function("train_model")
    def train_model(self, train_generator, train_steps, val_generator, val_steps):
        print("INFO: Started training model.", file=sys.stderr)

        assert train_steps > 0 or not self.args["do_train"], \
            "You specified a batch_size that is bigger than the size of the train set."
        assert val_steps > 0 or not self.args["do_train"], \
            "You specified a batch_size that is bigger than the size of the validation set."

        self.save_config()
        model_path = self.get_model_path()

        model = self.get_model(val_generator=val_generator, val_steps=val_steps)

        if self.args["do_train"]:
            if isinstance(model, Ensemble):
                x, y = np.array(self.validation_set[0]), self.validation_set[1]
            else:
                x, y = np.array(self.training_set[0]), self.training_set[1]
            model.fit(x, y)
            history = None

            if isinstance(model, Pipeline):
                feature_selection_path = self.get_feature_selection_path()
                model.steps[0][1].save(feature_selection_path)
                model.steps[1][1].save(model_path)
            else:
                model.save(model_path)
            print("INFO: Saved model to", model_path, file=sys.stderr)
        else:
            history = {
                "val_acc": [],
                "val_loss": [],
                "val_combined_loss": [],
                "acc": [],
                "loss": [],
                "combined_loss": []
            }
        return model, history

    def evaluate_model(self, model, test_generator, test_steps, with_print=True, set_name="test", threshold=None):
        if with_print:
            print("INFO: Started evaluation.", file=sys.stderr)

        if test_steps == 0:
            return None

        scores = ModelEvaluation.evaluate(model, test_generator, test_steps, set_name,
                                          threshold=threshold, with_print=with_print)
        return scores

    def save_predictions(self, model, threshold=None):
        print("INFO: Saving model predictions.", file=sys.stderr)

        fraction_of_data_set = clip_percentage(self.args["fraction_of_data_set"])

        generators = [self.make_train_generator, self.make_validation_generator, self.make_test_generator]
        generator_names = ["train", "val", "test"]
        for generator_fun, generator_name in zip(generators, generator_names):
            generator, steps = generator_fun(randomise=False)
            steps = int(np.rint(steps * fraction_of_data_set))

            if steps == 0:
                continue

            predictions = []
            for step in range(steps):
                x, y = next(generator)
                last_id = get_last_row_id()
                if hasattr(model, "predict_proba"):
                    y_pred = model.predict_proba(x)[:, 1]
                else:
                    y_pred = model.predict(x)
                y_pred = np.squeeze(y_pred)
                if y_pred.size == 1:
                    y_pred = [y_pred]

                for current_id, current_y_pred in zip(last_id, y_pred):
                    predictions.append([current_id, current_y_pred])
            row_ids = np.hstack(map(lambda x: x[0], predictions))
            outputs = np.hstack(map(lambda x: x[1], predictions))
            file_path = self.get_prediction_path(generator_name)

            num_predictions = 1 if len(outputs.shape) == 1 else outputs.shape[-1]
            assert num_predictions == 1, "Must have exactly one output prediction for binary classification."

            columns = ["SURVIVAL_STATUS"]

            df = pd.DataFrame(outputs, columns=columns, index=row_ids)
            df.index.name = "PATIENTID"
            df.to_csv(file_path, sep="\t")
            print("INFO: Saved raw model predictions to", file_path, file=sys.stderr)

            if threshold is not None:
                thresholded_file_path = self.get_thresholded_prediction_path(generator_name)
                df = pd.DataFrame((outputs > threshold).astype(int), columns=columns, index=row_ids)
                df.index.name = "PATIENTID"
                df.to_csv(thresholded_file_path, sep="\t")
                print("INFO: Saved thresholded model predictions to", thresholded_file_path, file=sys.stderr)

    def get_feature_importances(self, model):
        from cxplain import ZeroMasking
        from cxplain.util.test_util import TestUtil
        from cxplain.backend.masking.masking_util import MaskingUtil
        from cxplain.backend.causal_loss import calculate_delta_errors
        from cxplain.backend.numpy_math_interface import NumpyInterface

        x, y = self.test_set[0], self.test_set[1]
        masking = ZeroMasking()

        if isinstance(model, Pipeline):
            transform = model.steps[0][1]
            x = transform.transform(np.array(x))
            model = model.steps[1][1]

        num_features = np.array(x).shape[-1]
        max_num_feature_groups = int(np.rint(self.args["max_num_feature_groups"]))

        if max_num_feature_groups >= num_features:
            _, y_pred, all_y_pred_imputed = masking.get_predictions_after_masking(model,
                                                                                  x, y,
                                                                                  batch_size=len(x),
                                                                                  downsample_factors=(1,),
                                                                                  flatten=True)
            auxiliary_outputs = y_pred
            all_but_one_auxiliary_outputs = all_y_pred_imputed
            all_but_one_auxiliary_outputs = TestUtil.split_auxiliary_outputs_on_feature_dim(
                all_but_one_auxiliary_outputs
            )

            delta_errors = calculate_delta_errors(np.expand_dims(y, axis=-1),
                                                  auxiliary_outputs,
                                                  all_but_one_auxiliary_outputs,
                                                  NumpyInterface.binary_crossentropy,
                                                  math_ops=NumpyInterface)

            group_importances = np.mean(delta_errors, axis=0)
            feature_groups = np.expand_dims(np.arange(x[0].shape[-1]), axis=-1).tolist()
        else:
            class ModelWrapper(object):
                def __init__(self, wrapped_model, real_data, dummy_to_real_mapping):
                    self.wrapped_model = wrapped_model
                    self.real_data = real_data
                    self.dummy_to_real_mapping = dummy_to_real_mapping

                def map_from_dummy(self, dummy):
                    mask = np.ones(np.array(self.real_data).shape)
                    for i, row in enumerate(dummy):
                        for j, group in enumerate(self.dummy_to_real_mapping):
                            if row[j] == 0.:
                                mask[i, group] = 0
                    return self.real_data * mask

                def predict(self, x):
                    x = self.map_from_dummy(x)
                    y = MaskingUtil.predict_proxy(model, x)
                    if len(y.shape) == 1:
                        y = np.expand_dims(y, axis=-1)
                    return y

            num_groups = 1
            feature_groups = [np.random.permutation(np.arange(x[0].shape[-1]))]
            group_importances = [1.0]
            while num_groups < max_num_feature_groups:
                num_groups += 1

                # Recurse into largest relative importance group.
                did_find_splittable = False
                highest_importances = np.argsort(group_importances)[::-1]
                for highest_importance in highest_importances:
                    if len(feature_groups[highest_importance]) != 1:
                        did_find_splittable = True
                        break
                    else:
                        continue

                if not did_find_splittable:
                    # max_num_groups > len(features) - abort.
                    break

                rest = len(feature_groups[highest_importance]) % 2
                if rest != 0:
                    carry = feature_groups[highest_importance][-rest:].tolist()
                    feature_groups[highest_importance] = feature_groups[highest_importance][:-rest]
                else:
                    carry = []
                feature_groups[highest_importance] = np.split(feature_groups[highest_importance], 2)
                feature_groups[highest_importance][0] = np.array(feature_groups[highest_importance][0].tolist() + carry)
                recombined = feature_groups[:highest_importance] + \
                             feature_groups[highest_importance] + \
                             feature_groups[highest_importance + 1:]
                assert 0 not in map(len, recombined)
                feature_groups = recombined
                wrapped_model = ModelWrapper(model, x, feature_groups)

                dummy_data = np.ones((len(x), num_groups))
                _, y_pred, all_y_pred_imputed = masking.get_predictions_after_masking(wrapped_model,
                                                                                      dummy_data, y,
                                                                                      batch_size=len(x),
                                                                                      downsample_factors=(1,),
                                                                                      flatten=True)
                auxiliary_outputs = y_pred
                all_but_one_auxiliary_outputs = all_y_pred_imputed
                all_but_one_auxiliary_outputs = TestUtil.split_auxiliary_outputs_on_feature_dim(
                    all_but_one_auxiliary_outputs
                )

                delta_errors = calculate_delta_errors(np.expand_dims(y, axis=-1),
                                                      auxiliary_outputs,
                                                      all_but_one_auxiliary_outputs,
                                                      NumpyInterface.binary_crossentropy,
                                                      math_ops=NumpyInterface)

                group_importances = np.mean(delta_errors, axis=0)
        return feature_groups, group_importances

    def save_attributions(self, model):
        from precision_fda_brain_biomarker.models.baselines.logistic_regression import LogisticRegression
        from precision_fda_brain_biomarker.models.feature_selection.tier_select import TierSelect

        print("INFO: Saving model attributions.", file=sys.stderr)
        feature_groups, group_importances = self.get_feature_importances(model)

        def get_feature_names():
            if isinstance(model, Pipeline) and isinstance(model.steps[0][1], TierSelect):
                kept_indices = model.steps[0][1].kept_indices
                feature_names = [self.feature_names[i] for i in kept_indices]
            else:
                feature_names = self.feature_names
            return feature_names

        prefixes = [""]
        importances = [group_importances]
        if isinstance(model, Pipeline) and isinstance(model.steps[1][1], LogisticRegression):
            native_importances = np.abs(model.steps[1][1].coef_)[0]
            prefixes += ["weights."]
            importances += [native_importances]

        for prefix, group_importances in zip(prefixes, importances):
            sorted_idx = np.argsort(group_importances)[::-1]
            feature_groups_transformed = [feature_groups[idx] for idx in sorted_idx]
            feature_groups_transformed = list(map(lambda group:
                                                  list(map(lambda feature_idx: get_feature_names()[feature_idx],
                                                           sorted(group))),
                                                  feature_groups_transformed))
            group_importances = [group_importances[idx] for idx in sorted_idx]

            file_path = self.get_attribution_path(prefix=prefix)

            columns = ["RELATIVE_IMPORTANCE", "NUM_FEATURES", "FEATURE_NAMES"]

            df = pd.DataFrame(zip(group_importances,
                                  map(len, feature_groups_transformed),
                                  feature_groups_transformed), columns=columns,
                              index=np.arange(len(group_importances)))
            df.index.name = "FEATURE_GROUP_RANK"
            df.to_csv(file_path, sep="\t")
            print("INFO: Saved model attributions to", file_path, file=sys.stderr)

    @staticmethod
    def build_tensorboard(tmp_generator, tb_folder):
        for a_file in os.listdir(tb_folder):
            file_path = join(tb_folder, a_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e, file=sys.stderr)

        tb = TensorBoard(tb_folder, write_graph=False, histogram_freq=1, write_grads=True, write_images=False)
        x, y = next(tmp_generator)

        tb.validation_data = x
        tb.validation_data[1] = np.expand_dims(tb.validation_data[1], axis=-1)
        if isinstance(y, list):
            num_targets = len(y)
            tb.validation_data += [y[0]] + y[1:]
        else:
            tb.validation_data += [y]
            num_targets = 1

        tb.validation_data += [np.ones(x[0].shape[0])] * num_targets + [0.0]
        return tb


if __name__ == '__main__':
    app = MainApplication(parse_parameters())
    app.run()
