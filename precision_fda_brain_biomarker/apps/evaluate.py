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
import six
import sys
import time
import shutil
import numpy as np
from os.path import join
import keras.backend as K
from datetime import datetime
from precision_fda_brain_biomarker.apps.parameters import parse_parameters

if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


class EvaluationApplication(object):
    def __init__(self, args, do_setup=True, do_print=True):
        self.args = args

        if do_print:
            print(" ".join(sys.argv), file=sys.stderr)
            print("INFO: Args are:", self.args, file=sys.stderr)
            print("INFO: Running at", str(datetime.now()), file=sys.stderr)

        self.init_hyperopt_params()
        if do_setup:
            self.setup()

    def init_hyperopt_params(self):
        self.best_score_index = 0
        self.best_score = np.finfo(float).min
        self.best_params = ""
        self.best_model_name = "best_model.npy"

    def setup(self):
        seed = int(np.rint(self.args["seed"]))
        print("INFO: Seed is", seed, file=sys.stderr)

        os.environ['PYTHONHASHSEED'] = '0'

        import random as rn
        rn.seed(seed)
        np.random.seed(seed)

        import tensorflow as tf
        tf.set_random_seed(seed)

        #  Configure tensorflow not to use the entirety of GPU memory at start.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

    def train_model(self, train_generator, train_steps, val_generator, val_steps):
        return None, None

    def evaluate_model(self, model, test_generator, test_steps, with_print=True, set_name="", threshold=None):
        return None

    def make_train_generator(self):
        return None, None

    def make_validation_generator(self):
        return None, None

    def make_test_generator(self):
        return None, None

    def get_hyperopt_parameters(self):
        return {}

    def get_model_path(self):
        return ""

    def get_feature_selection_path(self):
        return ""

    def get_prediction_path(self, set_name):
        return ""

    def get_attribution_path(self):
        return ""

    def get_thresholded_prediction_path(self, set_name):
        return ""

    def save_predictions(self, model, threshold=None):
        return

    def save_attributions(self, model):
        return

    def load_data(self):
        return

    def run(self):
        evaluate_against = self.args["evaluate_against"]
        num_splits_outer = self.args["num_splits_outer"]
        if num_splits_outer < 2:
            raise AssertionError("__num_splits_outer__ must be at least 1.")
        if num_splits_outer == 2 or (evaluate_against == "val" and not self.args["do_hyperopt"]):
            return_values = self._run()
        else:
            split_index_outer_before = self.args["split_index_outer"]
            output_directory_before = self.args["output_directory"]

            eval_scores, test_scores = [], []
            for split_index_outer in range(num_splits_outer):
                self.args["split_index_outer"] = split_index_outer
                self.args["output_directory"] = os.path.join(output_directory_before,
                                                             "outer_{:d}".format(split_index_outer))
                if not os.path.exists(self.args["output_directory"]):
                    os.mkdir(self.args["output_directory"])
                eval_score, test_score = self._run()
                eval_scores.append(eval_score)
                test_scores.append(test_score)

            self.args["split_index_outer"] = split_index_outer_before
            self.args["output_directory"] = output_directory_before

            self.print_cv_results(test_scores, "OUTER")

            return_values = EvaluationApplication.aggregate_result_dicts(eval_scores, test_scores)
            EvaluationApplication.save_score_dicts(return_values[0], return_values[1], self.args["output_directory"])

        return return_values

    def _run(self):
        if self.args["do_hyperopt"]:
            return self.run_hyperopt()
        else:
            evaluate_against = self.args["evaluate_against"]
            if evaluate_against not in ("test", "val"):
                print("WARN: Specified wrong argument for --evaluate_against. Value was:", evaluate_against,
                      ". Defaulting to: val", file=sys.stderr)
                evaluate_against = "val"
            return self.run_single(evaluate_against=evaluate_against)

    def run_single(self, evaluate_against="test"):
        num_splits_inner = self.args["num_splits_inner"]
        if num_splits_inner < 2:
            raise AssertionError("__num__splits_inner__ must be at least 2.")
        if num_splits_inner == 2:
            return self._run_single(evaluate_against=evaluate_against)

        split_index_inner_before = self.args["split_index_inner"]
        output_directory_before = self.args["output_directory"]

        eval_scores, test_scores = [], []
        for split_index_inner in range(num_splits_inner):
            self.args["split_index_inner"] = split_index_inner
            self.args["output_directory"] = os.path.join(output_directory_before,
                                                         "inner_{:d}".format(split_index_inner))
            if not os.path.exists(self.args["output_directory"]):
                os.mkdir(self.args["output_directory"])
            eval_score, test_score = self._run_single(evaluate_against=evaluate_against)
            eval_scores.append(eval_score)
            test_scores.append(test_score)

        # Reset to initial values.
        self.args["split_index_inner"] = split_index_inner_before
        self.args["output_directory"] = output_directory_before

        self.print_cv_results(test_scores, "INNER_{:d}".format(self.args["split_index_outer"]))

        eval_score, test_score = EvaluationApplication.aggregate_result_dicts(eval_scores, test_scores)
        EvaluationApplication.save_score_dicts(eval_score, test_score, self.args["output_directory"])
        return eval_score, test_score

    @staticmethod
    def aggregate_result_dicts(eval_scores, test_scores):
        # Aggregate results across inner folds.
        output_dicts = [{}, {}]
        for dict_list, output_dict in zip([eval_scores, test_scores], output_dicts):
            for key in dict_list[0].keys():
                all_values = []
                for scores in dict_list:
                    all_values.append(scores[key])
                if key == "model_path" or key == "feature_selection_path":
                    output_dict[key] = ",".join(all_values)
                else:
                    output_dict[key] = np.mean(all_values)
                    output_dict[key + "_std"] = np.std(all_values)
                    output_dict[key + "_results"] = all_values
        return output_dicts

    @staticmethod
    def print_cv_results(eval_dicts, name):
        print("INFO: {} cross validation results (N={:d}) are:".format(name, len(eval_dicts)), file=sys.stderr)
        for key in eval_dicts[0].keys():
            if "_results" in key or "_std" in key or "_path" in key:
                continue
            try:
                values = list(map(lambda x: x[key], eval_dicts))
                print(key, "=", np.mean(values), "+-", np.std(values),
                      "CI=(", np.percentile(values, 2.5), ",", np.percentile(values, 97.5), "),",
                      "median=", np.median(values),
                      "min=", np.min(values),
                      "max=", np.max(values),
                      file=sys.stderr)
            except:
                print("ERROR: Could not get key", key, "for all score dicts.", file=sys.stderr)

    def _run_single(self, evaluate_against="test"):
        print("INFO: Run with args:",
              {k: v for k, v in self.args.items() if k != "feature_names"},  # Skip feature names in output.
              file=sys.stderr)

        self.load_data()

        save_predictions = self.args["save_predictions"]
        save_attributions = self.args["save_attributions"]

        train_generator, train_steps = self.make_train_generator()
        val_generator, val_steps = self.make_validation_generator()
        test_generator, test_steps = self.make_test_generator()

        print("INFO: Built generators with", train_steps,
              "training steps, ", val_steps,
              "validation steps and", test_steps, "test steps.",
              file=sys.stderr)

        model, history = self.train_model(train_generator,
                                          train_steps,
                                          val_generator,
                                          val_steps)

        loss_file_path = join(self.args["output_directory"], "losses.pickle")
        print("INFO: Saving loss history to", loss_file_path, file=sys.stderr)
        pickle.dump(self.args, open(loss_file_path, "wb"), pickle.HIGHEST_PROTOCOL)

        threshold = self.args["threshold"]
        if self.args["do_evaluate"]:
            if evaluate_against == "test":
                thres_generator, thres_steps = val_generator, val_steps
                eval_generator, eval_steps = test_generator, test_steps
            else:
                thres_generator, thres_steps = train_generator, train_steps
                eval_generator, eval_steps = val_generator, val_steps

            thres_score = self.evaluate_model(model, thres_generator, thres_steps,
                                              with_print=False, set_name="threshold")
            if threshold is None:
                # Get threshold from train or validation set.
                threshold = thres_score["threshold"]

            eval_score = self.evaluate_model(model, eval_generator, eval_steps,
                                             set_name=evaluate_against, threshold=threshold)
        else:
            eval_score = None
            thres_score = None

        if save_predictions:
            self.save_predictions(model, threshold=threshold)

        if save_attributions:
            self.save_attributions(model)

        if eval_score is None:
            test_score = self.evaluate_model(model, test_generator, test_steps,
                                             with_print=evaluate_against == "val", set_name="test")
            eval_score = test_score
        else:
            test_score = eval_score
            eval_score = thres_score

        eval_score["model_path"] = test_score["model_path"] = self.get_model_path()

        if self.args["feature_selection"] != "":
            eval_score["feature_selection_path"] = test_score["feature_selection_path"] =\
                self.get_feature_selection_path()

        EvaluationApplication.save_score_dicts(eval_score, test_score, self.args["output_directory"])

        return eval_score, test_score

    @staticmethod
    def save_score_dicts(eval_score, test_score, output_directory):
        eval_score_path = join(output_directory, "eval_score.pickle")
        with open(eval_score_path, "wb") as fp:
            pickle.dump(eval_score, fp, pickle.HIGHEST_PROTOCOL)
        test_score_path = join(output_directory, "test_score.pickle")
        with open(test_score_path, "wb") as fp:
            pickle.dump(test_score, fp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def get_random_hyperopt_parameters(initial_args, hyperopt_parameters, hyperopt_index):
        new_params = dict(initial_args)
        for k, v in hyperopt_parameters.items():
            if isinstance(v, list):
                min_val, max_val = v
                new_params[k] = np.random.uniform(min_val, max_val)
            elif isinstance(v, tuple):
                choice = np.random.choice(v)
                if hasattr(choice, "item"):
                    choice = choice.item()
                new_params[k] = choice
        return new_params

    @staticmethod
    def get_next_hyperopt_choice(initial_args, hyperopt_parameters, state=None):
        if state is None:
            state = [0 for _ in range(len(hyperopt_parameters))]

        new_params = dict(initial_args)
        for k, state_i in zip(sorted(hyperopt_parameters.keys()), state):
            v = hyperopt_parameters[k]
            if isinstance(v, tuple):
                choice = v[state_i]
                if hasattr(choice, "item"):
                    choice = choice.item()
                new_params[k] = choice
            else:
                raise AssertionError("Only hyperopt_parameters with finite numbers of permutations can be used"
                                     "with __get_next_hyperopt_choice_generator__.")

        for i, key in enumerate(sorted(hyperopt_parameters.keys())):
            state[i] += 1
            if state[i] % len(hyperopt_parameters[key]) == 0:
                state[i] = 0
            else:
                break

        return new_params, state

    @staticmethod
    def print_run_results(args, hyperopt_parameters, run_index, score, run_time):
        message = "Hyperopt run [" + str(run_index) + "]:"
        best_params_message = ""
        for k in hyperopt_parameters:
            if isinstance(args[k], six.string_types):
                best_params_message += k + "=" + "{:s}".format(args[k]) + ", "
            else:
                best_params_message += k + "=" + "{:.4f}".format(args[k]) + ", "
        best_params_message += "time={:.4f},".format(run_time) + "score={:.4f}".format(score)
        print("INFO:", message, best_params_message, file=sys.stderr)
        return best_params_message

    @staticmethod
    def calculate_num_hyperparameter_permutations(hyperopt_parameters):
        num_permutations = 1
        for param_range in hyperopt_parameters.values():
            if isinstance(param_range, list):
                return float("inf")
            else:
                num_permutations *= len(param_range)
        return num_permutations

    def run_hyperopt(self):
        num_hyperopt_runs = int(np.rint(self.args["num_hyperopt_runs"]))
        hyperopt_offset = int(np.rint(self.args["hyperopt_offset"]))

        self.init_hyperopt_params()
        initial_args = dict(self.args)
        hyperopt_parameters = self.get_hyperopt_parameters()
        print("INFO: Performing hyperparameter optimisation with parameters:", hyperopt_parameters, file=sys.stderr)

        state = None
        max_permutations = EvaluationApplication.calculate_num_hyperparameter_permutations(hyperopt_parameters)
        max_num_hyperopt_runs = min(max_permutations, num_hyperopt_runs)  # Do not perform more runs than necessary.
        enumerate_all_permutations = max_permutations <= num_hyperopt_runs

        job_ids, score_dicts, test_score_dicts, eval_dicts = [], [], [], []
        for i in range(max_num_hyperopt_runs):
            run_start_time = time.time()

            hyperopt_parameters = self.get_hyperopt_parameters()
            if enumerate_all_permutations:
                self.args, state = EvaluationApplication.get_next_hyperopt_choice(initial_args,
                                                                                  hyperopt_parameters,
                                                                                  state=state)
            else:
                self.args = EvaluationApplication.get_random_hyperopt_parameters(initial_args,
                                                                                 hyperopt_parameters,
                                                                                 hyperopt_index=i)

            if i < hyperopt_offset:
                # Skip until we reached the hyperopt offset.
                continue

            resample_with_replacement = self.args["resample_with_replacement"]
            if resample_with_replacement:
                self.load_data()

            eval_set = "test"
            score_dict, test_dict = self.run_single(evaluate_against=eval_set)
            score_dicts.append(score_dict)
            test_score_dicts.append(test_dict)

            if self.args["hyperopt_against_eval_set"]:
                eval_dict = test_dict
            else:
                eval_dict = score_dict
            score = eval_dict[self.args["hyperopt_metric"]]
            eval_dicts.append(eval_dict)

            run_time = time.time() - run_start_time

            # This is necessary to avoid memory leaks when repeatedly building new models.
            K.clear_session()

            best_params_message = EvaluationApplication.print_run_results(self.args,
                                                                          hyperopt_parameters,
                                                                          i, score, run_time)
            if score > self.best_score and self.args["do_train"]:
                self.best_score_index = i
                self.best_score = score
                self.best_params = best_params_message
                model_path = score_dict["model_path"]

                if self.args["feature_selection"] != "":
                    feature_selection_path = score_dict["feature_selection_path"]
                else:
                    feature_selection_path = None

                if "," in model_path:
                    model_path = model_path.split(",")
                else:
                    model_path = [model_path]

                for model_idx, cur_model_path in enumerate(model_path):
                    model_dir = os.path.dirname(cur_model_path)
                    model_name = os.path.basename(cur_model_path)

                    def copy_file(source_file_name, target_file_name):
                        source_file_path = join(model_dir, source_file_name)
                        if os.path.isfile(source_file_path):
                            target_file_path = join(model_dir, target_file_name)
                            shutil.copy(source_file_path, target_file_path)

                    files_to_copy = [
                        "config.json", "switch_config.json", "xgboost_config.json",
                        "program_args.pickle", model_name, "eval_score.pickle",
                        "test_score.pickle",
                        "train_predictions.tsv", "train_predictions.thresholded.tsv",
                        "val_predictions.tsv", "val_predictions.thresholded.tsv",
                        "test_predictions.tsv", "test_predictions.thresholded.tsv",
                    ]
                    model_filename, model_file_extension = os.path.splitext(model_name)
                    for model_idx in range(16):
                        files_to_copy.append(model_filename + str(model_idx) + model_file_extension)
                    # NOTE: The current model files will be overwritten at the next hyperoptimisation iteration.
                    #       We retain the best model's config and parameters only.
                    for file_to_copy in files_to_copy:
                        copy_file(file_to_copy, "best_" + file_to_copy)
                    if feature_selection_path is not None:
                        feature_selection_name = os.path.basename(feature_selection_path)
                        copy_file(feature_selection_name, "best_" + feature_selection_name)

        print("INFO: Best[", self.best_score_index, "] config was", self.best_params, file=sys.stderr)
        self.args = initial_args

        print("INFO: Best hyperopt score:", test_score_dicts[self.best_score_index], file=sys.stderr)

        # Override last experiment results with the overall best experiment results.
        model_path = score_dicts[self.best_score_index]["model_path"]

        if "," in model_path:
            model_path = model_path.split(",")
        else:
            model_path = [model_path]

        best_model_has_feature_selection = "feature_selection_path" in score_dicts[self.best_score_index]
        if best_model_has_feature_selection:
            feature_selection_paths = score_dicts[self.best_score_index]["feature_selection_path"].split(",")
        else:
            feature_selection_paths = [None]*len(model_path)

        for model_idx, (cur_model_path, feature_selection_path) in enumerate(zip(model_path, feature_selection_paths)):
            model_dir = os.path.dirname(cur_model_path)
            model_name = os.path.basename(cur_model_path)

            def copy_and_remove_source_file(source_file_name, target_file_name):
                source_file_path = join(model_dir, source_file_name)
                if os.path.isfile(source_file_path):
                    target_file_path = join(model_dir, target_file_name)
                    shutil.copy(source_file_path, target_file_path)
                    os.remove(source_file_path)

            files_to_copy = [
                "config.json", "switch_config.json", "xgboost_config.json",
                "program_args.pickle", model_name, "eval_score.pickle",
                "test_score.pickle",
                "train_predictions.tsv", "train_predictions.thresholded.tsv",
                "val_predictions.tsv", "val_predictions.thresholded.tsv",
                "test_predictions.tsv", "test_predictions.thresholded.tsv",
            ]
            model_filename, model_file_extension = os.path.splitext(model_name)
            for model_idx in range(16):
                files_to_copy.append(model_filename + str(model_idx) + model_file_extension)
            for file_to_copy in files_to_copy:
                copy_and_remove_source_file("best_" + file_to_copy, file_to_copy)
            if feature_selection_path is not None:
                feature_selection_name = os.path.basename(feature_selection_path)
                copy_and_remove_source_file("best_" + feature_selection_name, feature_selection_name)

        if len(score_dicts) != 0:
            ret_score_dict, ret_test_score_dict = score_dicts[self.best_score_index], \
                                                  test_score_dicts[self.best_score_index]
        else:
            print("ERROR: Unable to select best results for hyperparameter optimisation "
                  "because all results were empty.",
                  file=sys.stderr)
            ret_score_dict, ret_test_score_dict = {}, {}

        # Override last score dicts with best.
        EvaluationApplication.save_score_dicts(ret_score_dict, ret_test_score_dict, self.args["output_directory"])
        return ret_score_dict, ret_test_score_dict


if __name__ == "__main__":
    app = EvaluationApplication(parse_parameters())
    app.run()
