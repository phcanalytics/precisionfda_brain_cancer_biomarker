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

import sys
import numpy as np
from bisect import bisect_right
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve, \
    precision_recall_curve, auc


class ModelEvaluation(object):
    @staticmethod
    def calculate_statistics_binary(y_true, y_pred, set_name, with_print, all_num_tasks=None, threshold=None):
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)

            if all_num_tasks is not None:
                true_idx = np.where(y_true == 1)[0]
                false_idx = np.where(y_true == 0)[0]
                r2_true = pearsonr(np.abs(y_true[true_idx] - y_pred[true_idx]), all_num_tasks[true_idx])[0]
                r2_false = pearsonr(np.abs(y_true[false_idx] - y_pred[false_idx]), all_num_tasks[false_idx])[0]
                if with_print:
                    print("INFO: Num task / outcome correlation PD=", r2_true, ", Case=", r2_false, file=sys.stderr)

                r2_all, p_all = pearsonr(np.abs(y_true - y_pred), all_num_tasks)
                if with_print:
                    print("INFO: Num task / outcome correlation for all r_2=", r2_all, ", p=", p_all, file=sys.stderr)

            if threshold is None:
                # Choose optimal threshold based on closest-to-top-left selection on ROC curve.
                optimal_threshold_idx = np.argmin(np.linalg.norm(np.stack((fpr, tpr)).T -
                                                                 np.repeat([[0., 1.]], fpr.shape[0], axis=0), axis=1))
                threshold = thresholds[optimal_threshold_idx]

            if with_print:
                print("INFO: Using threshold at", threshold, file=sys.stderr)

            y_pred_thresholded = (y_pred > threshold).astype(np.int)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresholded).ravel()
            auc_score = roc_auc_score(y_true, y_pred)

            sens_at_95spec_idx = bisect_right(fpr, 0.05)
            if sens_at_95spec_idx == 0:
                # Report 0.0 if specificity goal can not be met.
                sens_at_95spec = 0.0
            else:
                sens_at_95spec = tpr[sens_at_95spec_idx - 1]

            if auc_score < 0.5:
                if with_print:
                    print("INFO: Inverting AUC.", file=sys.stderr)
                auc_score = 1. - auc_score

            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            auprc_score = auc(recall, precision)

            specificity = float(tn) / (tn + fp) if (tn + fp) != 0 else 0
            sensitivity = float(tp) / (tp + fn) if (tp + fn) != 0 else 0
            ppv = float(tp) / (tp + fp) if (tp + fp) != 0 else 0
            npv = float(tn) / (tn + fn) if (tn + fn) != 0 else 0

            accuracy = accuracy_score(y_true, y_pred_thresholded)
            f1_value = f1_score(y_true, y_pred_thresholded)

            if with_print:
                print("INFO: Performance on", set_name,
                      "AUROC =", auc_score,
                      ", with AUPRC =", auprc_score,
                      ", with accuracy =", accuracy,
                      ", with mean =", np.mean(y_true),
                      ", with f1 =", f1_value,
                      ", with specificity =", specificity,
                      ", with sensitivity =", sensitivity,
                      ", with sens@95spec =", sens_at_95spec,
                      ", with PPV =", ppv,
                      ", with NPV =", npv,
                      ", n =", len(y_true),
                      ", TP =", tp,
                      ", FP =", fp,
                      ", TN =", tn,
                      ", FN =", fn,
                      file=sys.stderr)
            return {
                "auroc": auc_score,
                "auprc": auprc_score,
                "f1": f1_value,
                "sens@95spec": sens_at_95spec,
                "ppv": ppv,
                "npv": npv,
                "specificity": specificity,
                "sensitivity": sensitivity,
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "accuracy": accuracy,
            }
        except:
            if with_print:
                print("WARN: Score calculation failed. Most likely, there was only one class present in y_true.",
                      file=sys.stderr)
            return None

    @staticmethod
    def evaluate(model, generator, num_steps, set_name="Test set",
                 selected_slices=list([-1]), with_print=True, threshold=None):
        all_outputs, all_num_tasks = [], []
        for _ in range(num_steps):
            generator_outputs = next(generator)
            if len(generator_outputs) == 3:
                batch_input, labels_batch, sample_weight = generator_outputs
            else:
                batch_input, labels_batch = generator_outputs

            if hasattr(model, "predict_proba"):
                y_pred = model.predict_proba(batch_input)[:, 1]
            else:
                y_pred = model.predict(batch_input)
            all_outputs.append((y_pred, labels_batch))

        all_num_tasks = None
        for i, selected_slice in enumerate(selected_slices):
            y_pred, y_true = [], []
            output_dim = model.output[i].shape[-1] if hasattr(model, "output") else 1
            for current_step in range(num_steps):
                model_outputs, labels_batch = all_outputs[current_step]

                if isinstance(model_outputs, list):
                    model_outputs = model_outputs[selected_slice]

                if isinstance(labels_batch, list):
                    labels_batch = labels_batch[selected_slice]

                y_pred.append(model_outputs)
                y_true.append(labels_batch)

            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)

            if output_dim != 1:
                y_true = y_true.reshape((-1, output_dim))
                y_pred = y_pred.reshape((-1, output_dim))
            else:
                y_pred = np.squeeze(y_pred)

            if (y_true.ndim == 2 and y_true.shape[-1] == 1) and \
               (y_pred.ndim == 1 and y_pred.shape[0] == y_true.shape[0]):
               y_pred = np.expand_dims(y_pred, axis=-1)

            assert y_true.shape[-1] == y_pred.shape[-1]
            assert y_true.shape[0] == y_pred.shape[0]

            if output_dim == 1:
                score_dict = ModelEvaluation.calculate_statistics_binary(y_true, y_pred,
                                                                         set_name + str(i), with_print,
                                                                         all_num_tasks,
                                                                         threshold=threshold)
            else:
                print("ERROR: The model's output dimension should be 1 for a binary classification task.", file=sys.stderr)

        return score_dict
