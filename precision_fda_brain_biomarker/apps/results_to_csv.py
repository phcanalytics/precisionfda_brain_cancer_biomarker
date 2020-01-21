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
import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd


class ResultsToCSVApplication(object):
    def __init__(self):
        pass

    @staticmethod
    def get_subchallenge_for_file_name(file_name):
        for subchallenge in ["sc1", "sc2", "sc3"]:
            if subchallenge in file_name:
                return subchallenge
        return None

    def run(self, search_directory, output_directory, pickle_file_name="eval_score.pickle"):
        csv_rows = {
            "sc1": [],
            "sc2": [],
            "sc3": [],
        }
        col_names = [
            "auroc", "auroc_std",
            "auprc", "auprc_std",
            "f1", "f1_std",
            "sensitivity", "sensitivity_std",
            "specificity", "specificity_std",
            "ppv", "ppv_std",
            "npv", "npv_std",
            "accuracy", "accuracy_std",
            "tn", "tn_std",
            "fn", "fn_std",
            "tp", "tp_std",
            "fp", "fp_std"
        ]
        files = glob.glob(search_directory + "/*")
        for file_candidate in files:
            if os.path.isdir(file_candidate):
                pickle_file = os.path.join(file_candidate, pickle_file_name)
                if os.path.isfile(pickle_file):
                    with open(pickle_file, "rb") as fp:
                        results_dict = pickle.load(fp)
                        base_file_name = os.path.basename(file_candidate)
                        subchallenge = ResultsToCSVApplication.get_subchallenge_for_file_name(
                            base_file_name
                        )
                        if subchallenge is None:
                            continue
                        csv_rows[subchallenge].append([base_file_name] + [results_dict[col_name]
                                                                          for col_name in col_names])
                else:
                    print("WARN:", pickle_file, "was not present.", file=sys.stderr)

        output_file_name = "results_{}.csv"
        for subchallenge in ["sc1", "sc2", "sc3"]:
            df = pd.DataFrame(csv_rows[subchallenge],
                              columns=["name"] + col_names,
                              index=np.arange(len(csv_rows[subchallenge])))
            df.index.name = "rowid"
            output_csv_path = os.path.join(output_directory, output_file_name.format(subchallenge))
            df.to_csv(output_csv_path)
            print("INFO: Saved result CSV to", output_csv_path, ".", file=sys.stderr)


if __name__ == "__main__":
    app = ResultsToCSVApplication()
    app.run(sys.argv[1], sys.argv[2])
