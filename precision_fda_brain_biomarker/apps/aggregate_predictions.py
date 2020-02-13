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
import numpy as np
import pandas as pd
from collections import OrderedDict


class AggregatePredictionsApplication(object):
    def __init__(self):
        pass

    def get_target_file(self, file_path):
        from pandas import read_csv

        # The CSV file must be placed in this file's directory.
        rows = read_csv(file_path, sep="\t")
        names = rows.columns[1:]
        values = rows.values
        return names, OrderedDict(values)

    def run(self, search_directory, target_file_path):
        all_patient_dicts = []
        for folder in sorted(glob.glob(os.path.join(search_directory, "outer_*"))):
            if os.path.isdir(folder):
                for subfolder in sorted(glob.glob(os.path.join(folder, "inner_*"))):
                    if os.path.isdir(subfolder):
                        target_file = os.path.join(subfolder, "test_predictions.thresholded.tsv")
                        names, patient_dict = self.get_target_file(target_file)
                        all_patient_dicts.append(patient_dict)

        all_patients, all_preds = [], []
        for patient in all_patient_dicts[0].keys():
            values = []
            for d in all_patient_dicts:
                values += [d[patient]]
            all_patients.append(patient)
            mean_pred = 1 if np.mean(values) > 0.5 else 0
            all_preds.append(mean_pred)
        columns = ["SURVIVAL_STATUS"]
        df = pd.DataFrame(all_preds, columns=columns, index=all_patients)
        df.index.name = "PATIENTID"
        df.to_csv(target_file_path, sep="\t")


if __name__ == "__main__":
    app = AggregatePredictionsApplication()
    app.run(sys.argv[1], sys.argv[2])
