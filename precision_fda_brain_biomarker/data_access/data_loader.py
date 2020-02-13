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
import numpy as np
from functools import reduce
from keras.utils import to_categorical
from collections import Counter, OrderedDict
from skmultilearn.model_selection import IterativeStratification
from precision_fda_brain_biomarker.apps.util import time_function


class DataLoader(object):
    DATA_FILE_NAME_TEMPLATE = \
        "sc{SUBCHALLENGE_INDEX}_Phase{PHASE_INDEX}_{SUBCHALLENGE_INDICATOR}_{MODALITY_TYPE}.{EXTENSION}"

    MODALITY_TYPE_FEATURE = "FeatureMatrix"
    MODALITY_TYPE_OUTCOME = "Outcome"
    MODALITY_TYPE_PHENOTYPE = "Phenotype"

    SUBCHALLENGE_MAP = {
        1: "GE",
        2: "CN",
        3: "CN_GE"
    }

    @staticmethod
    def get_dataset_properties(x, y):
        sex_map, race_map, grading_map, cancer_map, map_names = DataLoader.get_phenotype_maps()

        type_maps = sex_map, race_map, grading_map, cancer_map
        assert len(type_maps) == len(map_names), "Must have a name available for each map."

        offset, counters = 0, []
        for type_map in type_maps:
            offset_end = offset + len(type_map) + 1  # +1 for Unknown.
            type_categories = np.argmax(np.array(x)[:, offset:offset_end], axis=-1)
            counters.append(Counter(type_categories))
            offset = offset_end
        return counters, type_maps, map_names

    @staticmethod
    def relative_percentage_dict_to_string(dictionary):
        output_string = ""
        for i, (k, v) in enumerate(dictionary.items()):
            output_string += "{KEY} = {VALUE:.2f}%".format(KEY=k.lower(), VALUE=v)
            if i != len(dictionary) - 1:
                output_string += ", "
        return output_string

    @staticmethod
    def report_data_fold(x, y, set_name):
        if len(x) == 0 or len(y) == 0:
            return

        counters, type_maps, map_names = DataLoader.get_dataset_properties(x, y)

        output_message = ""
        for i, (counter, type_map, map_name) in enumerate(zip(counters, type_maps, map_names)):
            inverted_type_map = {v: k for k, v in type_map.items()}
            total = sum(counter.values())
            sorted_dict = OrderedDict(map(
                lambda counter_entry: (inverted_type_map[counter_entry[0]]
                                       if counter_entry[0] in inverted_type_map else "UNKNOWN",
                                       counter_entry[1] / float(total) * 100),
                counter.items()
            ))
            output_message += "  {NAME}: {DICT}".format(NAME=map_name,
                                                        DICT=DataLoader.relative_percentage_dict_to_string(
                                                             sorted_dict
                                                        ))
            if i != len(counters) - 1:
                output_message += "\n"

        fraction_survival = 1. - np.mean(y)
        print("INFO: Using ", set_name, " set with n = ", len(x),
              " with ", fraction_survival*100, " % overall survival rate.",
              "\n", output_message,
              file=sys.stderr, sep="")

    @staticmethod
    def split_dataset(x, y, p_ids, num_validation_samples, num_test_samples, phenotypes_len, seed,
                      split_index_inner=0, num_splits_inner=2,
                      split_index_outer=0, num_splits_outer=2):
        import random as rn

        synthetic_labels = np.concatenate([y, x[:, :phenotypes_len]], axis=-1)

        test_fraction = num_test_samples / float(len(x))
        if num_splits_outer != 2:
            fold_sizes_1 = None
        else:
            fold_sizes_1 = [test_fraction, 1.-test_fraction]

        rn.seed(seed)
        np.random.seed(seed)
        sss = IterativeStratification(n_splits=num_splits_outer, order=5, sample_distribution_per_fold=fold_sizes_1,
                                      random_state=seed)

        for _ in range(split_index_outer + 1):
            rest_index, test_index = next(sss.split(x, synthetic_labels))
        x_test, y_test, p_test = [x[idx] for idx in test_index], y[test_index], [p_ids[idx] for idx in test_index]
        x_rest, p_rest = [x[idx] for idx in rest_index], [p_ids[idx] for idx in rest_index]

        val_fraction = num_validation_samples / float(len(x_rest))
        if num_splits_inner != 2:
            fold_sizes_2 = None
        else:
            fold_sizes_2 = [val_fraction, 1.-val_fraction]

        rn.seed(seed)
        np.random.seed(seed)
        sss = IterativeStratification(n_splits=num_splits_inner, order=5, sample_distribution_per_fold=fold_sizes_2,
                                      random_state=seed)

        for _ in range(split_index_inner + 1):
            train_index, val_index = next(sss.split(x_rest, synthetic_labels[rest_index]))
        x_train, y_train = [x_rest[idx] for idx in train_index], y[rest_index][train_index]
        p_train = [p_rest[idx] for idx in train_index]
        x_val, y_val = [x_rest[idx] for idx in val_index], y[rest_index][val_index]
        p_val = [p_rest[idx] for idx in val_index]
        return (x_train, y_train, p_train), (x_val, y_val, p_val), (x_test, y_test, p_test)

    @staticmethod
    def get_data_ph1_sc1(args, seed=0, do_resample=False, resample_seed=0):
        return DataLoader.get_data(args, phase_index=1, subchallenge_index=1, seed=seed,
                                   do_resample=do_resample, resample_seed=resample_seed)

    @staticmethod
    def get_data_ph1_sc2(args, seed=0, do_resample=False, resample_seed=0):
        return DataLoader.get_data(args, phase_index=1, subchallenge_index=2, seed=seed,
                                   do_resample=do_resample, resample_seed=resample_seed)

    @staticmethod
    def get_data_ph1_sc3(args, seed=0, do_resample=False, resample_seed=0):
        return DataLoader.get_data(args, phase_index=1, subchallenge_index=3, seed=seed,
                                   do_resample=do_resample, resample_seed=resample_seed)

    @staticmethod
    def get_data_ph2_sc1(args, seed=0, do_resample=False, resample_seed=0):
        return DataLoader.get_data(args, phase_index=2, subchallenge_index=1, seed=seed,
                                   do_resample=do_resample, resample_seed=resample_seed)

    @staticmethod
    def get_data_ph2_sc2(args, seed=0, do_resample=False, resample_seed=0):
        return DataLoader.get_data(args, phase_index=2, subchallenge_index=2, seed=seed,
                                   do_resample=do_resample, resample_seed=resample_seed)

    @staticmethod
    def get_data_ph2_sc3(args, seed=0, do_resample=False, resample_seed=0):
        return DataLoader.get_data(args, phase_index=2, subchallenge_index=3, seed=seed,
                                   do_resample=do_resample, resample_seed=resample_seed)

    @staticmethod
    def get_phenotype_maps():
        sex_map = {
            # 'UNKNOWN': 0,
            "MALE": 1,
            "FEMALE": 2
        }

        race_map = {
            # 'UNKNOWN': 0,
            "WHITE": 1,
            "BLACK": 2
        }

        grading_map = {
            # 'UNKNOWN': 0,
            "II": 1,
            "III": 2,
            "IV": 3
        }

        cancer_map = {
            # 'UNKNOWN': 0,
            'ASTROCYTOMA': 1,
            'GBM': 2,
            'OLIGODENDROGLIOMA': 3,
            # 'MIXED': 4,
            # 'UNCLASSIFIED': 5
        }
        return sex_map, race_map, grading_map, cancer_map, \
               ["Sex", "Race", "Grading", "Cancer"]

    @staticmethod
    def preprocess_phenotypes(phenotypes, phenotypes_names):
        sex_idx, race_idx, who_grading_idx, cancer_type_idx = 0, 1, 2, 3

        sex_map, race_map, grading_map, cancer_map, _ = DataLoader.get_phenotype_maps()
        phenotypes[:, sex_idx] = list(map(lambda x: sex_map[x] if x in sex_map else 0, phenotypes[:, sex_idx]))
        phenotypes[:, race_idx] = list(map(lambda x: race_map[x] if x in race_map else 0, phenotypes[:, race_idx]))
        phenotypes[:, who_grading_idx] = list(map(lambda x: grading_map[x] if x in grading_map else 0,
                                                  phenotypes[:, who_grading_idx]))
        phenotypes[:, cancer_type_idx] = list(map(lambda x: cancer_map[x] if x in cancer_map else 0,
                                                  phenotypes[:, cancer_type_idx]))

        # Transform phenotype to categorical.
        maps = [sex_map, race_map, grading_map, cancer_map]
        inverted_maps = list(map(lambda m: {v: k for k, v in m.items()}, maps))
        phenotypes = np.concatenate([to_categorical(phenotypes[:, i], num_classes=len(maps[i])+1)
                                     for i in range(phenotypes.shape[-1])], axis=1)
        phenotypes_names = [[phenotypes_names[i] + "_" + (inverted_maps[i][j] if j in inverted_maps[i] else "NONE")
                             for j in range(len(maps[i])+1)]
                            for i in range(len(maps))]
        phenotypes_names = reduce(lambda a, b: a+b, phenotypes_names)

        return phenotypes.astype(float), phenotypes_names

    @staticmethod
    @time_function("load_data")
    def get_data(args, phase_index=1, subchallenge_index=1, do_resample=False, seed=0, resample_seed=0):
        x, y, patient_ids, phenotypes_len, phenotypes_names, features_names = \
            DataLoader.get_raw_data(args, phase_index=phase_index, subchallenge_index=subchallenge_index)

        validation_set_fraction = args["validation_set_fraction"]
        test_set_fraction = args["test_set_fraction"]

        split_index_inner = args["split_index_inner"]
        num_splits_inner = args["num_splits_inner"]
        split_index_outer = args["split_index_outer"]
        num_splits_outer = args["num_splits_outer"]

        ret_val = DataLoader.get_data_folds(x, y, patient_ids, phenotypes_len,
                                            validation_set_fraction, test_set_fraction,
                                            do_resample=do_resample, seed=seed, resample_seed=resample_seed,
                                            split_index_inner=split_index_inner, num_splits_inner=num_splits_inner,
                                            split_index_outer=split_index_outer, num_splits_outer=num_splits_outer,
                                            phase_index=phase_index)
        ret_val = list(ret_val)
        ret_val += [phenotypes_names + features_names.tolist()]
        return tuple(ret_val)

    @staticmethod
    def get_raw_data(args, phase_index=1, subchallenge_index=1):
        from pandas import read_csv

        dataset_path = args["dataset_path"]

        # The CSV file must be placed in this file's directory.
        phenotypes = read_csv(os.path.join(dataset_path,
                                           DataLoader.DATA_FILE_NAME_TEMPLATE.format(
                                               PHASE_INDEX=phase_index,
                                               SUBCHALLENGE_INDEX=subchallenge_index,
                                               SUBCHALLENGE_INDICATOR=DataLoader.SUBCHALLENGE_MAP[subchallenge_index],
                                               MODALITY_TYPE=DataLoader.MODALITY_TYPE_PHENOTYPE,
                                               EXTENSION="txt" if phase_index == 2 else "tsv"
                                           )), sep="\t")
        phenotypes_names = phenotypes.columns[1:]
        phenotypes = phenotypes.values

        features = read_csv(os.path.join(dataset_path,
                                         DataLoader.DATA_FILE_NAME_TEMPLATE.format(
                                             PHASE_INDEX=phase_index,
                                             SUBCHALLENGE_INDEX=subchallenge_index,
                                             SUBCHALLENGE_INDICATOR=DataLoader.SUBCHALLENGE_MAP[subchallenge_index],
                                             MODALITY_TYPE=DataLoader.MODALITY_TYPE_FEATURE,
                                             EXTENSION="txt" if phase_index == 2 else "tsv"
                                         )), sep="\t")

        features_names = features.columns.values[1:]
        features = features.values

        patient_id_phenotypes = phenotypes[:, 0:1]
        patient_id_features = features[:, 0:1]

        phenotypes = phenotypes[:, 1:]
        features = features[:, 1:].astype(float)

        if phase_index == 2:
            outcomes = np.zeros((len(phenotypes), 1))  # We do not have outcome data available in Phase 2.
        else:
            outcomes = read_csv(os.path.join(dataset_path,
                                             DataLoader.DATA_FILE_NAME_TEMPLATE.format(
                                                 PHASE_INDEX=phase_index,
                                                 SUBCHALLENGE_INDEX=subchallenge_index,
                                                 SUBCHALLENGE_INDICATOR=DataLoader.SUBCHALLENGE_MAP[subchallenge_index],
                                                 MODALITY_TYPE=DataLoader.MODALITY_TYPE_OUTCOME,
                                                 EXTENSION="txt" if phase_index == 2 else "tsv"
                                             )), sep="\t").values
            patient_id_outcomes = outcomes[:, 0:1]
            outcomes = outcomes[:, 1:]
            assert np.array_equal(patient_id_phenotypes,
                                  patient_id_outcomes), "Patient ID column must be equal across TSV files."

        assert np.array_equal(patient_id_phenotypes,
                              patient_id_features), "Patient ID column must be equal across TSV files."

        phenotypes, phenotypes_names_after_preprocessing = DataLoader.preprocess_phenotypes(phenotypes, phenotypes_names)
        phenotypes_len = phenotypes.shape[-1]

        num_patients = len(phenotypes)
        fraction_survival = 1. - np.mean(outcomes)

        print("INFO: Loaded", num_patients, "patients with",
              fraction_survival*100, "% overall survival rate.", file=sys.stderr)

        x, y = np.concatenate([phenotypes, features], axis=1), outcomes.astype(int)

        assert x.shape[-1] == len(phenotypes_names_after_preprocessing) + len(features_names)

        patient_ids = patient_id_phenotypes
        return x, y, patient_ids, phenotypes_len, phenotypes_names_after_preprocessing, features_names

    @staticmethod
    def get_data_folds(x, y, patient_ids, phenotypes_len, validation_set_fraction, test_set_fraction,
                       do_resample=False, seed=0, resample_seed=0,
                       split_index_inner=0, num_splits_inner=2,
                       split_index_outer=0, num_splits_outer=2,
                       phase_index=1):
        num_patients, input_dim = x.shape[0], x.shape[-1]

        if do_resample:
            random_state = np.random.RandomState(resample_seed)
            resampled_samples = random_state.randint(0, num_patients, size=num_patients)
            x = [x[idx] for idx in resampled_samples]
            patient_ids = [patient_ids[idx] for idx in resampled_samples]
            y = y[resampled_samples]

        num_test_samples = int(np.rint(test_set_fraction * num_patients))
        num_validation_samples = int(np.rint(validation_set_fraction * num_patients))

        if phase_index == 2:
            (x_train, y_train, p_train), (x_val, y_val, p_val) = ([], [], []), ([], [], [])
            (x_test, y_test, p_test) = (x, y, patient_ids)
        else:
            (x_train, y_train, p_train), (x_val, y_val, p_val), (x_test, y_test, p_test)\
                = DataLoader.split_dataset(x, y, patient_ids,
                                           num_validation_samples,
                                           num_test_samples,
                                           phenotypes_len,
                                           seed=seed,
                                           split_index_inner=split_index_inner,
                                           num_splits_inner=num_splits_inner,
                                           split_index_outer=split_index_outer,
                                           num_splits_outer=num_splits_outer)

        # Report fold stats before normalising dates.
        DataLoader.report_data_fold(x_train, y_train, "training set")
        DataLoader.report_data_fold(x_val, y_val, "validation set")
        DataLoader.report_data_fold(x_test, y_test, "test set")

        return (x_train, np.squeeze(y_train), np.squeeze(p_train)), \
               (x_val, np.squeeze(y_val), np.squeeze(p_val)), \
               (x_test, np.squeeze(y_test), np.squeeze(p_test)), \
               input_dim, 1
