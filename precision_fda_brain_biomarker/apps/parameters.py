"""
Copyright (C) 2019  F.Hoffmann-La Roche Ltd.
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

from argparse import ArgumentParser


def parse_parameters():
    parser = ArgumentParser(description='Precision FDA brain cancer biomarker starter project.')
    parser.add_argument("--dataset_path", required=True,
                        help="Folder containing the data files to be loaded.")
    parser.add_argument("--dataset", default="sc1",
                        help="The subchallenge data files to be loaded. One of: (sc1, sc2, sc3).")
    parser.add_argument("--evaluate_against", default="val",
                        help="Fold to evaluate trained models against. One of: (val, test).")
    parser.add_argument("--method", default="LogisticRegression",
                        help="Predictive model to be used."
                             "One of: (NaiveBayes, LogisticRegression, RandomForest, SVM, NeuralNetwork).")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed for the random number generator.")
    parser.add_argument("--num_splits_inner", type=int, default=2,
                        help="Number of splits used for inner cross validation.")
    parser.add_argument("--split_index_inner", type=int, default=0,
                        help="Split index for inner cross validation.")
    parser.add_argument("--num_splits_outer", type=int, default=2,
                        help="Number of splits used for outer cross validation.")
    parser.add_argument("--split_index_outer", type=int, default=0,
                        help="Split index for outer cross validation.")
    parser.add_argument("--output_directory", default="./",
                        help="Base directory of all output files.")
    parser.add_argument("--model_name", default="model.h5.npz",
                        help="Base directory of all output files.")
    parser.add_argument("--load_existing", default="",
                        help="Existing model to load.")
    parser.add_argument("--feature_selection", default="",
                        help="Feature selection method to apply. "
                             "(Default: No feature selection, One of ('', 'PCA', 'SparsePCA', 'FastICA', 'KPCA', '') "
                             "or a list of feature selection steps to be performed in sequence, delimited by a comma.")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of processes to use where available for multitasking.")
    parser.add_argument("--threshold", default=None, type=float,
                        help="Threshold override to use with predictive model. "
                             "If None, the threshold is learnt from the validation set. (Default: None).")
    parser.add_argument("--learning_rate", default=0.0001, type=float,
                        help="Learning rate to use for training.")
    parser.add_argument("--l1_weight", default=0.0, type=float,
                        help="L1 weight decay used on XGBoost.")
    parser.add_argument("--l2_weight", default=0.0, type=float,
                        help="L2 weight decay used on neural network weights and XGBoost.")
    parser.add_argument("--num_epochs", type=int, default=150,
                        help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size to use for training.")
    parser.add_argument("--early_stopping_patience", type=int, default=12,
                        help="Number of stale epochs to wait before terminating training")
    parser.add_argument("--num_units", type=int, default=8,
                        help="Number of neurons to use in NN layers.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers to use in NNs.")
    parser.add_argument("--activation", type=str, default="relu",
                        help="Activation to use in NNs.")
    parser.add_argument("--n_estimators", type=int, default=256,
                        help="Number of neurons to use in Random Forests.")
    parser.add_argument("--max_depth", type=int, default=3,
                        help="Number of layers to use in Random Forests.")
    parser.add_argument("--criterion", type=str, default="gini",
                        help="Splitting criterion to use in Random Forests.")
    parser.add_argument("--dropout", default=0.0, type=float,
                        help="Value of the dropout parameter used in training in the network.")
    parser.add_argument("--svm_c", default=1.0, type=float,
                        help="SVM penalty parameter.")
    parser.add_argument("--kernel", default=1.0, type=float,
                        help="SVM kernel parameter.")
    parser.set_defaults(probability=False)
    parser.add_argument("--enable_probability", dest='probability', action='store_true',
                        help="SVM enable_probability parameter.")
    parser.add_argument("--min_split_loss", default=0.0, type=float,
                        help="min_split_loss value used in XGBoost.")
    parser.add_argument("--subsample", default=1.0, type=float,
                        help="subsample value used in XGBoost.")
    parser.add_argument("--pca_n_components", default=10,
                        help="PCA variance covered parameter (0-1.0) for PCA and int for other methods.")
    parser.add_argument("--base_model", default="LogisticRegression",
                        help="Predictive model to be used when method == SwitchModel."
                             "One of: (NaiveBayes, LogisticRegression, RandomForest, SVM, NeuralNetwork).")
    parser.add_argument("--voting", default="soft",
                        help="Voting method used in Ensemble models. "
                             "(Default: 'soft', one of ('hard', 'soft', 'stacked'))")
    parser.add_argument("--max_num_feature_groups", type=int, default=20,
                        help="Maximum number of feature groups to use for calculating importance scores "
                             "(to limit computational effort).")
    parser.add_argument("--fraction_of_data_set", type=float, default=1,
                        help="Fraction of time_series to use for folds.")
    parser.add_argument("--n_neighbors", type=int, default=10,
                        help="number of neighbors for feature selection")
    parser.add_argument("--validation_set_fraction", type=float, default=0.2,
                        help="Fraction of time_series to hold out for the validation set.")
    parser.add_argument("--test_set_fraction", type=float, default=0.2,
                        help="Fraction of time_series to hold out for the test set.")
    parser.add_argument("--num_hyperopt_runs", type=int, default=35,
                        help="Number of hyperopt runs to perform.")
    parser.add_argument("--hyperopt_offset", type=int, default=0,
                        help="Offset at which to start the hyperopt runs.")
    parser.add_argument("--hyperopt_metric", default="f1",
                        help="Metric to evaluate models against after hyperparameter optimisation.")
    parser.add_argument("--tier", default="PHENO,TIER1",
                        help="tier/ge/pheno selections for each feature selection procedure."
                             "for each procedure, tiers are separated by colon, "
                             "and for a single procedure, tiers are separated by comma.")
    parser.set_defaults(do_train=False)
    parser.add_argument("--do_train", dest='do_train', action='store_true',
                        help="Whether or not to train a model.")
    parser.set_defaults(do_hyperopt=False)
    parser.add_argument("--do_hyperopt", dest='do_hyperopt', action='store_true',
                        help="Whether or not to perform hyperparameter optimisation.")
    parser.set_defaults(do_evaluate=False)
    parser.add_argument("--do_evaluate", dest='do_evaluate', action='store_true',
                        help="Whether or not to evaluate a model.")
    parser.set_defaults(hyperopt_against_eval_set=False)
    parser.add_argument("--hyperopt_against_eval_set", dest='hyperopt_against_eval_set', action='store_true',
                        help="Whether or not to evaluate hyperopt runs against the evaluation set.")
    parser.set_defaults(with_tensorboard=False)
    parser.add_argument("--with_tensorboard", dest='with_tensorboard', action='store_true',
                        help="Whether or not to serve tensorboard data.")
    parser.set_defaults(resample_with_replacement=False)
    parser.add_argument("--resample_with_replacement", dest='resample_with_replacement', action='store_true',
                        help="Whether or not to use resampling w/ replacement in the patient generator.")
    parser.set_defaults(save_predictions=True)
    parser.add_argument("--do_not_save_predictions", dest='save_predictions', action='store_false',
                        help="Whether or not to save predictions.")
    parser.set_defaults(save_attributions=False)
    parser.add_argument("--save_attributions", dest='save_attributions', action='store_true',
                        help="Whether or not to save attributions.")
    parser.set_defaults(whiten_pca=True)
    parser.add_argument("--do_not_whiten_pca", dest='whiten_pca', action='store_false',
                        help="Whether or not to use whitening with PCA.")

    return vars(parser.parse_args())


def clip_percentage(value):
    return max(0., min(1., float(value)))
