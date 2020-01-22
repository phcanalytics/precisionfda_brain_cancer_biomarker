## PrecisionFDA Brain Cancer Biomarker Challenge

### Authors

* Ying He, China
* Liming Li, China
* Wenjin Li, China
* Chenkai Lv, China
* Elina Koletou, Basel
* Patrick Schwab, Basel
* Gunther Jansen, Basel
* Paul Paczuski, SSF
* Rena Yang, SSF

License: [MIT-License](LICENSE.txt)

### Installation

* Download the dataset files on the challenge website and unzip them to your local file system path `path/to/precisionFDA_dataset`.
* Run `pip install .` (preferably in a [virtual environment](https://virtualenv.pypa.io/en/latest/)) to install all package dependencies.

### Usage

    >> python -u precision_fda_brain_biomarker/apps/main.py --dataset_path=path/to/precisionFDA_dataset --dataset=sc3 --output_directory=/path/to/fda_test_1 --do_train --do_evaluate --num_units=32 --num_layers=2 --method=NeuralNetwork
    INFO: Performance on val0 AUROC = 0.7807017543859649 , with AUPRC = 0.9278329194544808 , with accuracy = 0.72 , with mean = 0.76 , with f1 = 0.8108108108108109 , with specificity = 0.5 , with sensitivity = 0.789473684211 , with sens@95spec = 0.5789473684210527 , with PPV = 0.833333333333 , with NPV = 0.428571428571 , n = 25 , TP = 15 , FP = 3 , TN = 3 , FN = 4

Prediction files are written to `/path/to/fda_test_1/*_predictions.tsv` and `/path/to/fda_test_1/*_predictions.thresholded.tsv` for `train, test, val` folds. 

Trained models are written to `/path/to/fda_test_1/model.{FILE_EXTENSION}`.

The models available currently are: NaiveBayes, LogisticRegression, RandomForest, SVM, NeuralNetwork.

You can add feature selection with principal components analysis (PCA) to any model using the flag `--feature_selection=PCA`.

### Nested CV

    >> python -u precision_fda_brain_biomarker/apps/main.py --dataset_path=path/to/precisionFDA_dataset --dataset=sc3 --output_directory=/path/to/fda_test_1 --do_train --do_evaluate --num_units=32 --num_layers=2  --method=RandomForest --num_splits_inner=5 --num_splits_outer=5
    INFO: INNER_0 cross validation results (N=5) are:
    auroc = 0.6929627640529896 +- 0.1020621032940511 CI=( 0.5764473684210526 , 0.8395918367346937 ), median= 0.7071428571428571 min= 0.575 max= 0.8503401360544216
    auprc = 0.8804846412259323 +- 0.055872686861524856 CI=( 0.8109990684533063 , 0.948241699014431 ), median= 0.8767062036625151 min= 0.8088431818215436 max= 0.9494931522047347
    f1 = 0.7236591478696741 +- 0.10349188545560231 CI=( 0.5522222222222222 , 0.8356725146198831 ), median= 0.742857142857143 min= 0.5333333333333333 max= 0.8421052631578948
    sens@95spec = 0.3096240601503759 +- 0.2344381640346768 CI=( 0.010526315789473684 , 0.5714285714285714 ), median= 0.3 min= 0.0 max= 0.5714285714285714
    ppv = 0.8602777777777778 +- 0.04921921234840649 CI=( 0.8012500000000001 , 0.9288888888888889 ), median= 0.8666666666666667 min= 0.8 max= 0.9333333333333333
    npv = 0.32234334293157824 +- 0.07481719659216153 CI=( 0.25227272727272726 , 0.44871794871794873 ), median= 0.29411764705882354 min= 0.25 max= 0.46153846153846156
    specificity = 0.6142857142857143 +- 0.160102008298536 CI=( 0.41000000000000003 , 0.8428571428571429 ), median= 0.6 min= 0.4 max= 0.8571428571428571
    sensitivity = 0.633984962406015 +- 0.13124311839446567 CI=( 0.42190476190476195 , 0.788421052631579 ), median= 0.6666666666666666 min= 0.4 max= 0.8
    threshold = 0.7563637469329485 +- 0.030881664372426995 CI=( 0.7064768100335865 , 0.7895818423459773 ), median= 0.7632834333841108 min= 0.7019422103037772 max= 0.7908141925105987

You can obtain estimates of the variance of method performance across 5-fold splits by adding `--num_splits_inner=5 --num_splits_outer=5` to the command line argument list. If you train cross-validated models, a model and prediction list for each fold is saved to the output directory.

## Ensembling

    >> python -u precision_fda_brain_biomarker/apps/main.py --dataset_path=path/to/precisionFDA_dataset --dataset=sc3 --output_directory=/path/to/fda_test_1 --do_evaluate --method="/path/to/model/directory_1,/path/to/model/directory_2" --num_splits_inner=5 --num_splits_outer=5 --evaluate_against=test
    auroc = 0.6923076923076923 +- 0.0 CI=( 0.6923076923076923 , 0.6923076923076923 ), median= 0.6923076923076923 min= 0.6923076923076923 max= 0.6923076923076923
    auprc = 0.8280306439466842 +- 0.0 CI=( 0.8280306439466842 , 0.8280306439466842 ), median= 0.8280306439466842 min= 0.8280306439466842 max= 0.8280306439466842
    f1 = 0.8070175438596492 +- 0.0 CI=( 0.8070175438596492 , 0.8070175438596492 ), median= 0.8070175438596492 min= 0.8070175438596492 max= 0.8070175438596492
    sens@95spec = 0.038461538461538464 +- 0.0 CI=( 0.038461538461538464 , 0.038461538461538464 ), median= 0.038461538461538464 min= 0.038461538461538464 max= 0.038461538461538464
    ppv = 0.7419354838709677 +- 0.0 CI=( 0.7419354838709677 , 0.7419354838709677 ), median= 0.7419354838709677 min= 0.7419354838709677 max= 0.7419354838709677
    npv = 0.25 +- 0.0 CI=( 0.25 , 0.25 ), median= 0.25 min= 0.25 max= 0.25
    specificity = 0.11111111111111112 +- 1.3877787807814457e-17 CI=( 0.1111111111111111 , 0.1111111111111111 ), median= 0.1111111111111111 min= 0.1111111111111111 max= 0.1111111111111111
    sensitivity = 0.8846153846153847 +- 1.1102230246251565e-16 CI=( 0.8846153846153846 , 0.8846153846153846 ), median= 0.8846153846153846 min= 0.8846153846153846 max= 0.8846153846153846
    threshold = 0.5 +- 0.0 CI=( 0.5 , 0.5 ), median= 0.5 min= 0.5 max= 0.5

You can evaluate the performance of an ensemble of previously fitted models by passing a comma-seperated list of two or more directories (corresponding to the directories denoted in `--output_directory` in your command line invocations) to the `--method` parameter. Any feature selection potentially used in the original model will be re-instantiated along with the pre-trained model.

Note that ensembles do not need to be trained - they output the mean prediction across the equally-weighted ensemble member models.

Note also that the performance estimates may be biased if you evaluate your ensemble on data folds that the ensemble members were previously trained on (e.g. if you performed nested CV).

### How to add a custom model?

Create a new file that conforms to the interface specified by [BaseModel](precision_fda_brain_biomarker/models/baselines/base_model.py) and that of [scikit-learn predictors](https://scikit-learn.org/dev/developers/develop.html) in the `precision_fda_brain_biomarker/models/baselines` folder.

See for example:
* [Example how to do this for logistic regression](precision_fda_brain_biomarker/models/baselines/logistic_regression.py)
* [Example how to do this for random forests](precision_fda_brain_biomarker/models/baselines/random_forest.py)
* [How to do this for neural networks](precision_fda_brain_biomarker/models/baselines/neural_network.py)

If you want to additionally pass your model configuration through the command line you have to:

1. add connectors to the [command line args interface](precision_fda_brain_biomarker/apps/parameters.py), and
2. [pass them to the model](precision_fda_brain_biomarker/apps/main.py)

### How to add a custom feature selection step?

Create a new file that conforms to the interface specified by [BaseModel](precision_fda_brain_biomarker/models/baselines/base_model.py) and that of [scikit-learn transformers](https://scikit-learn.org/dev/developers/develop.html) in the `precision_fda_brain_biomarker/models/feature_selection` folder.

See for example:
* [Example how to do this for PCA](precision_fda_brain_biomarker/models/feature_selection/pca.py)

### Relevant links

* [PrecisionFDA Challenge Website](https://precision.fda.gov/challenges/8)

### Acknowledgements

This work was funded by F. Hoffmann-La Roche Ltd.
