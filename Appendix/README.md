# Appendix

## Prior Knowledge Biomarker Features  

For the prior knowledge approach, we engineered new features from the data based on known prognostic biomarkers and gene signatures known from literature.

Table 1. Numbers of biomarkers extracted from literature review for each sub-challenge and categorized in 4 Tiers depending on their impact to the disease.  

|  Sub Challenge  | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
| :---: | ---: | ---: | ---: | ---: |
| 1 |     3  |     4  |    26  |   2   |   
| 2 |     1  |     2  |    33  |   0   |   
| 3 |     4  |     6  |    59  |   2   |   

  
Table 2. Numbers of features selected from the data for each sub-challenge that are based on the above corresponding biomarkers from priori knowledge.  

|  Sub Challenge  | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
| :---: | ---: | ---: | ---: | ---: |
| 1 |    3  |    12  |    26  |   2   |   
| 2 |   46  |     2  |    27  |   0   |   
| 3 |   49  |    14  |    53  |   2   |   


[Here](Prior_knowledge_VFinal.csv) is the detailed table with information on the engineered features per sub-challenge and Tier.


## Model Hyperparameters

[This table](Model_Hyperparameters.csv) lists all the methods for which we performed hyperparameter optimization as part of our modelling.



## Stratified folds
Based on the assumption that the data given to us is representative of the overall population, we applied a stratified cross-validation [[Sechidis et al., 2011](https://rd.springer.com/chapter/10.1007/978-3-642-23808-6_10); [Szymański and Kajdanowicz, 2017](http://proceedings.mlr.press/v74/szyma%C5%84ski17a.html)] that preserved the same phenotype ratios and survival outcome in each fold.

In [this table](survival_phenotypes_in_stratified_folds.csv) we provide one example of the phenotype and survival ratios in the different folds, however the stratification is optimized so that the ratios should be approximately the same in all cases.


## Model evaluation

In the tables below we provide a variety of evaluation metrics for all the models we trained per sub-challenge, both for the inner and the outer cross validation folds.
The ENSEMBLE models were built upon evaluation of the models' F1 score in the inner cross validation folds.
The final best model per sub-challenge was chosen upon evaluation of the models' F1 score in the outer cross validation folds.

- Inner cross validation folds
    - [Sub-challenge 1](./model_evaluation/inner_cv_sc1.csv)
    - [Sub-challenge 2](./model_evaluation/inner_cv_sc2.csv)
    - [Sub-challenge 3](./model_evaluation/inner_cv_sc3.csv)
- Outer cross validation folds
    - [Sub-challenge 1](./model_evaluation/outer_cv_sc1.csv)
    - [Sub-challenge 2](./model_evaluation/outer_cv_sc2.csv)
    - [Sub-challenge 3](./model_evaluation/outer_cv_sc3.csv)