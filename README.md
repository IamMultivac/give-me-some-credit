# Kaggle Competition: Give Me Some Credit

**Competition Overview:** [Give Me Some Credit Overview](https://www.kaggle.com/competitions/GiveMeSomeCredit/overview)

**Model Developer:** [@julio.luna](https://github.com/julio.luna)

## Table of Contents

1. [Summary](#summary)
2. [Model Methodology and Outputs](#model-methodology-and-outputs)
3. [Feature Creation](#feature-creation)
4. [Feature Selection](#feature-selection)
5. [Hyperparameter Optimization](#hyperparameter-optimization)
6. [Model Evaluation](#model-evaluation)
7. [Model Explainability](#model-explainability)
8. [Model Ensemble in Detail](#model-ensemble-in-detail)
9. [Performance in the Competition](#performance-in-the-competition)
10. [Final Thoughts](#final-thoughts)
11. [Replicability](#replicability)
12. [Future Work](#future-work)

## Summary

This project was dedicated to the application of a diverse set of techniques to enhance model performance, encompassing feature ensemble, feature interactions, monotone constraints, and linear trees on gradient boosting models. As such, it served as an experimental playground for the application of state-of-the-art approaches to modeling.

Unlike a real-world challenge in which data engineering and feature creation are primary, this competition provided a set of 10 features that exhibited commendable standalone performance (ROC AUC: 0.8321). The challenge here was to explore creativity in combining and maximizing the utility of these features to enhance model performance, irrespective of the complexity or feasibility of a production environment.

After thorough exploration and feature combination, the project yielded a set of 170 features (160 additional to the original ones). This feature set encompassed a variety of aggregations and ensembling techniques, including model-based aggregation, feature interactions, and k-means clustering.

The final model, which achieved the highest private score, involved an ensemble of models via stacking, employing Multilayer Perceptrons (MLP) stacked by a logistic regression. Despite its potential infeasibility for a production environment, the model secured a private score of 0.86314, positioning it 5th in the competition, earning a gold medal.

## Introduction

In this project, various cutting-edge techniques were employed to enhance model performance. Notably, the final model exhibited high complexity, which may not be suitable for a production environment but demonstrates the potential for significant performance improvements.

### Evaluation

Submissions were evaluated based on the area under the ROC curve, providing a robust measure of model discrimination ability between predicted probabilities and actual target values.

## Model Methodology and Outputs

The model's approach was rooted in binary classification using the LightGBM algorithm, leveraging features created from all available data sources. Multiple models were developed and evaluated:

1. Model using the base features
2. Model using features selected by the Boruta feature selection process
3. Model using features selected by at least one of the previous methods
4. Model using all constructed features

The final model comprised a stacking ensemble of five different models, with predictions aggregated using a Multilayer Perceptron (MLP). The table below summarizes the different models, the number of features used, and their performance:

| Model    | Number of Features | OOF ROC AUC | Validation ROC AUC |
|----------|---------------------|------------|--------------------|
| boruta   | 68                  | 0.861483   | 0.865860           |
| ensemble | 82                  | 0.861489   | 0.865703           |
| base     | 173                 | 0.861300   | 0.865668           |
| fw       | 53                  | 0.861526   | 0.865560           |
| original | 10                  | 0.863661   | 0.865334           |

The results of this feature combination and model stacking process demonstrate that the boruta method achieved the highest score among all models, surpassing the base model with only 10 features.

## Feature Creation

Feature creation in this project involved the combination of the existing set of 10 features using various methods:

1. Model-based features: Different models, including Random Forest, AdaBoost, and Logistic Regression, were trained on the original features, and the output of these models was incorporated as new features.

2. Polynomial features: Interactions of order 2 (squared and product features) were generated, resulting in a larger feature set. A feature selection method was applied to retain only those with importance exceeding the median importance of all features, as determined by a LightGBM model.

3. Kmeans clustering: The base features were clustered, creating a new feature representing the cluster ID.

4. Encoding features: Interactions between features were created by averaging feature A based on the percentiles of feature B after splitting it into deciles. This approach was extended across many features.

In total, 170 features were created by using and combining the above described steps.

## Feature Selection

Feature selection in this project involved exploring three distinct and independent approaches, along with an ensemble technique that combined features selected in at least one process:

1. Minimum Redundancy Maximum Relevance using [SULOV](https://github.com/AutoViML/featurewiz)
2. All-relevant selection using Boruta
3. Ensemble model using features selected in at least one process

Comprehensive details can be found in the [notebook](https://github.com/IamMultivac/give-me-some-credit/blob/master/research/feature-selection.ipynb).

## Hyperparameter Optimization

Hyperparameter optimization was conducted using [Optuna](https://optuna.org/), allowing for an extensive exploration of hyperparameters that enhanced model generalization and regularization. This reduced the model's reliance on a few features for predictions and improved its overall performance. A detailed analysis is available [here](https://github.com/IamMultivac/give-me-some-credit/blob/master/research/hyperopt.ipynb).

## Model Evaluation

All models underwent evaluation on out-of-fold and validation sets to assess their performance on the private dataset. Three different types of ensembles were tested on all submodel predictions:

1. Arithmetic mean
2. Linear model
3. MLP model

The following table displays the results:

| Model       | Number of Features | OOF ROC AUC | Validation ROC AUC |
|-------------|---------------------|------------|--------------------|
| fw          | 32                  | 0.863464   | 0.867463           |
| boruta     | 46                  | 0.864788   | 0.867128           |
| all features | 115                 | 0.862991   | 0.867067           |
| ensemble    | 55                  | 0.862760   | 0.865526           |
| base_features | 10                  | 0.864081   | 0.865510           |

## Model Explainability

Explainability analysis was conducted using the Boruta model, which exhibited the best performance among the non-stacker models. SHAP values contributed to revealing the top 10 most important features, ranked from most to least impactful:

1. modelEnsemblePredictionAdaBoost
2. sumTotalTimePastDuePerCreditLoan
3. modelEnsemblePredictionRandomForest
4. age
5. RevolvingUtilizationOfUnsecuredLines
6. DebtRatio MonthlyIncome
7. RevolvingUtilizationOfUnsecuredLines age
8. NumberOfOpenCreditLinesAndLoans RevolvingUtilizationOfUnsecuredLines
9. IncomePerCreditLineLoan
10. PercentageNumberTimes30to59PastDue

These results indicate that the most impactful features are the inner models created from individual data sources. This approach effectively condensed information from each source, allowing the model to capture a high amount of information with a reduced set of features.

Regarding the importance by feature type (original vs. added features), it was observed that, in general, the original features exhibited higher median impact.

A comprehensive analysis is available [here](https://github.com/IamMultivac/give-me-some-credit/blob/master/research/shap-values-analysis.ipynb).

## Model Ensemble in Detail

This stage involved testing different aggregation strategies, ranging from simple averaging of model predictions to various stacking models. Four different LightGBM models were trained on different feature subsets, and their predictions were used to train and predict the target variable via a meta-learner, incorporating cross-validation predictions from every model.

While this approach yielded improved performance, it introduces complexity to model deployment and monitoring. Attention must be given to individual model behavior and hyperparameters of the stacker.

The final model consisted of an ensemble of three MLP algorithms, each using different activation functions. Predictions from these three models were stacked by a logistic regression.

A detailed analysis can be found [here](https://github.com/IamMultivac/give-me-some-credit/blob/master/research/train-model-pipeline.ipynb).

## Performance in the Competition

- **Public vs. Private Score**: The table presents both public and private scores for different models. It's common in competitions to have separate public and private datasets for testing, and it's important to consider both metrics. In this case, the private scores are slightly higher than the public scores for all models, suggesting similar performance on both datasets.

- **Multiple Stacking MLP and Stacking MLP**: The "Multiple Stacking MLP" and "Stacking MLP" models have similar scores, with the "Multiple Stacking MLP" having a slight edge, indicating that increased complexity may not significantly improve performance.

- **Stacking LR and Stacking AVG**: "Stacking LR" and "Stacking AVG" models also have similar scores. "Stacking LR" slightly outperforms "Stacking AVG," suggesting that a linear regression-based stacking approach performs similarly to a simple average ensemble.

- **Boruta + Optuna**: The "Boruta + Optuna" model has slightly lower scores compared to the stacking models but remains competitive.

Overall, the performance differences among these models are relatively small. The most complex model achieved the highest performance.

[Image](https://github.com/IamMultivac/give-me-some-credit/blob/master/img/kaggle-gsc.png)

## Final Thoughts

This competition provided a unique opportunity to explore and test innovative techniques for improving model performance. It's worth noting that the final model exhibits extremely high complexity, making it impractical for deployment and monitoring in a production environment.

Nonetheless, the techniques used for feature interactions, which are relatively cost-effective, may be worth considering when striving to enhance model performance in real-world scenarios.

## Replicability

To replicate the project, follow these steps:

```bash
!pip install -r requirements.txt
python3 main.py
