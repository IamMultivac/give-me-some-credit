import numpy as np
import pandas as pd
import sys

from lightgbm import LGBMClassifier as lgbm

from functools import reduce

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import SelectFromModel

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from time import time
import warnings; warnings.filterwarnings("ignore")

sys.path.append("../")

# local imports
from src.config import logger
from src.learner_params import target_column, space_column, prediction_column
from utils.functions__utils import train_binary
from utils.features_lists import base_features, delinquency_features, socio_demo_features, credit_features

agg_fns = ["min","max","mean","std"]
names = ["MLPClassifier", "RandomForestClassifier", "AdaBoostClassifier"]

merge_fn = lambda x,y: pd.merge(x,y, left_index = True, right_index =True)

seed = 42

base_model = lgbm(random_state = 42, n_estimators = 1000, verbose = -1)

bst_ls = [Pipeline([("imputer", SimpleImputer(strategy = "constant", fill_value = 0)),
                    ("model", MLPClassifier(random_state = seed))]),
          Pipeline([("imputer", SimpleImputer(strategy = "constant", fill_value = 0)),
                    ("model", RandomForestClassifier(random_state = seed))]),
          Pipeline([("imputer", SimpleImputer(strategy = "constant", fill_value = 0)),
                    ("model", AdaBoostClassifier(random_state = seed))])
         ]

def main(train_df:pd.DataFrame,
         test_df:pd.DataFrame,
         *,
         min_clusters:int = 8,
         verbose:bool = True
        ):
    """
    Perform feature engineering on bureau and bureau balance data.

    Parameters:
    -----------
    train_df : pd.DataFrame
        DataFrame containing train data.

    test_df : pd.DataFrame
        DataFrame containing test data.

    verbose : bool, optional
        If True, print verbose log messages. Default is True.

    Returns:
    --------
    pd.DataFrame
        DataFrame with engineered features.

    """
    T0 = time()
    frame = pd.concat([train_df, test_df], ignore_index = True)
    logger.info("Creating features...")
    frame.loc[:,"SumTotalTimesPastDue"] = frame[delinquency_features].sum(axis = 1)
    frame.loc[:,"MeanTotalTimesPastDue"] = frame[delinquency_features].mean(axis = 1)
    frame.loc[:,"StdTotalTimesPastDue"] = frame[delinquency_features].std(axis = 1)
    frame.loc[:,"MaxTotalTimesPastDue"] = frame[delinquency_features].max(axis = 1)
    frame.loc[:,"MinTotalTimesPastDue"] = frame[delinquency_features].min(axis = 1)
    
    frame.loc[:,"PercentageNumberTimes30to59PastDue"] = (frame["NumberOfTime30-59DaysPastDueNotWorse"]/
                                                                 frame["SumTotalTimesPastDue"]
                                             )
    frame.loc[:,"PercentageNumberTimes60to89PastDue"] = (frame["NumberOfTime60-89DaysPastDueNotWorse"]/
                                                                 frame["SumTotalTimesPastDue"]
                                             )
    frame.loc[:,"PercentageNumberTimes90plusPastDue"] = (frame["NumberOfTimes90DaysLate"]/
                                                                 frame["SumTotalTimesPastDue"]
                                             )
    
    frame.loc[:,"LogDifference30to5960to89"] = np.log1p(frame["NumberOfTime30-59DaysPastDueNotWorse"]) - np.log1p(frame["NumberOfTime60-89DaysPastDueNotWorse"])
    frame.loc[:,"LogDifference60to8990plus"] = np.log1p(frame["NumberOfTime60-89DaysPastDueNotWorse"]) - np.log1p(frame["NumberOfTimes90DaysLate"])
    
    
    frame["IncomePerCreditLineLoan"] = frame["MonthlyIncome"] / frame["NumberOfOpenCreditLinesAndLoans"]
    frame["IncomePerRealEstateLoan"] = frame["MonthlyIncome"] / frame["NumberRealEstateLoansOrLines"]
    frame["IncomePerDependent"] = frame["MonthlyIncome"] / frame["NumberOfDependents"]
    frame["IncomePerAge"] = frame["MonthlyIncome"] / frame["age"]
    frame["DependentsPerAge"] = frame["NumberOfDependents"] / frame["age"]
    frame["RealEstateLoansPerCreditLineLoan"] = frame["NumberRealEstateLoansOrLines"] / frame["NumberOfOpenCreditLinesAndLoans"]
    # frame["UtilizationTimesCreditLineLoan"] = frame["RevolvingUtilizationOfUnsecuredLines"] * frame["NumberOfOpenCreditLinesAndLoans"]
    # frame["IncomeTimesUtilization"] = frame["MonthlyIncome"] * frame["RevolvingUtilizationOfUnsecuredLines"]
    frame["UtilizationOverDebtRatio"] = frame["RevolvingUtilizationOfUnsecuredLines"] / frame["DebtRatio"]
    frame["IncomeOverDebtRatio"] = frame["MonthlyIncome"] / frame["DebtRatio"]
    frame["SumTotalTimesPastDuePerCreditLineLoan"] =  frame["NumberOfOpenCreditLinesAndLoans"]/frame["SumTotalTimesPastDue"]

    # modellling features
    logger.info("Creating model based features...")
    n_clusters = int(np.log10(train_df.shape[0]) * min_clusters)
    n_clusters = max(8, n_clusters)
    kmf = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                 ("scaler", StandardScaler()),
                 ("cluster", MiniBatchKMeans(random_state=42, n_clusters=n_clusters))
                ]).fit(train_df[base_features])
    
    train_df = train_df.assign(**{"cluster_group":kmf.predict(train_df[base_features])})
    test_df = test_df.assign(**{"cluster_group":kmf.predict(test_df[base_features])})

    aux_cluster_group_df = pd.concat([train_df, test_df], ignore_index = True)["cluster_group"]

    frame = pd.merge(frame, aux_cluster_group_df, left_index = True, right_index = True)

    poly_feats = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                 ("featurizer", PolynomialFeatures(include_bias = False))
                ]).fit(train_df[base_features])
    poly_feats_names = poly_feats["featurizer"].get_feature_names_out(base_features)

    tmp_train = train_df.assign(**pd.DataFrame(poly_feats.transform(train_df[base_features]),
             columns = poly_feats_names))
    
    tmp_poly_feats_names = [x for x in poly_feats_names if x not in base_features]

    sfm = SelectFromModel(estimator = base_model,
                        threshold="median",
                        prefit=False).fit(tmp_train[tmp_poly_feats_names], train_df[target_column]) 
    extra_columns = sfm.get_feature_names_out()

    aux_train_df = train_df.assign(**pd.DataFrame(poly_feats.transform(train_df[base_features]),
             columns = poly_feats_names))
    
    aux_test_df = test_df.assign(**pd.DataFrame(poly_feats.transform(test_df[base_features]),
             columns = poly_feats_names))
    aux_poly_feats_df = pd.concat([aux_train_df, aux_test_df], ignore_index = True)[extra_columns]

    frame = pd.merge(frame, aux_poly_feats_df, left_index = True, right_index = True)

    ldf = []
    for name, bst in zip(names, bst_ls):
        base_learner_logs = train_binary(train_df,
                                           base_features,
                                           target_column,
                                           bst
                                          )
        base_learner_train= base_learner_logs["data"]
        
        base_learner_test = base_learner_logs["p"](test_df)
    
        tmp_base_learner_df = pd.concat([base_learner_train, base_learner_test], ignore_index = True)[[prediction_column]].rename(columns = {prediction_column:f"{prediction_column}_{name}"})

        ldf.append(tmp_base_learner_df)

    base_learner_df = reduce(merge_fn, ldf)

    frame = pd.merge(frame, base_learner_df, left_index = True, right_index = True)
    
    logger.info("Model based features successfully created.")
    
    # encoding features
    for socio in socio_demo_features:
        for credit in credit_features:
            for agg_fn in agg_fns:
                cuts, bins = pd.qcut(train_df[socio], q=10, labels=False, retbins=True, duplicates = "drop")
                _mapa = dict(train_df.groupby(cuts)[credit].agg(agg_fn))
                train_df.loc[:,f"CategoryEncoded_{socio}_{agg_fn}_{credit}"]= pd.cut(train_df[socio], bins=bins, labels=False).map(_mapa)
                test_df.loc[:,f"CategoryEncoded_{socio}_{agg_fn}_{credit}"] = pd.cut(test_df[socio], bins=bins, labels=False).map(_mapa)
    aux_columns = [x for x in train_df.columns if "CategoryEncoded_" in x] 

    aux_encoded_df = pd.concat([train_df, test_df], ignore_index = True)[aux_columns]

    frame = pd.merge(frame, aux_encoded_df, left_index = True, right_index = True)

    frame = frame.replace(np.Inf,np.NaN)
    frame = frame.replace(-np.Inf,np.NaN)

    logger.info(f"Successfully created featureset of length: {len(frame)} in: {((time() - T0) / 60):.2f} minutes")

    if verbose:
        frame.info(verbose = True, show_counts = True)

    return frame

