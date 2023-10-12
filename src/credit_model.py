import cloudpickle as cp
import joblib
import pandas as pd
import numpy as np

from functools import reduce

import sys

sys.path.append("../")

# local imports
from preprocess.features.make_features import main as make_features

from src.learner_params import (
    space_column,
    target_column,
    prediction_column,
    params_all, 
    params_ensemble,
    params_fw, 
    params_original,
    MODEL_PARAMS as params_boruta,
)

from utils.functions__training import lgbm_classification_learner
from utils.feature_selection_lists import fw_features, boruta_features, ensemble_features
from utils.features_lists import all_features_list, base_features
from utils.functions__utils import train_binary

from src.config import logger


class CreditModel:
    """
    CreditModel is a class for creating an input dataset, training a model, and making inferences.

    Parameters:
        application_train_df (pd.DataFrame): DataFrame containing training data.
        application_test_df (pd.DataFrame): DataFrame containing testing data.
        bureau_balance_df (pd.DataFrame): DataFrame containing bureau balance data.
        bureau_df (pd.DataFrame): DataFrame containing bureau data.
        installments_payments_df (pd.DataFrame): DataFrame containing installments payments data.
        pos_cash_balance_df (pd.DataFrame): DataFrame containing POS cash balance data.
        previous_application_df (pd.DataFrame): DataFrame containing previous application data.
    """

    def __init__(
        self,
        train_df,
        test_df
    ):
        """ """
        self.train_df = train_df
        self.test_df = test_df
       
    def create_input_dataset(self, verbose: bool = False):
        """
        Create an input dataset by processing and merging multiple DataFrames.

        Parameters:
            verbose (bool, optional): Whether to display verbose output. Default is False.
        """
        logger.info("Creating bureau features...")
        self.frame  = make_full_features(self.train_df, self.test_df, verbose = verbose)
        logger.info("Features creation finished.")

        return self.frame

    def train_predict_fns(self, save_estimator_path: str = None):
        """
        Train a model and save the trained estimator to a specified path.

        Parameters:
            save_estimator_path (str, optional): Path to save the trained estimator. Default is None.
        """
        model_logs = model_pipeline(train_df = data,
                            validation_df = validation_df,
                            params = MODEL_PARAMS,
                            target_column = target_column,
                            features = boruta_features,
                            cv = 3,
                            random_state = 42,
                            apply_shap = False
                          )
        
        fw_full_logs = model_pipeline(train_df = data,
                            validation_df = validation_df,
                            params = params_fw,
                            target_column = target_column,
                            features = fw_features,
                            cv = 3,
                            random_state = 42,
                            apply_shap = False,
                            save_estimator_path=None
                          )

        base_full_logs = model_pipeline(train_df = data,
                            validation_df = validation_df,
                            params = params_original,
                            target_column = target_column,
                            features = base_features,
                            cv = 3,
                            random_state = 42,
                            apply_shap = False,
                            save_estimator_path=None
                          )

        ensemble_full_logs = model_pipeline(train_df = data,
                            validation_df = validation_df,
                            params = params_ensemble,
                            target_column = target_column,
                            features = ensemble_features,
                            cv = 3,
                            random_state = 42,
                            apply_shap = False,
                            save_estimator_path=None
                          )

        all_full_logs = model_pipeline(train_df = data,
                            validation_df = validation_df,
                            params = params_all,
                            target_column = target_column,
                            features = all_features_list,
                            cv = 3,
                            random_state = 42,
                            apply_shap = False,
                            save_estimator_path=None
                          )

        return self

    def train_meta_learner(self):
        """
        """

        l= []
        for name, _df in zip(names, ldf):
            aux = _df[[space_column, "prediction"]].rename(columns = {"prediction":f"prediction_{name}"})
            l.append(aux)
        df_predictions_train= reduce(lambda x,y:pd.merge(x,y,on = space_column), l)



        ###
        estimators = [("tanh",MLPClassifier(random_state=42,
                                            activation="tanh",
                                            max_iter=300,
                                            learning_rate="adaptive")
                      ),
                     ("relu", MLPClassifier(random_state=101,
                                            activation="relu",
                                            max_iter=300,
                                            learning_rate="adaptive")
                     ),
                     ("sigmoid",MLPClassifier(random_state=0,
                                              activation="logistic",
                                              max_iter=300,
                                              learning_rate="adaptive")
                     )
                     ]
        
        model = StackingClassifier(estimators, final_estimator=LogisticRegressionCV(random_state=42,scoring="roc_auc"))
        aux = df_predictions_train.merge(data[[space_column, target_column]], on = space_column)
        result = train_binary(aux, columns, target_column, model)
        ###

    def make_inference(self, save_data_path: str = None, apply_shap: bool = False):
        """
        Make inferences using a trained model.

        Parameters:
            model_path (str, optional): Path to the trained model. Default is None.
            apply_shap (bool, optional): Whether to apply SHAP values during inference. Default is False.
        """

        predictions = self.bst(self.frame, apply_shap=apply_shap)

        if isinstance(save_data_path, str):
            predictions.to_csv(save_data_path, index=False)

        return predictions
