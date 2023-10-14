import cloudpickle as cp
import joblib
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegressionCV

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

from utils.functions__training import model_pipeline
from utils.feature_selection_lists import (
    fw_features,
    boruta_features,
    ensemble_features,
)
from utils.features_lists import all_features_list, base_features
from utils.functions__utils import train_binary
from utils.lists_constants import names, columns

from src.config import logger


class CreditModel:
    """
    CreditModel is a class for creating an input dataset, training a model, and making inferences.

    Parameters:
        train_df (pd.DataFrame): DataFrame containing training data.
        test_df (pd.DataFrame): DataFrame containing testing data.
    """

    def __init__(self, train_df, test_df):
        """
        Initialize a CreditModel instance.

        Parameters:
            train_df (pd.DataFrame): DataFrame containing training data.
            test_df (pd.DataFrame): DataFrame containing testing data.
        """
        self.train_df = train_df
        self.test_df = test_df

    def _create_dummy_dataset(self, columns, taget_column, space_column):
        """
        Create a dummy dataset with specified columns.

        Parameters:
            columns (list): List of column names for the dummy dataset.
            taget_column (str): The target column name.
            space_column (str): The space column name.

        Returns:
            pd.DataFrame: A DataFrame with the specified columns and NaN values.
        """
        extra_columns = [taget_column, space_column]
        df = pd.DataFrame(columns=columns + extra_columns, index=[0])

        for col in columns:
            df[col] = np.NaN

        return df

    def create_input_dataset(self, verbose: bool = False):
        """
        Create an input dataset by processing and merging multiple DataFrames.

        Parameters:
            verbose (bool, optional): Whether to display verbose output. Default is False.

        Returns:
            pd.DataFrame: The processed input dataset.
        """
        self.frame = make_features(self.train_df, self.test_df, verbose=verbose)
        logger.info("Features creation finished.")

        return self.frame

    def train_predict_fns(self, apply_shap=False):
        """
        Train machine learning models for feature selection and prediction.

        Parameters:
            apply_shap (bool, optional): Whether to apply SHAP values. Default is False.
        """
        self.train_df = self.frame[self.frame[target_column].notnull()].reset_index(
            drop=True
        )
        logger.info("Training boruta features learner...")
        self.boruta_logs = model_pipeline(
            train_df=self.train_df,
            validation_df=self._create_dummy_dataset(
                boruta_features, target_column, space_column
            ),
            params=params_boruta,
            target_column=target_column,
            features=boruta_features,
            cv=3,
            random_state=42,
            apply_shap=apply_shap,
        )
        logger.info(
            f"Boruta features learner finished. | ROC AUC: {self.boruta_logs['metrics']['roc_auc']['out_of_fold']}"
        )

        logger.info("Training featurewiz features  learner learner...")
        self.fw_logs = model_pipeline(
            train_df=self.train_df,
            validation_df=self._create_dummy_dataset(
                fw_features, target_column, space_column
            ),
            params=params_fw,
            target_column=target_column,
            features=fw_features,
            cv=3,
            random_state=42,
            apply_shap=apply_shap,
        )
        logger.info(
            f"Featurewiz features learner finished. | ROC AUC: {self.fw_logs['metrics']['roc_auc']['out_of_fold']}"
        )

        logger.info("Training original features learner...")
        self.original_logs = model_pipeline(
            train_df=self.train_df,
            validation_df=self._create_dummy_dataset(
                base_features, target_column, space_column
            ),
            params=params_original,
            target_column=target_column,
            features=base_features,
            cv=3,
            random_state=42,
            apply_shap=apply_shap,
        )
        logger.info(
            f"Original features learner finished. | ROC AUC: {self.original_logs['metrics']['roc_auc']['out_of_fold']}"
        )

        logger.info("Training ensemble features learner...")
        self.ensemble_logs = model_pipeline(
            train_df=self.train_df,
            validation_df=self._create_dummy_dataset(
                ensemble_features, target_column, space_column
            ),
            params=params_ensemble,
            target_column=target_column,
            features=ensemble_features,
            cv=3,
            random_state=42,
            apply_shap=apply_shap,
        )
        logger.info(
            f"Ensemble features learner finished. | ROC AUC: {self.ensemble_logs['metrics']['roc_auc']['out_of_fold']}"
        )

        logger.info("Training all features learner...")
        self.all_features_logs = model_pipeline(
            train_df=self.train_df,
            validation_df=self._create_dummy_dataset(
                all_features_list, target_column, space_column
            ),
            params=params_all,
            target_column=target_column,
            features=all_features_list,
            cv=3,
            random_state=42,
            apply_shap=apply_shap,
        )
        logger.info(
            f"All features learner finished. | ROC AUC: {self.all_features_logs['metrics']['roc_auc']['out_of_fold']}"
        )

        return self

    def train_meta_learner(self, save_estimator_path: str = None):
        """
        Train a meta-learner using predictions from multiple models.

        Parameters:
            save_estimator_path (str, optional): Path to save the trained meta-learner estimator. Default is None.
        """
        l = []
        ldf = [
            self.boruta_logs["data"]["oof_df"],
            self.fw_logs["data"]["oof_df"],
            self.original_logs["data"]["oof_df"],
            self.ensemble_logs["data"]["oof_df"],
            self.all_features_logs["data"]["oof_df"],
        ]
        for name, _df in zip(names, ldf):
            aux = _df[[space_column, "prediction"]].rename(
                columns={"prediction": f"prediction_{name}"}
            )
            l.append(aux)
        df_predictions_train = reduce(lambda x, y: pd.merge(x, y, on=space_column), l)

        estimators = [
            (
                "tanh",
                MLPClassifier(
                    random_state=42,
                    activation="tanh",
                    max_iter=300,
                    learning_rate="adaptive",
                ),
            ),
            (
                "relu",
                MLPClassifier(
                    random_state=101,
                    activation="relu",
                    max_iter=300,
                    learning_rate="adaptive",
                ),
            ),
            (
                "sigmoid",
                MLPClassifier(
                    random_state=0,
                    activation="logistic",
                    max_iter=300,
                    learning_rate="adaptive",
                ),
            ),
        ]

        self.model = StackingClassifier(
            estimators,
            final_estimator=LogisticRegressionCV(random_state=42, scoring="roc_auc"),
        )
        aux = df_predictions_train.merge(
            self.train_df[[space_column, target_column]], on=space_column
        )
        logger.info("Training meta learner...")
        self.learner = train_binary(aux, columns, target_column, self.model)
        logger.info("Meta learner training finished.")

        return self

    def make_inference(self, save_data_path: str = None):
        """
        Make inferences using a trained model and save the results to a file.

        Parameters:
            save_data_path (str, optional): Path to save the inference results. Default is None.

        Returns:
            pd.DataFrame: Inference results, including the predicted probabilities.
        """
        l = []
        lpf = [
            self.boruta_logs["lgbm_classification_learner"],
            self.fw_logs["lgbm_classification_learner"],
            self.original_logs["lgbm_classification_learner"],
            self.ensemble_logs["lgbm_classification_learner"],
            self.all_features_logs["lgbm_classification_learner"],
        ]
        self.test_df = self.frame[self.frame[target_column].isnull()].reset_index(
            drop=True
        )
        for name, predict_fn in zip(names, lpf):
            aux = predict_fn["predict_fn"](self.test_df)[
                [space_column, "prediction"]
            ].rename(columns={"prediction": f"prediction_{name}"})
            l.append(aux)
        logger.info("Running inference on the private set...")
        df_predictions = reduce(lambda x, y: pd.merge(x, y, on=space_column), l)
        df_predictions.loc[:, "Probability"] = self.learner["model"].predict_proba(
            df_predictions[columns]
        )[:, 1]
        logger.info("Inference finished.")

        if isinstance(save_data_path, str):
            df_predictions[[space_column, "Probability"]].to_csv(
                save_data_path, index=False
            )

        return df_predictions
