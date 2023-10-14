from datetime import datetime
import pandas as pd
import numpy as np

import sys

sys.path.append("../")

# local imports
from src.config import logger
from src.credit_model import CreditModel
from utils.functions__utils import load_csv_files


def main(
    data_directory: str = None,
    save_data_path: str = None,
    apply_shap:bool = False,
    save_estimator_path: str = None,
):
    """
    Perform the main processing steps for credit modeling.

    Parameters:
        data_directory (str, optional): The directory containing the dataset files. Default is None.
        save_data_path (str, optional): The path to save the inference results. Default is None.
        apply_shap (bool, optional): Whether to apply SHAP values during model training. Default is False.
        save_estimator_path (str, optional): Path to save the trained estimator. Default is None.

    Returns:
        pd.DataFrame: Inference results, including the predicted probabilities.
    """

    logger.info(" [1] loading the datasets...")
    loaded_data = load_csv_files(data_directory)

    train_df = loaded_data["cs-training"]
    test_df = loaded_data["cs-test"]
    logger.info("datasets loaded.")

    credit_model = CreditModel(
        train_df,
        test_df
    )

    logger.info("[2] creating input dataset...")
    input_dataset = credit_model.create_input_dataset(verbose=False)
    logger.info("input dataset created.")

    logger.info("[3] training predict function...")
    bst = credit_model.train_predict_fns(apply_shap = apply_shap)
    meta_learner = bst.train_meta_learner(save_estimator_path = save_estimator_path)
    logger.info("predict function trained.")

    output_df = credit_model.make_inference(
        save_data_path=save_data_path
    )

    return output_df


if __name__ == "__main__":
    fecha = datetime.now().strftime("%Y%m%d")
    output_df = main(
        "data/", save_data_path=f"submissions/final_submission_{fecha}.csv"
    )
