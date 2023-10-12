target_column = "SeriousDlqin2yrs"
space_column = "Id"
prediction_column = "prediction"



monotone_const_dict ={'CategoryEncoded_MonthlyIncome_max_DebtRatio': 1.0,
 'CategoryEncoded_MonthlyIncome_max_RevolvingUtilizationOfUnsecuredLines': -1.0,
 'CategoryEncoded_age_max_DebtRatio': -1.0,
 'CategoryEncoded_age_max_NumberOfOpenCreditLinesAndLoans': -1.0,
 'CategoryEncoded_age_max_NumberRealEstateLoansOrLines': -1.0,
 'CategoryEncoded_age_mean_NumberOfTime30-59DaysPastDueNotWorse': 1.0,
 'CategoryEncoded_age_mean_RevolvingUtilizationOfUnsecuredLines': 1.0,
 'CategoryEncoded_cluster_group_max_DebtRatio': -1.0,
 'CategoryEncoded_cluster_group_max_NumberOfOpenCreditLinesAndLoans': -1.0,
 'CategoryEncoded_cluster_group_max_NumberOfTime30-59DaysPastDueNotWorse': -1.0,
 'CategoryEncoded_cluster_group_max_NumberOfTime60-89DaysPastDueNotWorse': -1.0,
 'CategoryEncoded_cluster_group_max_NumberOfTimes90DaysLate': 1.0,
 'CategoryEncoded_cluster_group_max_RevolvingUtilizationOfUnsecuredLines': -1.0,
 'CategoryEncoded_cluster_group_mean_DebtRatio': -1.0,
 'CategoryEncoded_cluster_group_mean_NumberOfTime30-59DaysPastDueNotWorse': 1.0,
 'CategoryEncoded_cluster_group_mean_NumberOfTime60-89DaysPastDueNotWorse': 1.0,
 'CategoryEncoded_cluster_group_std_NumberOfOpenCreditLinesAndLoans': -1.0,
 'CategoryEncoded_cluster_group_std_NumberRealEstateLoansOrLines': -1.0,
 'DebtRatio': 1.0,
 'DebtRatio MonthlyIncome': 1.0,
 'DebtRatio NumberOfDependents': 1.0,
 'DebtRatio NumberOfOpenCreditLinesAndLoans': -1.0,
 'DebtRatio NumberOfTime30-59DaysPastDueNotWorse': 1.0,
 'DebtRatio NumberRealEstateLoansOrLines': 1.0,
 'DebtRatio RevolvingUtilizationOfUnsecuredLines': 1.0,
 'DebtRatio age': 1.0,
 'DependentsPerAge': 1.0,
 'IncomeOverDebtRatio': -1.0,
 'IncomePerAge': -1.0,
 'IncomePerCreditLineLoan': -1.0,
 'IncomePerDependent': -1.0,
 'IncomePerRealEstateLoan': -1.0,
 'LogDifference30to5960to89': 1.0,
 'MaxTotalTimesPastDue': 1.0,
 'MeanTotalTimesPastDue': 1.0,
 'MonthlyIncome': -1.0,
 'MonthlyIncome NumberOfDependents': -1.0,
 'MonthlyIncome NumberOfOpenCreditLinesAndLoans': -1.0,
 'MonthlyIncome NumberOfTime30-59DaysPastDueNotWorse': 1.0,
 'MonthlyIncome NumberRealEstateLoansOrLines': -1.0,
 'MonthlyIncome RevolvingUtilizationOfUnsecuredLines': 1.0,
 'MonthlyIncome age': -1.0,
 'NumberOfDependents NumberOfOpenCreditLinesAndLoans': 1.0,
 'NumberOfDependents RevolvingUtilizationOfUnsecuredLines': 1.0,
 'NumberOfDependents age': 1.0,
 'NumberOfOpenCreditLinesAndLoans': -1.0,
 'NumberOfOpenCreditLinesAndLoans NumberRealEstateLoansOrLines': 1.0,
 'NumberOfOpenCreditLinesAndLoans RevolvingUtilizationOfUnsecuredLines': 1.0,
 'NumberOfOpenCreditLinesAndLoans age': -1.0,
 'NumberOfTime30-59DaysPastDueNotWorse RevolvingUtilizationOfUnsecuredLines': 1.0,
 'NumberOfTime30-59DaysPastDueNotWorse age': 1.0,
 'NumberOfTimes90DaysLate': 0.0,
 'NumberRealEstateLoansOrLines RevolvingUtilizationOfUnsecuredLines': 1.0,
 'NumberRealEstateLoansOrLines age': -1.0,
 'PercentageNumberTimes30to59PastDue': -1.0,
 'PercentageNumberTimes60to89PastDue': 1.0,
 'PercentageNumberTimes90plusPastDue': 1.0,
 'RealEstateLoansPerCreditLineLoan': 1.0,
 'RevolvingUtilizationOfUnsecuredLines': 1.0,
 'RevolvingUtilizationOfUnsecuredLines age': 1.0,
 'StdTotalTimesPastDue': 1.0,
 'SumTotalTimesPastDuePerCreditLineLoan': -1.0,
 'UtilizationOverDebtRatio': 1.0,
 'age': -1.0,
 'cluster_group': -1.0,
 'prediction_AdaBoostClassifier': 1.0,
 'prediction_MLPClassifier': 1.0,
 'prediction_RandomForestClassifier': 1.0}



base_learners_params = {
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "num_leaves": 39,
    "min_child_samples": 94,
    "subsample": 0.9674035250863153,
    "learning_rate": 0.0088112031800569,
    "colsample_bytree": 0.9750067130759722,
    "lambda_l1": 8.865861216071197,
    "lambda_l2": 0.02873006473756534,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


boruta_learner_params = {
    "boosting_type": "gbdt",
    "n_estimators": 550,
    "num_leaves": 39,
    "min_child_samples": 94,
    "subsample": 0.9674035250863153,
    "learning_rate": 0.01,
    "colsample_bytree": 0.9750067130759722,
    "lambda_l1": 8.865861216071197,
    "lambda_l2": 0.02873006473756534,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}

test_params = {
    "learner_params": {
        "learning_rate": 0.0088112031800569,
        "n_estimators": 2090,
        "extra_params": {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 39,
            "min_child_samples": 94,
            "subsample": 0.9674035250863153,
            "colsample_bytree": 0.9750067130759722,
            "lambda_l1": 8.865861216071197,
            "lambda_l2": 0.02873006473756534,
            "n_jobs": -1,
            "random_state": 42,
            "monotone_constraints": None,
            "verbose": -1,
        },
    }
}

MODEL_PARAMS = {
	'learner_params': {
		'n_estimators': 3073, 
        'learning_rate': 0.0287322553886959,
		'extra_params': {
			'objective': 'binary',
			'metric': 'binary_logloss',
			'boosting_type': 'dart',
			 'lambda_l1': 1.3295519436441523e-07,
             'lambda_l2': 8.532928874611256,
             'num_leaves': 9,
             'feature_fraction': 0.7,
             'bagging_fraction': 0.9638644341977596,
             'bagging_freq': 1,
             'min_child_samples': 20,
			'n_jobs': -1,
			'random_state': 42,
			'monotone_constraints': list(monotone_const_dict.values()),
			'verbose': -1
		}
	}
}


params_original = {
    "learner_params": {
        "learning_rate": 0.005603627873630697,
        "n_estimators": 5926,
        "extra_params": {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            'lambda_l1': 2.1179144447747353e-08,
             'lambda_l2': 3.393255954585769e-08,
             'num_leaves': 18,
             'feature_fraction': 0.58,
             'bagging_fraction': 0.9991117695058381,
             'bagging_freq': 1,
             'min_child_samples': 20,
            "n_jobs": -1,
            "random_state": 42,
            "monotone_constraints": None,
            "verbose": -1,
        },
    }
}

params_ensemble = {
    "learner_params": {
        "learning_rate": 0.005603627873630697,
        "n_estimators": 5926,
        "extra_params": {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
             'lambda_l1': 0.0006278595123135671,
             'lambda_l2': 8.715856738389705,
             'num_leaves': 6,
             'feature_fraction': 0.5,
             'bagging_fraction': 1.0,
             'bagging_freq': 0,
             'min_child_samples': 20,
            "n_jobs": -1,
            "random_state": 42,
            "monotone_constraints": None,
            "verbose": -1,
        },
    }
}

params_fw = {
    "learner_params": {
        "learning_rate": 0.005603627873630697,
        "n_estimators": 5926,
        "extra_params": {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
             'lambda_l1': 4.1326356980799586e-05,
             'lambda_l2': 0.6016241539010313,
             'num_leaves': 18,
             'feature_fraction': 0.4,
             'bagging_fraction': 0.9403276152498784,
             'bagging_freq': 4,
            "n_jobs": -1,
            "random_state": 42,
            "monotone_constraints": None,
            "verbose": -1,
        },
    }
}

params_all = {
    "learner_params": {
        "learning_rate": 0.005603627873630697,
        "n_estimators": 5926,
        "extra_params": {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            'lambda_l1': 4.1326356980799586e-05,
             'lambda_l2': 0.6016241539010313,
             'num_leaves': 18,
             'feature_fraction': 0.4,
             'bagging_fraction': 0.9403276152498784,
             'bagging_freq': 4,
             'min_child_samples': 50,
            "n_jobs": -1,
            "random_state": 42,
            "monotone_constraints": None,
            "verbose": -1,
        },
    }
}

##

##


