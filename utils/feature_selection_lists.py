fw_features = ['CategoryEncoded_MonthlyIncome_max_DebtRatio',
 'CategoryEncoded_MonthlyIncome_max_NumberOfTime30-59DaysPastDueNotWorse',
 'CategoryEncoded_MonthlyIncome_mean_NumberOfOpenCreditLinesAndLoans',
 'CategoryEncoded_MonthlyIncome_mean_RevolvingUtilizationOfUnsecuredLines',
 'CategoryEncoded_MonthlyIncome_std_NumberOfTime30-59DaysPastDueNotWorse',
 'CategoryEncoded_NumberOfDependents_max_NumberOfOpenCreditLinesAndLoans',
 'CategoryEncoded_NumberOfDependents_max_RevolvingUtilizationOfUnsecuredLines',
 'CategoryEncoded_age_max_DebtRatio',
 'CategoryEncoded_age_max_RevolvingUtilizationOfUnsecuredLines',
 'CategoryEncoded_age_mean_DebtRatio',
 'CategoryEncoded_age_mean_NumberOfTime30-59DaysPastDueNotWorse',
 'CategoryEncoded_age_std_NumberOfOpenCreditLinesAndLoans',
 'CategoryEncoded_age_std_NumberRealEstateLoansOrLines',
 'CategoryEncoded_cluster_group_max_NumberOfTimes90DaysLate',
 'CategoryEncoded_cluster_group_std_DebtRatio',
 'CategoryEncoded_cluster_group_std_RevolvingUtilizationOfUnsecuredLines',
 'DebtRatio',
 'DebtRatio MonthlyIncome',
 'DebtRatio NumberOfDependents',
 'DebtRatio NumberOfOpenCreditLinesAndLoans',
 'DebtRatio NumberOfTime30-59DaysPastDueNotWorse',
 'DebtRatio NumberRealEstateLoansOrLines',
 'DebtRatio RevolvingUtilizationOfUnsecuredLines',
 'DebtRatio^2',
 'DependentsPerAge',
 'IncomeOverDebtRatio',
 'IncomePerCreditLineLoan',
 'IncomePerDependent',
 'IncomePerRealEstateLoan',
 'MaxTotalTimesPastDue',
 'MeanTotalTimesPastDue',
 'MonthlyIncome NumberOfDependents',
 'MonthlyIncome NumberOfTime30-59DaysPastDueNotWorse',
 'MonthlyIncome age',
 'NumberOfDependents RevolvingUtilizationOfUnsecuredLines',
 'NumberOfDependents age',
 'NumberOfOpenCreditLinesAndLoans NumberRealEstateLoansOrLines',
 'NumberOfOpenCreditLinesAndLoans age',
 'NumberOfTime30-59DaysPastDueNotWorse RevolvingUtilizationOfUnsecuredLines',
 'NumberRealEstateLoansOrLines RevolvingUtilizationOfUnsecuredLines',
 'NumberRealEstateLoansOrLines age',
 'PercentageNumberTimes30to59PastDue',
 'PercentageNumberTimes90plusPastDue',
 'RealEstateLoansPerCreditLineLoan',
 'RevolvingUtilizationOfUnsecuredLines^2',
 'StdTotalTimesPastDue',
 'SumTotalTimesPastDuePerCreditLineLoan',
 'UtilizationOverDebtRatio',
 'age',
 'cluster_group',
 'prediction_AdaBoostClassifier',
 'prediction_MLPClassifier',
 'prediction_RandomForestClassifier']


boruta_features = ['CategoryEncoded_MonthlyIncome_max_DebtRatio',
 'CategoryEncoded_MonthlyIncome_max_RevolvingUtilizationOfUnsecuredLines',
 'CategoryEncoded_age_max_DebtRatio',
 'CategoryEncoded_age_max_NumberOfOpenCreditLinesAndLoans',
 'CategoryEncoded_age_max_NumberRealEstateLoansOrLines',
 'CategoryEncoded_age_mean_NumberOfTime30-59DaysPastDueNotWorse',
 'CategoryEncoded_age_mean_RevolvingUtilizationOfUnsecuredLines',
 'CategoryEncoded_cluster_group_max_DebtRatio',
 'CategoryEncoded_cluster_group_max_NumberOfOpenCreditLinesAndLoans',
 'CategoryEncoded_cluster_group_max_NumberOfTime30-59DaysPastDueNotWorse',
 'CategoryEncoded_cluster_group_max_NumberOfTime60-89DaysPastDueNotWorse',
 'CategoryEncoded_cluster_group_max_NumberOfTimes90DaysLate',
 'CategoryEncoded_cluster_group_max_RevolvingUtilizationOfUnsecuredLines',
 'CategoryEncoded_cluster_group_mean_DebtRatio',
 'CategoryEncoded_cluster_group_mean_NumberOfTime30-59DaysPastDueNotWorse',
 'CategoryEncoded_cluster_group_mean_NumberOfTime60-89DaysPastDueNotWorse',
 'CategoryEncoded_cluster_group_std_NumberOfOpenCreditLinesAndLoans',
 'CategoryEncoded_cluster_group_std_NumberRealEstateLoansOrLines',
 'DebtRatio',
 'DebtRatio MonthlyIncome',
 'DebtRatio NumberOfDependents',
 'DebtRatio NumberOfOpenCreditLinesAndLoans',
 'DebtRatio NumberOfTime30-59DaysPastDueNotWorse',
 'DebtRatio NumberRealEstateLoansOrLines',
 'DebtRatio RevolvingUtilizationOfUnsecuredLines',
 'DebtRatio age',
 'DependentsPerAge',
 'IncomeOverDebtRatio',
 'IncomePerAge',
 'IncomePerCreditLineLoan',
 'IncomePerDependent',
 'IncomePerRealEstateLoan',
 'LogDifference30to5960to89',
 'MaxTotalTimesPastDue',
 'MeanTotalTimesPastDue',
 'MonthlyIncome',
 'MonthlyIncome NumberOfDependents',
 'MonthlyIncome NumberOfOpenCreditLinesAndLoans',
 'MonthlyIncome NumberOfTime30-59DaysPastDueNotWorse',
 'MonthlyIncome NumberRealEstateLoansOrLines',
 'MonthlyIncome RevolvingUtilizationOfUnsecuredLines',
 'MonthlyIncome age',
 'NumberOfDependents NumberOfOpenCreditLinesAndLoans',
 'NumberOfDependents RevolvingUtilizationOfUnsecuredLines',
 'NumberOfDependents age',
 'NumberOfOpenCreditLinesAndLoans',
 'NumberOfOpenCreditLinesAndLoans NumberRealEstateLoansOrLines',
 'NumberOfOpenCreditLinesAndLoans RevolvingUtilizationOfUnsecuredLines',
 'NumberOfOpenCreditLinesAndLoans age',
 'NumberOfTime30-59DaysPastDueNotWorse RevolvingUtilizationOfUnsecuredLines',
 'NumberOfTime30-59DaysPastDueNotWorse age',
 'NumberOfTimes90DaysLate',
 'NumberRealEstateLoansOrLines RevolvingUtilizationOfUnsecuredLines',
 'NumberRealEstateLoansOrLines age',
 'PercentageNumberTimes30to59PastDue',
 'PercentageNumberTimes60to89PastDue',
 'PercentageNumberTimes90plusPastDue',
 'RealEstateLoansPerCreditLineLoan',
 'RevolvingUtilizationOfUnsecuredLines',
 'RevolvingUtilizationOfUnsecuredLines age',
 'StdTotalTimesPastDue',
 'SumTotalTimesPastDuePerCreditLineLoan',
 'UtilizationOverDebtRatio',
 'age',
 'cluster_group',
 'prediction_AdaBoostClassifier',
 'prediction_MLPClassifier',
 'prediction_RandomForestClassifier']




ensemble_features = ['CategoryEncoded_MonthlyIncome_max_DebtRatio',
 'CategoryEncoded_MonthlyIncome_max_NumberOfTime30-59DaysPastDueNotWorse',
 'CategoryEncoded_MonthlyIncome_max_RevolvingUtilizationOfUnsecuredLines',
 'CategoryEncoded_MonthlyIncome_mean_NumberOfOpenCreditLinesAndLoans',
 'CategoryEncoded_MonthlyIncome_mean_RevolvingUtilizationOfUnsecuredLines',
 'CategoryEncoded_MonthlyIncome_std_NumberOfTime30-59DaysPastDueNotWorse',
 'CategoryEncoded_NumberOfDependents_max_NumberOfOpenCreditLinesAndLoans',
 'CategoryEncoded_NumberOfDependents_max_RevolvingUtilizationOfUnsecuredLines',
 'CategoryEncoded_age_max_DebtRatio',
 'CategoryEncoded_age_max_NumberOfOpenCreditLinesAndLoans',
 'CategoryEncoded_age_max_NumberRealEstateLoansOrLines',
 'CategoryEncoded_age_max_RevolvingUtilizationOfUnsecuredLines',
 'CategoryEncoded_age_mean_DebtRatio',
 'CategoryEncoded_age_mean_NumberOfTime30-59DaysPastDueNotWorse',
 'CategoryEncoded_age_mean_RevolvingUtilizationOfUnsecuredLines',
 'CategoryEncoded_age_std_NumberOfOpenCreditLinesAndLoans',
 'CategoryEncoded_age_std_NumberRealEstateLoansOrLines',
 'CategoryEncoded_cluster_group_max_DebtRatio',
 'CategoryEncoded_cluster_group_max_NumberOfOpenCreditLinesAndLoans',
 'CategoryEncoded_cluster_group_max_NumberOfTime30-59DaysPastDueNotWorse',
 'CategoryEncoded_cluster_group_max_NumberOfTime60-89DaysPastDueNotWorse',
 'CategoryEncoded_cluster_group_max_NumberOfTimes90DaysLate',
 'CategoryEncoded_cluster_group_max_RevolvingUtilizationOfUnsecuredLines',
 'CategoryEncoded_cluster_group_mean_DebtRatio',
 'CategoryEncoded_cluster_group_mean_NumberOfTime30-59DaysPastDueNotWorse',
 'CategoryEncoded_cluster_group_mean_NumberOfTime60-89DaysPastDueNotWorse',
 'CategoryEncoded_cluster_group_std_DebtRatio',
 'CategoryEncoded_cluster_group_std_NumberOfOpenCreditLinesAndLoans',
 'CategoryEncoded_cluster_group_std_NumberRealEstateLoansOrLines',
 'CategoryEncoded_cluster_group_std_RevolvingUtilizationOfUnsecuredLines',
 'DebtRatio',
 'DebtRatio MonthlyIncome',
 'DebtRatio NumberOfDependents',
 'DebtRatio NumberOfOpenCreditLinesAndLoans',
 'DebtRatio NumberOfTime30-59DaysPastDueNotWorse',
 'DebtRatio NumberRealEstateLoansOrLines',
 'DebtRatio RevolvingUtilizationOfUnsecuredLines',
 'DebtRatio age',
 'DebtRatio^2',
 'DependentsPerAge',
 'IncomeOverDebtRatio',
 'IncomePerAge',
 'IncomePerCreditLineLoan',
 'IncomePerDependent',
 'IncomePerRealEstateLoan',
 'LogDifference30to5960to89',
 'MaxTotalTimesPastDue',
 'MeanTotalTimesPastDue',
 'MonthlyIncome',
 'MonthlyIncome NumberOfDependents',
 'MonthlyIncome NumberOfOpenCreditLinesAndLoans',
 'MonthlyIncome NumberOfTime30-59DaysPastDueNotWorse',
 'MonthlyIncome NumberRealEstateLoansOrLines',
 'MonthlyIncome RevolvingUtilizationOfUnsecuredLines',
 'MonthlyIncome age',
 'NumberOfDependents NumberOfOpenCreditLinesAndLoans',
 'NumberOfDependents RevolvingUtilizationOfUnsecuredLines',
 'NumberOfDependents age',
 'NumberOfOpenCreditLinesAndLoans',
 'NumberOfOpenCreditLinesAndLoans NumberRealEstateLoansOrLines',
 'NumberOfOpenCreditLinesAndLoans RevolvingUtilizationOfUnsecuredLines',
 'NumberOfOpenCreditLinesAndLoans age',
 'NumberOfTime30-59DaysPastDueNotWorse RevolvingUtilizationOfUnsecuredLines',
 'NumberOfTime30-59DaysPastDueNotWorse age',
 'NumberOfTimes90DaysLate',
 'NumberRealEstateLoansOrLines RevolvingUtilizationOfUnsecuredLines',
 'NumberRealEstateLoansOrLines age',
 'PercentageNumberTimes30to59PastDue',
 'PercentageNumberTimes60to89PastDue',
 'PercentageNumberTimes90plusPastDue',
 'RealEstateLoansPerCreditLineLoan',
 'RevolvingUtilizationOfUnsecuredLines',
 'RevolvingUtilizationOfUnsecuredLines age',
 'RevolvingUtilizationOfUnsecuredLines^2',
 'StdTotalTimesPastDue',
 'SumTotalTimesPastDuePerCreditLineLoan',
 'UtilizationOverDebtRatio',
 'age',
 'cluster_group',
 'prediction_AdaBoostClassifier',
 'prediction_MLPClassifier',
 'prediction_RandomForestClassifier']