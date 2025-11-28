# Fairness project

Fairness project for the University of Helsinki

Dataset: Bangladeshi University Students Mental Health

Dataset: Medical Insurance Cost Prediction
[https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction?resource=download](https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction?resource=download)

Dataset: Realistic Loan Approval Dataset | US & Canada
[https://www.kaggle.com/datasets/parthpatel2130/realistic-loan-approval-dataset-us-and-canada?select=Loan_approval_data_2025.csv](https://www.kaggle.com/datasets/parthpatel2130/realistic-loan-approval-dataset-us-and-canada?select=Loan_approval_data_2025.csv)

Unlike most loan datasets available online, this one is built on real banking criteria from US and Canadian financial institutions. Drawing from 3 years of hands-on finance industry experience, the dataset incorporates realistic correlations and business logic that reflect how actual lending decisions are made. This makes it perfect for data scientists looking to build portfolio projects that showcase not just coding ability, but genuine understanding of credit risk modelling.

Target feature is 'loan_status'.

- Total Records: 50,000
- Features:  20 (customer_id + 18 predictors + 1 target)
- Target Distribution: 55% Approved, 45% Rejected
- Missing Values: 0 (Complete dataset)
- Product Types: Credit Card, Personal Loan, Line of Credit
- Market:  United States & Canada
- Use Case: Binary Classification (Approved/Rejected)

Result:
Accuracy on test set using NeuralNetwork: 90.79%
Accuracy on test set using Random Forest: 91.42%

For AIF360 to work properly, need to install: aif360, 'aif360[Reductions]', 'aif360[inFariness]'.

Results for measuring fairness:

====== AGE>40 ======
- SPD: -0.2549
- DI: 0.6492
- Mean Diff: -0.2549

Strong unfairness

Younger applicants have 25% lower change of getting loan.

====== INCOME_TOP20 ======
- SPD: -0.1535
- DI: 0.7719
- Mean Diff: -0.1535

Moderate unfairness

People outside the top 20% have 15% lower change of getting loan.

====== YEARS_EMPLOYED_TOP20 ======
- SPD: -0.199
- DI: 0.7196
- Mean Diff: -0.19

Strong unfairness

People outside the top 20% have 20% lower change of getting loan.

- analyze data
  - value counts
  - unique values of different columns
  - check if some fairness curations already done - AIF360
- define target value
  - panic attack (after long and hard consideration this is our choice)
  - depression
  - seek treatment
- define fairness issues
  - gender issue
  - financial issue
  - regional issue
- create basic model
  - neural network
  - random forest
  - logistic regression
- mitigate fairness issues

  ## Report 1

  - explain data
    - which dataset and what is it about
    - value counts
    - plots of distributions
    - target value
  - fairness issue
