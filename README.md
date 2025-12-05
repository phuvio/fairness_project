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
Accuracy on test set using NeuralNetwork: 90.69%
Accuracy on test set using Random Forest: 91.26%

Equal Opportunity is the best fairness measure in this case. It shows if all applicants have the same opportunity to receive a loan. It is defined by calculating True Positive Rate (TPR). TPR is the fraction of positive cases which were correctly predicted out of all the positive cases. It is usually referred to as sensitivity or recall, and it represents the probability of the positive subjects to be classified correctly as such. It is given by the formula: TPR = P(prediction = +| actual = +) = TP/(TP + FN).

We tested following priviledges feature:

- whose annual income was in the top 20%
- who are over 40 years old
- top 20% with the longest working career

Equal opportunity for age>40:

- TPR Privileged: 0.9628
- TPR Unprivileged: 0.8991
- Difference (Priv - Unpriv): 0.0638

Equal Opportunity for income_top20:

- TPR Privileged: 0.9478
- TPR Unprivileged: 0.9178
- Difference (Priv - Unpriv): 0.0299

Equal Opportunity for years_employed_top20:

- TPR Privileged: 0.9559
- TPR Unprivileged: 0.9143
- Difference (Priv - Unpriv): 0.0416

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

  ## Report 2

  - explain the research problem and data
  - why we had age>40? as target variable
  - new loss function - explain how and why
  - no random forest
  - results - figure accuracy vs fairness with different grid search values for lambda and epoch

  ## Final report

  5 pages
  - conlusions
