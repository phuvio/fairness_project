# Fairness project

Fairness project for the University of Helsinki

Dataset: Bangladeshi University Students Mental Health

Dataset: Medical Insurance Cost Prediction
[https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction?resource=download](https://www.kaggle.com/datasets/mohankrishnathalla/medical-insurance-cost-prediction?resource=download)

This dataset was created by Mohan Krishna Thalla and it provides information about 100,000 individuals including their demographics, socioeconomic status, health conditions, lifestyle factors, insurance plans, and medical expenditures. It consists of 100,000 rows and 54 features. Columns person_id and risk_score dropped. Target feature is is_high_risk.

Used RidgeCV to select most important features. 20 features out of 70 selected. Selected features:
'age', 'bmi', 'annual_medical_cost', 'avg_claim_amount', 'total_claims_paid', 'chronic_count', 'hypertension', 'diabetes', 'asthma', 'copd', 'cardiovascular_disease', 'cancer_history', 'kidney_disease', 'liver_disease', 'arthritis', 'mental_health', 'proc_surgery_count', 'had_major_procedure', 'smoker_Former', 'smoker_Never'

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
