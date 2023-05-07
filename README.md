# Bayesian Mortality Prediction with Sensitivity Analysis

Patients admitted to the hospital would ideally be returned to good health and discharged in an appropriately timely manner. To make such a decision, it would be reasonable to desire a decision rule on when to discharge a patient based on their predicted likelihood to die.

In this case, we reviewed a dataset from a 2015 study from the Institute of Cardiology and Allied Hospital in Pakistan where 299 patients were admitted with heart failure and roughly 1/3 of the patients had died post discharge. Certain medical history measurements were provided for each patient in the form of age, sex, smoking status, diabetes status, and anaemia status. Along with this, measurements of the patients' ejection fraction, serum creatinine, blood pressure, serum sodium, platelets, and creatinine phosphokinase were also included. For some background, serum creatinine was used as a measurement of kidney performance and ejection fraction was used to measure heart performance. In this paper we will focus on age, smoking status, diabetes status, anaemia status, ejection fraction, and serum creatinine as those were identified as significant in the reviewed literature.

We reviewed two papers, "Survival analysis of heart failure patients: A case study", Ahmad et al. (2017) and "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone", Chicco et al. (2020)) who had also taken a look at this data set. Ahmad et al. had applied cox regression proportional hazard model and used out of sample AUC in order to measure the model. They found that age, serum creatinine, blood pressure, ejection fraction, and anaemia were significant factors and their model had an out of sample AUC of 0.81. Chicco et al. had applied a large variety of machine learning methods including Random Forest, SVM, XGBOOST, etc. However, they found that the Random Forest did the best with an out of Sample AUC of around 0.80 and identified that serum creatinine and ejection fraction were significant predictors of mortality.

Based on the above readings, we had noted the large variation in methodological approaches and resulting dierence in model parameters selected. We decided to take a Bayesian approach to the dataset and apply Markov Chain Monte Carlo (MCMC) methods to analyze mortality predictions. In order to compare our models, we decided to use out of sample AUC as the other authors also used this approach.
