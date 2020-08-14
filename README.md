# Clinical predictors of COVID-19 mortality:

The scripts in this repository is used to reproduce the results for ["Clinical predictors of COVID-19 mortality”](https://www.medrxiv.org/content/10.1101/2020.05.19.20103036v1). 

To cite this paper:
	
	Yadaw, A., Li, Y.c., Bose, S., Iyengar, R., Bunyavanich, S., & Pandey, G. 2020. Clinical predictors of COVID-19 mortality. medRxiv doi:10.1101/2020.05.19.20103036
	
Program files are shared in the folders by figures name (Figure 2, Figure 3, and Figure 4). 
    
## Setup environments: 
This project is developed in Jupyter Notebook environment. So the following are required:

	Python 3.7.3
	Pandas 0.24.2
	NumPy 1.16.2
	Scikit-learn 0.20.3
	Scipy 1.2.1
	Matplotlib 3.1.1
	seaborn 0.9.0
	xgboost 0.90
          
	  
## Configuration

To analyze your own data with this pipeline, a few variables have to be configured in `config.ini`. The default configuration is shown as below:

	[FileIO]
	df_train = train_features_data.csv
	df_test = test_features_data.csv
	y_train = train_labels_data.csv
	y_test = test_labels_data.csv
	predicted_scores = Predicted_&_actual_scores_3f_17F_test1.csv, Predicted_&_actual_scores_3f_17F_test2.csv
	rfe_result_csv = RFE_result_RL.csv

	[Continuous_feat]
	continuous_feat = AGE, BMI, TEMPERATURE, TEMP_MAX, SYSTOLIC_BP,DIASTOLIC_BP, O2_SAT, O2SAT_MIN

### Under the section of `[FileIO]`:

`df_train` is the file containing the features of training data (not including the outcome/label), which is in csv format

`df_test` is the file containing the features of testing data (not including the outcome/label), which is in csv format

`y_train` is the file containing the outcome/label of training data, which is in csv format

`y_test` is the file containing the outcome/label of testing data, which is in csv format

`predicted_scores` is the filename you would like to output the prediction scores of testing data to, where the number of filenames should be equal to the testing data you put, and format in csv

`rfe_result_csv` is the filename you would like to output the result of recursive feature elimination


### Under the section of `[Continuous_feat]`:

`continuous_feat` is the list of continuous feature(s), since we will impute the continuous feature and categorical feature in different way.

## COVID-19 Data: 

Anonymized electronic medical record (EMR) data from patients diagnosed with COVID-19 within the Mount Sinai Hospital System, 
New York, NY from March 9th through April 6th, 2020 were included in the study. On 6th April, 5051 COVID-19 positive patients 
treated at the Mount Sinai Health System, data split in training set and test set.

## Missing Value Imputation: 

In figure 2A, we have first attempted to find the optimal percentage of missing values in each variable across the patients in 
the development set (missing value level) that could be imputed and lead to more accurate prediction. In this script, we have 
pre-processed the development set data and split it randomly 100 times into training and validation set. Used four different 
classifiers (Random Forest, Logistic Regression, Support Vector Machine and XGBoost) with increment of 5% missing value imputation.
 
## RFE model: 

In figure 2B, we used a setup analogous to missing value imputation, and the Recursive Feature Elimination (RFE) algorithm, 
we evaluated the performance of the four classification algorithms with different number of features selected from the full
set of features. The average AUC scores from 100 runs of this process are shown here, along with error bars.
 
## Folder Figure 3: 

Figure 3 script is self-descriptive. Here, we have plotted AUC curve based on optimum imputation features (17F) model and 
top three features from RFE 3F model. Also calibration of the plot has been done by using ‘calibration_curve’ library from sklearn.
    
## Folder Figure 4: 

Figure 4 script is self-descriptive, here we have ran RFE simulation simulations for top three features. This script generate
three independent hundred simulation of RFE model for four classifiers by using 17 features and selected top three features and
saved in a csv file and plot horizontal bar graphs. One can develop figure 4A of the paper by choosing single seed of 300 runs 
or by summation of three runs output data.

Characteristic table p-value and odd ratio calculation is provided by (‘table_pval_effect_size.py’), we used Cohen’s_d to 
determine the effect size of the experiment and ttest to calculate p-value.  
To cite this paper (doi: https://doi.org/10.1101/2020.05.19.20103036)
