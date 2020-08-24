# Clinical predictors of COVID-19 mortality:

The jupyter notebook in this repository is used to reproduce the results for ["Clinical predictors of COVID-19 mortality”](https://www.medrxiv.org/content/10.1101/2020.05.19.20103036v1), which is avaliable on MedRxiv now. 

To cite this paper:
	
	Yadaw, A., Li, Y.c., Bose, S., Iyengar, R., Bunyavanich, S., & Pandey, G. 2020. Clinical predictors of COVID-19 mortality. medRxiv doi:10.1101/2020.05.19.20103036
	
    
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
          
	  
## Dataset:
Due to privacy policy of Mount Sinai Hospital System, we cannot open the covid19 dataset to public. Instead, we are using a subset of public dataset: [UCI Diabetes](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

	Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, “Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.

## Configuration

To analyze your own data with this pipeline, a few variables have to be configured in `config_diabetes.ini`, (or your own configuration file, by editing the 1st line: `config_fn = 'config_diabetes.ini'` of the code cell under `Define filename of configuration and read configuration`). 

## Testing the code
1. Download the code by:

	git clone https://github.com/SBCNY/Clinical-predictors-of-COVID-19-mortality.git

2. Start your jupyter notebook server ([Official tutorial](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html#starting-the-notebook-server))

3. Run the whole notebook (by the fast forward like button ⏩  in the menu bar.

The default configuration is shown as below:

	[FileIO]
	# File path of your data
	file_path = ./diabetes/
	# Directory under 'file_path' containing 'df_train', 'df_test', 'y_train', 'y_test', 'other_test_feature', 'other_test_label'
	data_path = data/
	# Directory for generated figures, which will be created under 'file_path' if not exists
	plot_path = plot/
	# Directory for result files, which will be created under 'file_path' if not exists
	result_path = result/
	# The fileneame of raw data you would like to analyse, which should be put under the directory of'data_path' inside 'file_path'.
	raw_filename = diabetics_small_set.csv
	df_train = train_features_data.csv
	df_test = test_features_data.csv
	y_train = train_labels_data.csv
	y_test = test_labels_data.csv
	rfe_result_csv = RFE_result.csv
	other_test_feature = None
	other_test_label = None
	predicted_score_format = prediction_scores_test{}.csv

	[Preprocessing]
	# The columns you would like to drop
	columns_to_drop = encounter_id, payer_code
	# The column name to distinguish patient
	patient_id_column = patient_nbr
	# The column name you would like to keep with only matched with certain value
	keep_only_matched_patient_column = 
	# Keep only this value of the above column
	keep_only_matched_patient_value = 
	# The column you would like to predict
	outcome_column = readmitted
	outcome_pos_value = >30, <30
	outcome_neg_value = NO
	# These value will be converted to np.nan as missing value
	unknown_value = ?
	# The values you would like to replaced with
	value_replacing_dictionary = {'[0-10)': 5, '[10-20)': 15, '[20-30)':25, '[30-40)': 35, '[40-50)': 45,
				'[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95, 
				'[0-25)': 12.5, '[25-50)': 37.5, '[50-75)': 67.5, '[75-100)': 87.5,
				'[100-125)': 112.5, '[125-150)': 137.5, '[150-175)': 162.5, '[175-200)': 187.5,
				'>200': 212.5}



	[Continuous_feature]
	# Continuous feature of 'df_train'
	continuous_feature = age, weight, time_in_hospital, num_lab_procedures, num_procedures, num_medications, number_outpatient, number_inpatient, number_emergency, number_diagnoses

	[RFE]
	number_of_feature_to_select = ignore
	step_size = 5

	[Model_comparison]
	# Number of subset feature to choose, a parameter for Section 3.2 and Section 4, if there is any update of this parameter, you can rerun the script from Section 3.2
	number_of_subset_features = 6

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
