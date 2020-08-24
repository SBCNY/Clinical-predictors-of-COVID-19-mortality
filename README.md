# Clinical predictors of COVID-19 mortality:

The jupyter notebook in this repository is used to reproduce the results for ["Clinical predictors of COVID-19 mortality”](https://www.medrxiv.org/content/10.1101/2020.05.19.20103036v1), which is avaliable on MedRxiv now. To cite this paper:

Yadaw, A., Li, Y.C., Bose, S., Iyengar, R., Bunyavanich, S., & Pandey, G. 2020. Clinical predictors of COVID-19 mortality. medRxiv doi:10.1101/2020.05.19.20103036
	
    
## Setup environments
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
Due to privacy policy of Mount Sinai Hospital System, we cannot open the covid19 dataset to public. Instead, we are using a subset of public dataset: [UCI Diabetes](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008). (B. Strack et al., “Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.)

## Configuration

To analyze your own data with this pipeline, a few variables have to be configured in `config_diabetes.ini`, (or your own configuration file, by editing the 1st line: `config_fn = 'config_diabetes.ini'` of the code cell under `Define filename of configuration and read configuration`). 

## Testing the code
* 1. Download the code by:
		
		`git clone https://github.com/SBCNY/Clinical-predictors-of-COVID-19-mortality.git`

* 2. Start your jupyter notebook server ([Official tutorial](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html#starting-the-notebook-server)) and open `Clinical_predictor_notebook.ipynb`.


* 3. Run the whole notebook (by the fast forward like button⏩  in the toolbar.)

## COVID-19 Data: 

Anonymized electronic medical record (EMR) data from patients diagnosed with COVID-19 within the Mount Sinai Hospital System, New York, NY from March 9th through April 6th, 2020 were included in the study. On 6th April, 5051 COVID-19 positive patients treated at the Mount Sinai Health System, data split in training set and test set.

## Pipeline Summary
### 1. Preprocessing
This part is converting the raw data to be capable in the following steps:
* Irrelevant columns defined in `[Preprocessing]columns_to_drop` of `config_diabetes.ini` will be dropped out. 
* Column defined in `[Preprocessing]patient_id_column` is used as patient index, to remove duplicate patient entry. 
* If you would like to keep only patients matching with a value of a column, you can define `keep_only_matched_patient_column` and `keep_only_matched_patient_value` (which are blanks under the configuration of this dataset). 
* The outcome array will be pulled out, by the variable `outcome_column`, and binarized by matching the value of `outcome_pos_value` and `outcome_neg_value`  defined respectively in the configuration file. 
* For the strings which are considered as missing value, can be defined `unknown_value`. We are also able to replace the value to desired value globally by dictionary defined in configuration by `value_replacing_dictionary`.
 * After these steps, label encoding will be performed to categorical columns. Finally, the preprocessed dataset will be splited into training (development set)and testing set (80% and 20% respectively). 

### 2. Missing Value Imputation

* We have first attempted to find the optimal percentage of missing values in each variable across the patients in the development set (missing value level) that could be imputed and lead to more accurate prediction. In this step, we have pre-processed the development set data and split it randomly 100 times into training and validation set. Used four different classifiers (Random Forest, Logistic Regression, Support Vector Machine and XGBoost) with increment of 5% missing value imputation. The performance is visualized at the bottom of the notebook (figure 2A)
 
### 3. Recursive Feature Elimination (RFE) model building 

* We used a setup analogous to missing value imputation, and the Recursive Feature Elimination (RFE) algorithm, we evaluated the performance of the four classification algorithms with different number of features selected from the full set of features. 

* The list of number of features can be defined in two ways: 
  * a. `[RFE]number_of_feature_to_select` which should be a list of number, separated by `,`. If you are going to use this way, please set it to be `ignore`
  * b. `[RFE]step_size` which should be a integer, as the step size of list from 1 to total number of features. If you are going to use this way, please set it to be `ignore`

* The average AUC scores from 100 runs of this process are shown in figure 2B, along with error bars. 
 
### 4. Model Testing
2 Models will be tested, the (XGBoost) classifier trained by full set of features, and the subset of features ('n' features selected by RFE, where 'n' is defined in `[Model_comparison]number_of_subset_features`.) ROC curve (and its AUC) and calibration curve of testing set can be visualized in figure 3.
    
