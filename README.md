The code for this article (citation below) is provided as a jupyter notebook to enable users to apply our methodology to clinical datasets.

Yadaw, A., Li, Y.C., Bose, S., Iyengar, R., Bunyavanich, S., & Pandey, G. 2020. Clinical predictors of COVID-19 mortality. The Lancet Digital Health, 2(10):e516-e525, doi:https://doi.org/10.1016/S2589-7500(20)30217-X.

The notebook as well as associated files have detailed embedded comments that should assist users in using this code for their respective dataset/study.
    
## Setup environment
The following Python version and packages are required to execute this notebook:

	Python 3.7.3
	Pandas 0.24.2
	NumPy 1.16.2
	Scikit-learn 0.20.3
	Scipy 1.2.1
	Matplotlib 3.1.1
	seaborn 0.9.0
	xgboost 0.90
          
	  
## Sample data
Due to IRB constraints, we are unable to publicly share the COVID-19 EHR dataset used in our study. Thus, as sample data to test our code, we are providing a randomly selected subset (5,000 patients; readmission/data/diabetics_small_set.csv) of the publicly available [Hospital readmission](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008) dataset, which is of a similar nature as our original dataset.

## Configuration

<!--To analyze your own data with this pipeline, a few variables have to be configured in `config_diabetes.ini`, (or your own configuration file, by editing the 1st line: `config_fn = 'config_diabetes.ini'` of the code cell under `Define filename of configuration and read configuration`).-->

EHR-derived datasets are generally a mix of continuous and categorical variables, both of which may contain missing values. Our code is designed for converting categorical variables into numerical ones using [label encoding] (https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd). These variables, as well as other necessary operations that need to be carried out before our methodology is applied, need to be specified in the config.ini file. A sample file for the example readmission dataset is provided in the repo. More details are provided in the summary of the various analysis steps below.

## Applying the code
* Download the code using the download button or the command:
		
		git clone https://github.com/SBCNY/Clinical-predictors-of-COVID-19-mortality.git

* Start your jupyter notebook server ([Official tutorial](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html#starting-the-notebook-server)) and open `Clinical_predictor_notebook.ipynb`.


* The notebook can be run by using the fast forward like button⏩  in the toolbar for the hospital readmission dataset.


## Pipeline (notebook) Summary
### 0. Configuration (before running the pipeline)
The configuration file `config.ini` enables the user to define variables (eg. filename, specific file path, outcome variable, etc.) outside the notebook.
* The configuration file is separated into several sections: `[FileIO]`, `[Preprocessing]`, `[Continuous_feature]`, `[RFE]`, `[Model_comparison]`.
  * `[FileIO]` contains the path and filename of the data etc.
  * `[Preprocessing]` contains the variables needed in the preprocessing steps.
  * `[Continuous_feature]` contains the list of features that you would like to consider as continuous, since continuous and categorical features are handled in different ways in the analysis code.
  * `[RFE]` has two options for the number of features to be selected. More details are provided in section 3 of this summary.
  * `[Model_comparison]` contains the final number of top selected features you would like to include in the reduced model based on the results of the feature selection analysis. This is defined as 6 for the sample readmission data.

### 1. Preprocessing
This part pre-processes the raw data through the following steps:
* For columns irrelevant to the outcome, except patient ID, should be defined in `[Preprocessing]columns_to_drop` of `config.ini`, which will be dropped out in our analysis (e.g. ID of medical insurance company). 
* The column defined in `[Preprocessing]patient_id_column` is used as the patient index, which is used in turn to remove duplicate patient entries. 
* If you are only interested in particular groups of patients, like in our paper focusing on patients detected with COVID-19 only, you can define `keep_only_matched_patient_column` and `keep_only_matched_patient_value` (which are blanks in the configuration file of the readmission dataset). 
* The outcome array will be pulled from the variable `outcome_column`, and binarized by matching the value of `outcome_pos_value` and `outcome_neg_value`  defined respectively in the configuration file. 
* The strings that denote missing values can be defined in the `unknown_value` variable in the configuration value. We are also able to replace the value to desired value globally by dictionary defined in configuration by `value_replacing_dictionary`.
* After these steps, label encoding will be applied to the categorical columns. 
* Finally, the preprocessed dataset will be splited into the development and test sets in a 75:25 ratio. 

### 2. Missing Value Imputation

We first attempt to find the percentage of missing values in each variable across the patients in the development set (missing value level) that can be reliably imputed and lead to more accurate prediction. In this step, we split the development set data randomly 100 times into training and validation set for training and evaluating candidate classifiers at different missing value levels. Four classification algorithms (Random Forest, Logistic Regression, Support Vector Machine and XGBoost) are tested at increments of 5% missing value levels. The final performance can be visualized in a figure generated at the bottom of this code block of the notebook, analogous to Figure 2A in our article.
 
### 3. Feature selection using Recursive Feature Elimination (RFE) 

* We use a setup analogous to missing value imputation, and the Recursive Feature Elimination (RFE) algorithm, to evaluate the performance of the four classification algorithms listed above with different number of features selected from the full set of features. 

* The list of number of features can be defined in two ways in the configuration file: 
  * a. `[RFE]number_of_feature_to_select`, which should be the list of numbers of features you wish to test, separated by `,`. If you are not using this option, please set it to `ignore`.
  * b. `[RFE]step_size`, which should be a integer, as the step size of list from 1 to total number of features. If you are not using this option, please set it to `ignore`.

* The average AUC scores from 100 runs of this process, along with error bars, are shown in the figure at the bottom of this code block, analogous to Figure 2B in our article. 
 
### 4. Model Testing
2 Models will be tested, namely the (XGBoost) classifier trained by full set of features, and the subset of features ('n' features selected by RFE, where 'n' is defined in `[Model_comparison]number_of_subset_features`.). The resultant ROC curve (and its AUC score) and calibration curve (and its slope and intercept) on the test set is generated at the bottom of this code block, generating plots analogous to those in Figure 3 in our article.
    
    
## Contact
In case of issues with the code, please let us know through the Issues functionality and/or contact Arjun S. Yadaw (arjun.yadaw@mssm.edu), Yan-chak Li (yan-chak.li@mssm.edu) and Gaurav Pandey (gaurav.pandey@mssm.edu).
