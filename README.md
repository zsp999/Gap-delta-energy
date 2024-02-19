# Gap-Δenergy
Gap-Δenergy, a new metric of the bond energy state, assisting to predict molecular toxicity


First, the preprocess_data folder is where the data is preprocessed. Need to download from the https://hmdb.ca/downloads hmdb_metabolites.xml into the/data folder. Downloaded from https://go.drugbank.com/releases/latest#biotech-sequences SDF format Approved, Experimental, Nutraceutical, Illicit, Withdrawn and Investigational molecules. Downloaded from http://www.t3db.ca/downloads SDF format of toxic molecules. We also provide these documents at https://github.com/zsp999/Gap-Δ-energy/tree/main/data.


process_metabolite_druglike_tox.ipynb preprocesses the above xml or sdf file to obtain a csv file containing small molecule SMILES and class information.


### Preprocessing Data
The preprocess_data directory contains scripts for preprocessing data.

Downloading Required Files:

You need to download hmdb_metabolites.xml from [HMDB website](https://hmdb.ca/downloads) and place it in the /data directory.
Download SDF formatted files for Approved, Experimental, Nutraceutical, Illicit, Withdrawn, and Investigational molecules from [DrugBank](https://go.drugbank.com/releases/latest#biotech-sequences).
Download SDF formatted files for toxic molecules from [T3DB](http://www.t3db.ca/downloads).
Alternatively, you can find these files provided in [this repository](https://github.com/zsp999/Gap-Δ-energy/tree/main/data) except hmdb_metabolites.xml.


The process_metabolite_druglike_tox.ipynb notebook preprocesses the XML or SDF files mentioned above to obtain CSV files containing small molecule SMILES and category information.


### Cal-Δenergey

In the cal-Δenergy section, calculations for gap-Δenergy of molecules are performed.

cal_gapenergy_metabolite_druglike_tox.ipynb: Computes gap-Δenergy for metabolic, drug-like, and toxic small molecules, calculating the complete gap.


cal_gapenergy_tox_exo.ipynb: Calculates Δenergy corresponding to only certain gaps for toxic and exogenous small molecules as features. Generates gap-Δenergy225 using feature engineering. And computes molecular descriptors as features. 


cal_gapenergy_analyze.ipynb: Analyzes the composition of Δenergy for each gap.


### Machine_learning_classfiers

This folder contains classifiers for predicting the toxicity of small molecules using machine learning.

classifiers.ipynb: Trains SVM, RF, XGB, LGBM, KNN, and MLP classifiers using gap-Δenergy225 and molecular descriptors.
Compares their performances.


classifiers_to_select_features.ipynb: Performs feature selection for the XGB classifier to identify useful features from gap-Δenergy and molecular descriptors.


classifiers_using_selectedfeatures.ipynb: Trains XGB models on the selected features. Obtains toxicity probability predictions for gap-Δenergy and molecular descriptors.


classifiers_finalstacking_using_proba_LR.ipynb: Utilizes the probabilities obtained in the previous step to train an LR classifier for a better model.


### Apply

apply.ipynb: Input the molecules you want to calculate into the your_smis list, run the notebook, and you will obtain the toxicity probabilities predicted by the final XGB classifier and LR classifier.