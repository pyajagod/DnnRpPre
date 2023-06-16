# DnnRpPre
A predictive model for the relationship between RNA expression and surface protein levels.

## DNNRpPre
This is a prediction model based on DNN model, using the framework is keras, based on python3.8.

### class MetrixAndLossFunction
This is a class for model evaluation and contains several evaluation functions that can be used, which can be selected according to your requirements.

### class DataProcessing
This is a data preprocessing class that you will need to modify based on your own data when using the model.

### class DnnRpPre
This is the main model class, where the model has been defined as a hyperparametric search. You need the TUNE value in the ifOrNotTune function to choose whether to perform a hyperparametric search, which is generally required the first time you use the model.

