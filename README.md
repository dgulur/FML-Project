# FML-Project

This repo is divided into 4 sections - each for one classifier 
* Naive Bayes 
* Logistic Regression
* SVM - Support Vector Machines
* CNN - Convolution Neural Networks

In order to run any python file you will require:
* OCT Data Set Images 
* df_prime_train.csv - Contains Training Data 
* df_prime_test.csv - Contains Test Data 

The csv files can also be found in this repo

Use this format to run the file: 
python <file_name>.py --annot_train_prime <path_to_training_data> --annot_test_prime <path_to_test_data> --data_root <path_to_OLIVES_PRIME_FULL_Dataset>

Here is an example 
python resnet.py --annot_train_prime df_prime_train.csv --annot_test_prime df_prime_test.csv --data_root D:\FML-Project/OLIVES/OLIVES/Prime_FULL
