from Logger import Log
from MLModeler import Modeler 
from Tester import Test 
from Sampler import Sample
import copy
import pandas as pd 
import numpy as np 
import os

#Change working directory to import data
os.chdir("/Users/Sam/Documents/Depauw/04 Senior Year/Semester 2/Math_Senior_Seminar/Data")

#Read in fraud data
fraud_data = pd.read_csv("creditcard.csv")

# Master Log for entire experimental study metadata. 
MasterLog = Log("Test_Master_Log", "Test_Results_Log", 

									#Masterlog column names
								  [[
								  	#Execution Metadata
								    "Execution_Date",
									"Execution_Time",
									"Sampler_Duration_Sec",
									"Modeler_Duration_Sec",
									"Execution_Duration_Sec",
									#5
									

									#Sampler rows
									"Sample_Method",
									"Total_Row_Num",
									"NonFraud_Row_Num",
									"Fraud_Row_Num",
									"Fraud_Orig_Row_Num",
									"Fraud_Synth_Row_Num",
									#6

									#Modeler Meta-parameters
									"Models_Used",
									"SVM_Params",
									"RF_Estimators",
									"KNN_Neighbors",
									"Monte_Carlo_Bool",
									"Monte_Carlo_Samp_Size",
									"Kfold_Bool", "Kfold_Num",
									"Precision_Wt", "Recall_Wt",
									#10

									#Modeler SVM Performance
									"SVM_TP", "SVM_FP", "SVM_TN", "SVM_FN",
									"SVM_Pos_Precision", "SVM_Neg_Precision",
									"SVM_Pos_Recall", "SVM_Neg_Recall",
									"SVM_Precision", "SVM_Recall",
									"SVM_Accuracy", "SVM_F_Measure",
									#12

									#Model Random Forest Performance   
									"RF_TP", "RF_FP", "RF_TN", "RF_FN",
									"RF_Pos_Precision", "RF_Neg_Precision",
									"RF_Pos_Recall", "RF_Neg_Recall",
									"RF_Precision", "RF_Recall",
									"RF_Accuracy", "RF_F_Measure", 
									#12

									#Model Gaussian Naive Bayes performance
									"GNB_TP", "GNB_FP", "GNB_TN", "GNB_FN",
									"GNB_Pos_Precision", "GNB_Neg_Precision",
									"GNB_Pos_Recall", "GNB_Neg_Recall",
									"GNB_Precision", "GNB_Recall",
									"GNB_Accuracy", "GNB_F_Measure",
									#12 

									#Model KNN performance
									"KNN_TP", "KNN_FP", "KNN_TN", "KNN_FN",
									"KNN_Pos_Precision", "KNN_Neg_Precision",
									"KNN_Pos_Recall", "KNN_Neg_Recall",
									"KNN_Precision", "KNN_Recall",
									"KNN_Accuracy", "KNN_F_Measure",
									#12

									#Model Logistic performance 
									"LOG_TP", "LOG_FP", "LOG_TN", "LOG_FN",
									"LOG_Pos_Precision", "LOG_Neg_Precision",
									"LOG_Pos_Recall", "LOG_Neg_Recall",
									"LOG_Precision", "LOG_Recall",
									"LOG_Accuracy", "LOG_F_Measure",
									#12

									#Model Ensemble meta-parameters
									"Ensemble_Bool",
									"Ensemble_SVM_Weight", "Ensemble_RF_Weight", 
									"Ensemble_KNN_Weight", "Ensemble_GNB_Weight", "Ensemble_LOG_Weight",
									#6

									#Ensemble performance
									"Ensemble_TP", "Ensemble_FP", "Ensemble_TN", "Ensemble_FN",
									"Ensemble_Pos_Precision", "Ensemble_Neg_Precision",
									"Ensemble_Pos_Recall", "Ensemble_Neg_Recall",
									"Ensemble_Precision", "Ensemble_Recall",
									"Ensemble_Accuracy", "Ensemble_F_Measure",
									#12

									#Results log filename
									"Res_Log_Filename"]],
									#1

									#Results column names
									["Results_Log"])

#################################################################################################################################


#############################################################################################################################################
# Experimental Testing included below. Everything above is boilerplate code
#############################################################################################################################################
s1 = Sample(fraud_data, total_size = 10000, target_ratio = 0.1)

#test2 = Test(Sample(fraud_data, total_size = 1000,sample_method = "Under"),Modeler(),MasterLog)
for i in range(1,11):
	print("\n\n\n" + str(i) + "\n\n\n")
	Test(s1,Modeler(specific_model = "RF", n_estimators = i*10,test_ratio = 0.3),MasterLog)

#Save all data from masterlog
MasterLog.saveMasterLog()











