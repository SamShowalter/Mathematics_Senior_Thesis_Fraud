from Logger import Log
from MLModeler import Modeler 
from Tester import Test 
from Sampler import Sample
from Ensembler import Ensemble
import copy
import pandas as pd 
import numpy as np 
import os

#Change working directory to import data
os.chdir("/Users/Sam/Documents/Depauw/04_Senior_Year/Semester_2/Math_Senior_Seminar/Data")

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
									"Total_Sample_Row_Num",
									"Sample_NonFraud_Row_Num",
									"Sample_Fraud_Row_Num",
									"Sample_Fraud_Orig_Row_Num",
									"Sample_Fraud_Synth_Row_Num",
									"Total_Test_Data_Row_Num",
									"Test_Data_NonFraud_Row_Num",
									"Test_Data_Fraud_Row_Num",
									#6

									#Modeler Meta-parameters
									"Models_Used",
									"SVM_Params",
									"RF_Estimators",
									"KNN_Neighbors",
									"LOG_Params",
									"Monte_Carlo_Samp_Size",
									"Precision_Wt", "Recall_Wt",
									
									#Cost metaparameters
									"Default_Fraud_Cost",
									"Fraud_Multiplier",

									#Modeler SVM Performance
									"SVM_TP", "SVM_FP", "SVM_TN", "SVM_FN",
									"SVM_Pos_Precision", "SVM_Neg_Precision",
									"SVM_Pos_Recall", "SVM_Neg_Recall",
									"SVM_Precision", "SVM_Recall",
									"SVM_Accuracy", "SVM_F_Measure",
									"SVM_Fraud_Cost",
									#12

									#Model Random Forest Performance   
									"RF_TP", "RF_FP", "RF_TN", "RF_FN",
									"RF_Pos_Precision", "RF_Neg_Precision",
									"RF_Pos_Recall", "RF_Neg_Recall",
									"RF_Precision", "RF_Recall",
									"RF_Accuracy", "RF_F_Measure", 
									"RF_Fraud_Cost",
									#12

									#Model Gaussian Naive Bayes performance
									"GNB_TP", "GNB_FP", "GNB_TN", "GNB_FN",
									"GNB_Pos_Precision", "GNB_Neg_Precision",
									"GNB_Pos_Recall", "GNB_Neg_Recall",
									"GNB_Precision", "GNB_Recall",
									"GNB_Accuracy", "GNB_F_Measure",
									"GNB_Fraud_Cost",
									#12 

									#Model KNN performance
									"KNN_TP", "KNN_FP", "KNN_TN", "KNN_FN",
									"KNN_Pos_Precision", "KNN_Neg_Precision",
									"KNN_Pos_Recall", "KNN_Neg_Recall",
									"KNN_Precision", "KNN_Recall",
									"KNN_Accuracy", "KNN_F_Measure",
									"KNN_Fraud_Cost",
									#12

									#Model Logistic performance 
									"LOG_TP", "LOG_FP", "LOG_TN", "LOG_FN",
									"LOG_Pos_Precision", "LOG_Neg_Precision",
									"LOG_Pos_Recall", "LOG_Neg_Recall",
									"LOG_Precision", "LOG_Recall",
									"LOG_Accuracy", "LOG_F_Measure",
									"LOG_Fraud_Cost",
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
									"Ensemble_Fraud_Cost",
									#12

									#Results log filename
									"Res_Log_Filename"]],
									#1

									#Results column names
									["Results_Log"])

#############################################################################################################################################
#Email functionality so that updates can be received in real time.

#############################################################################################################################################
# Send emails about progress
def sendProgressEmail(subject, message):
	#Establish server
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.starttls()

	#Log in to my email account
	server.login("dpudatascience@gmail.com", "Data4good!")

	#Send the final message
	final_message = 'Subject: {}\n\n{}'.format(subject, message)
	server.sendmail("dpudatascience@gmail.com", "samuelrshowalter@gmail.com", final_message)

	#Quit the server
	server.quit()

def generateProgressInformation(Test, ProgressInfo):
	pass

#############################################################################################################################################
# Experimental Testing included below. Everything above is boilerplate code
#############################################################################################################################################
s1 = Sample(fraud_data, 
			sample_method = 'SMOTE',
			total_sample_size = 1000, 
			target_ratio = 0.2)

#make copies of the masterlog

# Test(s1,Modeler(test_ratio = 0.3), MasterLog)
Test(s1,
	Modeler(test_ratio = 0.4, 
			ensemble_bool = True, 
			monte_carlo_samp_size = 1),
	MasterLog)

#Save all data from masterlog
MasterLog.saveMasterLog()











