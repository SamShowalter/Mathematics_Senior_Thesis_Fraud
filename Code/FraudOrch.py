import pandas as pd 
import numpy as np 
import os

#Change working directory to import data
os.chdir("/Users/Sam/Documents/Depauw/04 Senior Year/Semester 2/Math_Senior_Seminar/Mathematics_Senior_Thesis_Fraud/Data")

#Read in fraud data
#fraud_data = pd.read_csv("creditcard.csv")

MasterLog = Log("Test_Collection_Log", "Test_Metadata",  "Test_Results_Log",  #Collection columns names
															  [["Execution_Date",
																"Execution_Time",
																"Execution_Duration_Sec",
																"Stock_Ticker",
																"Scheduled_Start_Date",
																"Scheduled_End_Date",
																"Actual_Start_Date",
																"Actual_End_Date",
																"Row_Count",
																"Column_Count",
																"NaN_Row_Count",
																"Trend_Specific",
																"Status_Message"]],

																#Results metadata names (need to fill out)
																["Test Metadata"],

																#Results column names
																["Not Applicable"])

MLLog = Log("Test_Collection_Log", "Test_Metadata",  "Test_Results_Log",  
																#Collection columns names
															  	["Not Applicable"],

																#Results metadata names (need to fill out)
																["Not Applicable"],

																#Results column names
																[["Execution_Date",
																"Execution_Time",
																"Execution_Duration_Sec",
																"Stock_Ticker",
																"Stock_Start_Date",
																"Stock_End_Date",
																"Model_Tag",
																"Trend_Specific",
																"Test_Period_Years",
																"Model_Specific_Info",
																"Accuracy"]])

'''
This model iteratively returns a single value,
the accuracy of the model as found by 
num_correct/tot_num_samples. This is the function
that will be called iteratively when the sampling
engine runs for all sklearn models. Inputs are 
actual labels, and predicted labels.
'''
def accuracy(actual, predicted):
	return (actual == predicted).value_counts().tolist()[1] / actual.size

df = pd.DataFrame({'actual':[1,1,3],'predicted':[4,1,6]}, dtype = 'int')

print(((df.actual == 1) & (df.predicted == 1)).value_counts().tolist()[1])