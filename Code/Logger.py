import datetime as dt 
import pandas as pd 
import os


# NEEDS TO BE UPDATED FOR FRAUD DATA

class Log():

	# Collection Log tracks the information about the datasets 
	# Being collected by API and ensures that all meta-data information 
	# is stored.
	def __init__(self, master_log_name, 
					   res_log_name, 
					   masterColNames, 
					   resColNames):

		#Log names
		self.MasterLogName = master_log_name
		self.ResLogName = res_log_name

		#Two DataFrame logs for performance and data collection
		self.MasterLog = pd.DataFrame(columns = masterColNames)
		self.ResLog = pd.DataFrame(columns = resColNames)

	# Add MasterLog record
	def addMasterLogRecord(self, test):
		new_record_df = pd.DataFrame(
								      [[
								       #Test Execution Information
								       test.ExecutionDateStart,
								       test.ExecutionTimeStart,
								       test.Sample.SampleDuration,
								       test.ModelDuration,
								       test.TestDuration,

								       #Information from Sample
								       test.Sample.SampleMethod,
								       test.Sample.TotalRowNum,			#Total rows
								       test.Sample.NonFraudRowNum,		#Total Non-fraud Rows
								       test.Sample.FraudRowNum,			#Non-fraud rows
								       test.Sample.FraudOrigRowNum,		#Fraud original rows
								       test.Sample.FraudSynthRowNum,	#Fraud synthetic rows

								       #Information from modeler
								       test.Modeler.Classifiers,
								       test.Modeler.SVMParams,
								       test.Modeler.RFEstimators,
								       test.Modeler.KNNeighbors,
								       test.Modeler.MonteCarlo,
								       test.Modeler.MonteCarloSampSize,
								       test.Modeler.KFoldBool, test.Modeler.KFoldNum,
								       test.Modeler.PrecisionWt, test.Modeler.RecallWt,

								       #Modeler information about Support vector machine
								       test.Modeler.SVMPerf[0],		#TP
								       test.Modeler.SVMPerf[1],		#FP
								       test.Modeler.SVMPerf[2],		#TN
								       test.Modeler.SVMPerf[3],		#FN
								       test.Modeler.SVMPerf[4],		#Positive precision
								       test.Modeler.SVMPerf[5],		#Negative precision
								       test.Modeler.SVMPerf[6],		#Positive recall
								       test.Modeler.SVMPerf[7],		#Negative recall
								       test.Modeler.SVMPerf[8],		#Precision
								       test.Modeler.SVMPerf[9],		#Recall
								       test.Modeler.SVMPerf[10],	#Accuracy
								       test.Modeler.SVMPerf[11],	#F-measure

								       #Modeler information about Random forest
								       test.Modeler.RFPerf[0],		#TP
								       test.Modeler.RFPerf[1],		#FP
								       test.Modeler.RFPerf[2],		#TN
								       test.Modeler.RFPerf[3],		#FN
								       test.Modeler.RFPerf[4],		#Positive precision
								       test.Modeler.RFPerf[5],		#Negative precision
								       test.Modeler.RFPerf[6],		#Positive recall
								       test.Modeler.RFPerf[7],		#Negative recall
								       test.Modeler.RFPerf[8],		#Precision
								       test.Modeler.RFPerf[9],		#Recall
								       test.Modeler.RFPerf[10],		#Accuracy
								       test.Modeler.RFPerf[11],		#F-measure

								       #Modeler information about Gaussian Naive Bayes
								       test.Modeler.GNBPerf[0],		#TP
								       test.Modeler.GNBPerf[1],		#FP
								       test.Modeler.GNBPerf[2],		#TN
								       test.Modeler.GNBPerf[3],		#FN
								       test.Modeler.GNBPerf[4],		#Positive precision
								       test.Modeler.GNBPerf[5],		#Negative precision
								       test.Modeler.GNBPerf[6],		#Positive recall
								       test.Modeler.GNBPerf[7],		#Negative recall
								       test.Modeler.GNBPerf[8],		#Precision
								       test.Modeler.GNBPerf[9],		#Recall
								       test.Modeler.GNBPerf[10],	#Accuracy
								       test.Modeler.GNBPerf[11],	#F-measure
								       
								       #Modeler information about K-Nearest Neighbors
								       test.Modeler.KNNPerf[0],		#TP
								       test.Modeler.KNNPerf[1],		#FP
								       test.Modeler.KNNPerf[2],		#TN
								       test.Modeler.KNNPerf[3],		#FN
								       test.Modeler.KNNPerf[4],		#Positive precision
								       test.Modeler.KNNPerf[5],		#Negative precision
								       test.Modeler.KNNPerf[6],		#Positive recall
								       test.Modeler.KNNPerf[7],		#Negative recall
								       test.Modeler.KNNPerf[8],		#Precision
								       test.Modeler.KNNPerf[9],		#Recall
								       test.Modeler.KNNPerf[10],	#Accuracy
								       test.Modeler.KNNPerf[11],	#F-measure

								       #Modeler information about Logistic Regression
								       test.Modeler.LOGPerf[0],		#TP
								       test.Modeler.LOGPerf[1],		#FP
								       test.Modeler.LOGPerf[2],		#TN
								       test.Modeler.LOGPerf[3],		#FN
								       test.Modeler.LOGPerf[4],		#Positive precision
								       test.Modeler.LOGPerf[5],		#Negative precision
								       test.Modeler.LOGPerf[6],		#Positive recall
								       test.Modeler.LOGPerf[7],		#Negative recall
								       test.Modeler.LOGPerf[8],		#Precision
								       test.Modeler.LOGPerf[9],		#Recall
								       test.Modeler.LOGPerf[10],	#Accuracy
								       test.Modeler.LOGPerf[11],	#F-measure

								       #Modeler information about Ensemble Weights
								       test.Modeler.EnsembleBool,
								       test.Modeler.EnsembleWts[0],		#SVM
								       test.Modeler.EnsembleWts[1],		#RF
								       test.Modeler.EnsembleWts[2],		#GNB
								       test.Modeler.EnsembleWts[3],		#KNN
								       test.Modeler.EnsembleWts[4],		#LOG

								       #Modeler information about Ensemble Performance
								       test.Modeler.EnsemblePerf[0],		#TP
								       test.Modeler.EnsemblePerf[1],		#FP
								       test.Modeler.EnsemblePerf[2],		#TN
								       test.Modeler.EnsemblePerf[3],		#FN
								       test.Modeler.EnsemblePerf[4],		#Positive precision
								       test.Modeler.EnsemblePerf[5],		#Negative precision
								       test.Modeler.EnsemblePerf[6],		#Positive recall
								       test.Modeler.EnsemblePerf[7],		#Negative recall
								       test.Modeler.EnsemblePerf[8],		#Precision
								       test.Modeler.EnsemblePerf[9],		#Recall
								       test.Modeler.EnsemblePerf[10],		#Accuracy
								       test.Modeler.EnsemblePerf[11],		#F-measure

								       #Results Log filename for the modeler
								       test.Modeler.ResLogFilename]],

									   #Add the Collection Log Column Names
									   columns = self.MasterLog.columns)

		self.MasterLog = pd.concat([self.MasterLog ,new_record_df], axis = 0)
		self.MasterLog.reset_index(drop = True, inplace = True)

	def addResultRecord(self, model):
		new_metadata_df = pd.DataFrame(
								      [[
								      	#Test Execution Information
								      	dt.datetime.now().date(),
								       	dt.datetime.now().time(),
								       	model.ModelDurationSec,

								       	#Train Data Information
								       	model.TestRatio,
										model.SampleInfo[0],			#Total Rows
										model.SampleInfo[1],			#NonFraud Rows
										model.SampleInfo[2],			#Number of fraud rows
										model.SampleInfo[3],			#Original Fraud Rows
										model.SampleInfo[4],			#Synthetic Fraud Rows

										#Test Data Information (may not be needed or accessible. REVISIT)
										# model.TestData[0],			#Total Rows	
										# model.TestData[1],			#Original Rows
										# model.TestData[2],			#Synthetic Rows

										#Model General Information
										model.ModelName,
										model.EnsembleWts[0],		#SVM
										model.EnsembleWts[1],		#RF
										model.EnsembleWts[2],		#GNB
										model.EnsembleWts[3],		#KNN
										model.EnsembleWts[4],		#LOG
										model.PrecisionWt, model.RecallWt,

										#Model performance information
										model.ModelPerf[0],		#TP
								       	model.ModelPerf[1],		#FP
								       	model.ModelPerf[2],		#TN
								       	model.ModelPerf[3],		#FN
								       	model.ModelPerf[4],		#Positive precision
								       	model.ModelPerf[5],		#Negative precision
								       	model.ModelPerf[6],		#Positive recall
								       	model.ModelPerf[7],		#Negative recall
								       	model.ModelPerf[8],		#Precision
								       	model.ModelPerf[9],		#Recall
								       	model.ModelPerf[10],		#Accuracy
								       	model.ModelPerf[11],		#F-measure
								        ]],

									   #Add the Collection Log Column Names
									   columns = self.ResLog.columns)

		self.ResLog = pd.concat([self.ResLog ,new_metadata_df], axis = 0)
		self.ResLog.reset_index(drop = True, inplace = True)


	# Save the collection log as a csv
	def saveMasterLog(self):
		#Change working directory for Master Logs
		os.chdir("/Users/Sam/Documents/Depauw/04 Senior Year/Semester 2/Math_Senior_Seminar/Data/MasterLogs")

		self.MasterLog.to_csv(str(dt.datetime.now().strftime("%m_%d")) + "-" + str(dt.datetime.now().strftime("%H.%M.%S")) + "-MasterLog.csv", sep = ",")

	# Save the results log as a csv
	def saveResultsLog(self, resLogName):
		#Change working directory for Result Logs
		os.chdir("/Users/Sam/Documents/Depauw/04 Senior Year/Semester 2/Math_Senior_Seminar/Data/ResLogs")

		#Save the log
		self.ResLog.to_csv(resLogName, sep = ",")



