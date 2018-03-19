import pandas as pd 
import numpy as np 
import datetime as dt
from Logger import Log
from Ensembler import Ensemble
import copy
from sklearn.model_selection import train_test_split

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Support Vector Machine model
from sklearn.svm import SVC

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Stochastic Gradient Descent Optimizer
from sklearn.linear_model import SGDClassifier

# Gaussian Naive Bayes Optimizer
from sklearn.naive_bayes import GaussianNB

# Machine learning logger for performance 
MLLog = Log("Master_Log",  "Results_Log",  
								#Masterlog column names
							  	["Master_Log"],

								#Results column names
								[[
								#Execution Metadata
								"Execution_Date",
								"Execution_Time",
								"Execution_Duration_Sec",

								#Sample metadata
								"Test_Ratio",
								"Total_Row_Num",
								"NonFraud_Row_Num",
								"Fraud_Row_Num",
								"Fraud_Orig_Row_Num",
								"Fraud_Synth_Row_Num",

								#Model metadata
								"Model_Name",
								"SVM_Wt", "RF_Wt", "GNB_Wt", "KNN_Wt", "LOG_Wt",
								"Precision_Wt", "Recall_Wt",

								#Model performance
								"TP", "FP", "TN", "FN",
								"Pos_Precision", "Neg_Precision",
								"Pos_Recall", "Neg_Recall",
								"Precision", "Recall",
								"Accuracy", "F-Measure",
								"Fraud_Cost"]])

#################################################################################################################################

# Class takes a dataset sample and runs all ML analysis on it, storing the results
class Modeler():

	def __init__(self,
						sample = None,
					    test_ratio = 0.4,		
						n_neighbors = 5, 					
						SVMparams = ('rbf',1,5), 			
						n_estimators = 30,					
						monte_carlo_samp_size = 30,
						ensemble_bool = False,
						prec_wt = 0.5,
						recall_wt = 0.5,
						specific_model = None,
						default_cost = 1.0,
						fraud_mult = 2.54):

		#ML Model parameters
		self.KNNeighbors = n_neighbors
		self.SVMParams = SVMparams
		self.RFEstimators = n_estimators

		# Performance and logistic information
		self.EnsembleBool = ensemble_bool

		#Houses all predictions for each model to be used in ensemble
		if self.EnsembleBool:
			self.EnsembleTrainDF = pd.DataFrame()
			self.EnsembleTestDF = pd.DataFrame()

		# Test period fold size
		self.TestRatio = test_ratio
		self.MonteCarloSampSize = monte_carlo_samp_size
		self.PrecisionWt = prec_wt
		self.RecallWt = recall_wt
		
		#Logging information
		self.Log = copy.deepcopy(MLLog)
		self.ResLogFilename = str(dt.datetime.now().strftime("%m_%d")) + "-" + str(dt.datetime.now().strftime("%H.%M.%S")) + "-ResLog.csv"

		#Specific model information
		self.SpecificModel = specific_model

		#Cost information
		self.DefCostAmt = default_cost
		self.FraudMult = fraud_mult

	def setSample(self,sample):

		self.Sample = sample

		#Set sample information
		self.SampleInfo =  [self.Sample.TotalRowNum,
							self.Sample.NonFraudRowNum,
							self.Sample.FraudRowNum,
							self.Sample.FraudOrigRowNum,
							self.Sample.FraudSynthRowNum]

	def run_model(self):

		#General Bank of Classifiers (used with  "All")
		classifiers = [self.SVM_train_test_model,
			           self.RF_train_test_model,
			           self.GNB_train_test_model,
			           self.KNN_train_test_model,
			           self.LOG_train_test_model]

		#Dynamically Change what classifiers are used
		if self.SpecificModel is None:
			self.Classifiers = "All"
		else:
			self.Classifiers = self.SpecificModel

		#Update result log filename
		self.ResLogFilename = self.Classifiers + "-" + self.ResLogFilename

		# IF monte carlo analysis is asked for
		self.model_engine(classifiers)

	'''
	This is the sklearn KNN model. By passing in the train and test
	data, we can train the model and then test it. This function
	does exactly that and then returns the accuracy, as found
	with the function iter_accuracy
	'''
	def KNN_train_test_model(self, X_train, X_test, y_train, y_test, ensemble_train = False):
		KNN_clf = KNeighborsClassifier(n_neighbors = self.KNNeighbors)
		KNN_clf.fit(X_train,y_train)
		predicted = KNN_clf.predict(X_test)
		actual = y_test

		return self.evaluatePerformance(actual,predicted,X_test.Amount,ensemble_train)

	'''
	This is the sklearn SVM model. By passing in the train and test
	data, we can train the model and then test it. This function
	does exactly that and then returns the accuracy, as found
	with the function iter_accuracy
	'''
	def SVM_train_test_model(self, X_train, X_test, y_train, y_test, ensemble_train = False):
		SVM_clf = SVC(kernel = self.SVMParams[0], C = self.SVMParams[1], gamma = self.SVMParams[2])
		SVM_clf.fit(X_train,y_train)
		predicted = SVM_clf.predict(X_test)
		actual = y_test

		return self.evaluatePerformance(actual,predicted,X_test.Amount,ensemble_train)

	'''
	This is the sklearn GNB model. By passing in the train and test
	data, we can train the model and then test it. This function
	does exactly that and then returns the accuracy, as found
	with the function iter_accuracy
	'''
	def GNB_train_test_model(self, X_train, X_test, y_train, y_test, ensemble_train = False):
		GNB_clf = GaussianNB()
		GNB_clf.fit(X_train, y_train)
		predicted = GNB_clf.predict(X_test)
		actual = y_test

		return self.evaluatePerformance(actual,predicted,X_test.Amount,ensemble_train)

	'''
	This is the sklearn Random Forest model. By passing 
	in the train and test data, we can train the model and then test it. 
	This function does exactly that and then returns the accuracy, as 
	found with the function iter_accuracy.
	'''
	def RF_train_test_model(self, X_train, X_test, y_train, y_test, ensemble_train = False):
		RF_clf = RandomForestClassifier(n_estimators = self.RFEstimators)
		RF_clf.fit(X_train, y_train)
		predicted = RF_clf.predict(X_test)

		actual = y_test

		return self.evaluatePerformance(actual,predicted,X_test.Amount,ensemble_train)


	'''
	This is the sklearn Logistic Regression model. By passing 
	in the train and test data, we can train the model and then test it. 
	This function does exactly that and then returns the accuracy, as 
	found with the function iter_accuracy.
	'''
	def LOG_train_test_model(self, X_train, X_test, y_train, y_test, ensemble_train = False):
		LOG_clf = LogisticRegression()
		LOG_clf.fit(X_train, y_train)
		predicted = LOG_clf.predict(X_test)
		actual = y_test

		return self.evaluatePerformance(actual,predicted,X_test.Amount, ensemble_train)

	'''
	returns accuracy of the sample
	'''
	def accuracy(self, actual, predicted):
		return (actual == predicted).value_counts().get(True,0) / actual.size


	'''
	Positive precision as a function of True Positive and False Positive
	'''
	def posPrecision(self,TP, FP):
		try:
			return TP / (TP + FP)
		except:
			return 0


	'''
	Negative precision as a function of True Negative and False Negative
	'''
	def negPrecision(self,TN, FP):
		try:
			return TN / (TN + FP)
		except:
			return 0


	'''
	Positive recall as a function of True Positive and False Negative
	'''
	def posRecall(self,TP, FN):
		try:
			return TP / (TP + FN)
		except:
			return 0


	'''
	Precision as a function of positive and negative precision
	'''
	def precision(self,pPrecision, nPrecision):
		return (pPrecision * 0.5) + (nPrecision * 0.5)


	'''
	Recall as a function of positive and negative recall
	'''
	def recall(self,pRecall, nRecall):
		return (pRecall * 0.5) + (nRecall * 0.5)


	'''
	Negative precision as a function of True Negative and False Positive
	'''
	def negRecall(self,TN, FP):
		
		try:
			return TN / (TN + FP)
		except:
			return 0

	'''
	F-measure as a function of precision and recall
	'''
	def fMeasure(self,precision, recall):
		if (precision + recall) == 0:
			return 0

		return (2 * precision * recall) / (precision + recall)

	'''
	Evaluate performance of a test. All metrics used
	'''
	def evaluatePerformance(self,actual, predicted, amounts, ensemble_train = False):
		# Variable initialization for consistent logging
		self.EnsembleWts = (0,0,0,0,0)

		#Results data frame (for finding costs)
		resultsDF = pd.DataFrame()
		resultsDF['actual'] = actual
		resultsDF['predicted'] = predicted
		resultsDF['amount'] = amounts

		if self.EnsembleBool:
			if ensemble_train:
				self.EnsembleTrainDF[self.ModelName + "_preds"] = predicted
			else:
				self.EnsembleTestDF[self.ModelName + "_preds"] = predicted

		# Accuracy for the whole test
		accuracy = self.accuracy(actual,predicted)

		# True positive, False positive, True negative, False negative
		TP = ((actual == 1) & (predicted == 1)).value_counts().get(True,0)
		FP = ((actual == 1) & (predicted == 0)).value_counts().get(True,0) 
		TN = ((actual == 0) & (predicted == 0)).value_counts().get(True,0)
		FN = ((actual == 0) & (predicted == 1)).value_counts().get(True,0)

		#Print results DF
		#print(resultsDF)

		#Get the total cost of the model
		fraudChargesFound = resultsDF[(resultsDF.actual == 1) & (resultsDF.predicted == 1)].loc[:,'amount'].sum()
		fraudChargesLost = resultsDF[(resultsDF.actual == 1) & (resultsDF.predicted == 0)].loc[:,'amount'].sum()
		fraudCost = self.fraudCost(TP, fraudChargesFound, FP,fraudChargesLost,FN)

		# Precision and recall metrics (positive and negative)
		posPrecision = self.posPrecision(TP,FP)
		negPrecision = self.negPrecision(TN,FN)
		posRecall = self.posRecall(TP, FN)
		negRecall = self.negRecall(TN, FP)

		# Weighted average total for precision and recall
		precision = self.precision(posPrecision, negPrecision)
		recall = self.recall(posRecall, negRecall)

		# F-measure for the dataset
		fMeasure = self.fMeasure(precision, recall)

		# Return all of these values to the dataset. for DEBUGGING
		print(TP, FP, TN, FN, posPrecision, negPrecision, posRecall, negRecall, precision, recall, accuracy, fMeasure, fraudCost)

		#Return the performance of the model
		return (TP, FP, TN, FN, posPrecision, negPrecision, posRecall, negRecall, precision, recall, accuracy, fMeasure, fraudCost)

	#Determine the cost of the fraud detection algorithm
	def fraudCost(self, TP, TPAmt, FP, FPAmt, FN):
		'''
		True negatives cost nothing
		False negatives cost a fixed rate of contacting customer (or unblocking charge)
		True positives pay the company the cost of the charge saved, less the cost of dealing with charge * 10
		False positives cost the charge amount multiplied by the fraud multipler
		'''
		return (-(TPAmt - self.DefCostAmt * TP*10) + FP*(self.DefCostAmt)) + FPAmt*(self.FraudMult) + (FN * (self.DefCostAmt))


	def add_to_results_dict(self):
		if self.ModelName not in self.ResultsDict:
			self.ResultsDict[self.ModelName] = [self.ModelPerf]
		else:
			self.ResultsDict[self.ModelName].append(self.ModelPerf)


	'''
	Main engine of the model for the ensemble. For each model specified above, this 
	function will store the performance of each model, then record any additional information
	about the ensemble's performance
	'''
	def model_engine(self,classifiers):

		#Initialize self.ResultsDict
		self.ResultsDict = {}
		
		# Check all of classifiers
		for j in range(self.MonteCarloSampSize):

			#Initialize Ensemble object
			self.Ensemble = Ensemble(self)

			#Default uses entire sample
			X_train = self.Sample.Sample.iloc[:,:-1]
			X_test = self.Sample.TestData.iloc[:,:-1]
			y_train = self.Sample.Sample.iloc[:,-1]
			y_test = self.Sample.TestData.iloc[:,-1]

			#Adjust train_and test if the ensemble needs to be trained
			if self.EnsembleBool:

				#Get train and test variables, ignoring all test Data
				X_train_ensemble, X_test_ensemble, y_train_ensemble, y_test_ensemble = train_test_split(self.Sample.Sample.iloc[:,:-1],self.Sample.Sample.iloc[:,-1],test_size = self.TestRatio)
				
				#Rest index in testing data
				X_test_ensemble.Amount.reset_index(drop = True, inplace = True)
				y_test_ensemble.reset_index(drop = True, inplace = True)

			#Run through and test each classifier to get base predictions
			for classifier in classifiers:

				#Get model name
				self.ModelName = classifier.__name__.rsplit('_')[0]

				#Start time for model
				startTime = dt.datetime.utcnow()

				#Get performance and other metadata
				if self.EnsembleBool:
					#Do not need to store the performance of models for train data
					classifier(X_train_ensemble,X_test_ensemble,y_train_ensemble,y_test_ensemble, ensemble_train = True)
				
				#Final model performance that needs to be stored
				self.ModelPerf = classifier(X_train,X_test,y_train,y_test)

				#Add model performance to results DF 
				self.add_to_results_dict()
				
				#Get duration of the model
				self.ModelDurationSec = (dt.datetime.utcnow() - startTime).total_seconds()

			if self.EnsembleBool:
				#Add ensemble information by changing model tag
				self.ModelName = "Ensemble"

				#Add final attributes to dataframe to check accuracy
				self.EnsembleTrainDF['actual'] = y_test.iloc[-1]
				self.EnsembleTrainDF['amount'] = X_test_ensemble["Amount"]

				#Evolve the ensemble model and add performance metrics at the end
				self.EnsembleWts = self.Ensemble.evolve()
				self.EnsembleTestDF['ensemble_predicted'] = self.EnsembleTestDF.iloc[:,:5].apply(lambda row: round(sum(row*self.Ensemble.convertWts(self.Ensemble.BestWts))), axis = 1)
				self.EnsembleTestDF['actual'] = self.Sample.TestData["Class"]
				self.EnsembleTestDF['amount'] = self.Sample.TestData["Amount"]

				#Evaluate the performance of the model overall
				self.ModelPerf = self.evaluatePerformance(self.EnsembleTestDF['ensemble_predicted'], self.EnsembleTestDF['actual'],self.EnsembleTestDF['amount'])

				#Add model performance of ensemble to results DF 
				self.add_to_results_dict()

				#Add record to Logger
				self.Log.addResultRecord(self)

				#Reset ensemble prediction dataframes for train and test
				self.EnsembleTrainDF = pd.DataFrame()
				self.EnsembleTestDF = pd.DataFrame()

			#Resample master sample
			self.Sample.Resample()

		#Save Results Log
		self.Log.saveResultsLog(self.ResLogFilename)
		
		#Format and store the average results
		self.resultsDF = pd.DataFrame.from_dict(self.ResultsDict)

		#Get average results
		averageResults = self.resultsDF.apply(lambda col: tuple(map(np.mean, zip(*col))),axis = 0).to_dict()

		#Store average results for each model (order does not matter because of keys)
		self.SVMPerf = averageResults.get('SVM',(0,0,0,0,0,0,0,0,0,0,0,0,0))
		self.RFPerf = averageResults.get('RF',(0,0,0,0,0,0,0,0,0,0,0,0,0))
		self.GNBPerf = averageResults.get('GNB',(0,0,0,0,0,0,0,0,0,0,0,0,0))
		self.KNNPerf = averageResults.get('KNN',(0,0,0,0,0,0,0,0,0,0,0,0,0))
		self.LOGPerf = averageResults.get('LOG',(0,0,0,0,0,0,0,0,0,0,0,0,0))

		#This will be the only one not equal to zero
		self.EnsemblePerf = averageResults.get('Ensemble',(0,0,0,0,0,0,0,0,0,0,0,0,0))




