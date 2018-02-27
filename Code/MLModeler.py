import pandas as pd 
import numpy as np 
import datetime as dt
from Logger import Log
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
								"Accuracy", "F-Measure"]])

#################################################################################################################################

# Class takes a dataset sample and runs all ML analysis on it, storing the results
class Modeler():

	def __init__(self,
						sample = None,
					    test_period_fold_size = 1,
					    test_ratio = 0.4,		
						n_neighbors = 5, 					
						SVMparams = ('rbf',1,5), 			
						n_estimators = 30,					
						monte_carlo = True,
						monte_carlo_samp_size = 30,
						k_fold_bool = False,
						ensemble_bool = False,
						k_fold_num = 1,
						prec_wt = 0.5,
						specific_model = None):

		#ML Model parameters
		self.KNNeighbors = n_neighbors
		self.SVMParams = SVMparams
		self.RFEstimators = n_estimators

		# Performance and logistic information
		self.KFoldBool = k_fold_bool
		self.KFoldNum = k_fold_num
		self.EnsembleBool = ensemble_bool
		self.TestPeriodFoldSize = test_period_fold_size
		self.TestRatio = test_ratio
		self.MonteCarlo = monte_carlo
		self.MonteCarloSampSize = monte_carlo_samp_size
		self.PrecisionWt = prec_wt
		self.RecallWt = (1 - prec_wt)
		self.Log = copy.deepcopy(MLLog)
		self.ResLogFilename = str(dt.datetime.now().strftime("%m_%d")) + "-" + str(dt.datetime.now().strftime("%H.%M")) + "-ResLog.csv"

		#Specific model information
		self.SpecificModel = specific_model

	def setSample(self,sample):
		self.Sample = sample
		self.SampleInfo =  [self.Sample.TotalRowNum,
							self.Sample.NonFraudRowNum,
							self.Sample.FraudRowNum,
							self.Sample.FraudSynthRowNum,
							self.Sample.FraudOrigRowNum]

	def run_model(self):
		classifiers = [self.RF_train_test_model,
			           self.LOG_train_test_model,
			           self.GNB_train_test_model,
			           self.KNN_train_test_model,
			           self.SVM_train_test_model]

		#Change to be dynamic
		self.Classifiers = "All"

		# IF monte carlo analysis is asked for
		if self.MonteCarlo:
			self.model_engine(classifiers)

	'''
	This is the sklearn KNN model. By passing in the train and test
	data, we can train the model and then test it. This function
	does exactly that and then returns the accuracy, as found
	with the function iter_accuracy
	'''
	def KNN_train_test_model(self, X_train, X_test, y_train, y_test):
		KNN_clf = KNeighborsClassifier(n_neighbors = self.KNNeighbors)
		KNN_clf.fit(X_train,y_train)
		predicted = KNN_clf.predict(X_test)
		actual = y_test

		return self.evaluatePerformance(actual,predicted)

	'''
	This is the sklearn SVM model. By passing in the train and test
	data, we can train the model and then test it. This function
	does exactly that and then returns the accuracy, as found
	with the function iter_accuracy
	'''
	def SVM_train_test_model(self, X_train, X_test, y_train, y_test):
		SVM_clf = SVC(kernel = self.SVMParams[0], C = self.SVMParams[1], gamma = self.SVMParams[2])
		SVM_clf.fit(X_train,y_train)
		predicted = SVM_clf.predict(X_test)
		actual = y_test

		return self.evaluatePerformance(actual,predicted)

	'''
	This is the sklearn GNB model. By passing in the train and test
	data, we can train the model and then test it. This function
	does exactly that and then returns the accuracy, as found
	with the function iter_accuracy
	'''
	def GNB_train_test_model(self, X_train, X_test, y_train, y_test):
		GNB_clf = GaussianNB()
		GNB_clf.fit(X_train, y_train)
		predicted = GNB_clf.predict(X_test)
		actual = y_test

		return self.evaluatePerformance(actual,predicted)

	'''
	This is the sklearn Random Forest model. By passing 
	in the train and test data, we can train the model and then test it. 
	This function does exactly that and then returns the accuracy, as 
	found with the function iter_accuracy.
	'''
	def RF_train_test_model(self, X_train, X_test, y_train, y_test):
		RF_clf = RandomForestClassifier(n_estimators = self.RFEstimators)
		RF_clf.fit(X_train, y_train)
		predicted = RF_clf.predict(X_test)
		actual = y_test

		return self.evaluatePerformance(actual,predicted)


	'''
	This is the sklearn Logistic Regression model. By passing 
	in the train and test data, we can train the model and then test it. 
	This function does exactly that and then returns the accuracy, as 
	found with the function iter_accuracy.
	'''
	def LOG_train_test_model(self, X_train, X_test, y_train, y_test):
		LOG_clf = LogisticRegression()
		LOG_clf.fit(X_train, y_train)
		predicted = LOG_clf.predict(X_test)
		actual = y_test

		return self.evaluatePerformance(actual,predicted)

	'''
	returns accuracy of the sample
	'''
	def accuracy(self, actual, predicted):
		return (actual == predicted).value_counts().get(True,0) / actual.size


	'''
	Positive precision as a function of True Positive and False Positive
	'''
	def posPrecision(self,TP, FP):
		return TP / (TP + FP)


	'''
	Negative precision as a function of True Negative and False Negative
	'''
	def negPrecision(self,TN, FP):
		return TN / (TN + FP)


	'''
	Positive recall as a function of True Positive and False Negative
	'''
	def posRecall(self,TP, FN):
		return TP / (TP + FN)


	'''
	Precision as a function of positive and negative precision
	'''
	def precision(self,pPrecision, nPrecision):
		return (pPrecision * self.PrecisionWt) + (nPrecision * (1-self.PrecisionWt))

	
	'''
	Recall as a function of positive and negative recall
	'''
	def recall(self,pRecall, nRecall):
		return (pRecall * self.RecallWt) + (nRecall * (1-self.RecallWt))


	'''
	Negative precision as a function of True Negative and False Positive
	'''
	def negRecall(self,TN, FP):
		return TN / (TN + FP)

	
	'''
	F-measure as a function of precision and recall
	'''
	def fMeasure(self,precision, recall):
		return (2 * precision * recall) / (precision + recall)

	'''
	Evaluate performance of a test.
	'''
	def evaluatePerformance(self,actual, predicted, countsOnly = False):
		# Variable initialization for consistent logging
		self.EnsembleWts = (0,0,0,0,0)

		resultsDF = pd.DataFrame(actual,predicted)

		# Accuracy for the whole test
		accuracy = self.accuracy(actual,predicted)

		# True positive, False positive, True negative, False negative
		TP = ((actual == 1) & (predicted == 1)).value_counts().get(True,0)
		FP = ((actual == 1) & (predicted == 0)).value_counts().get(True,0) 
		TN = ((actual == 0) & (predicted == 0)).value_counts().get(True,0)
		FN = ((actual == 0) & (predicted == 1)).value_counts().get(True,0)

		# If the results only need counts (for visualization)
		if countsOnly:
			return (TP, FP, TN, FN)

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

		# Return all of these values to the dataset. for debugging
		print(TP, FP, TN, FN, posPrecision, negPrecision, posRecall, negRecall, precision, recall, accuracy, fMeasure)

		#Return the performance of the model
		return (TP, FP, TN, FN, posPrecision, negPrecision, posRecall, negRecall, precision, recall, accuracy, fMeasure)

		

	'''
	Main engine of the model. For each model specified above, this 
	function will run it and store the accuracy data as a dictionary.
	The keys for this dictionary are the names of the functions above,
	before the first underscore. This function allows you to specify
	the number of samples you would like to collect, the test ratio for
	how much of the dataset you want to predict, and a list of the
	models that you want to provide. For this model we are going to predict
	all five.
	'''
	def model_engine(self,classifiers):

		# Results dictionary
	    results_dict = {}

	    # Check tons of classifiers
	    for classifier in classifiers:
	        res_list = []
	        model_tag = classifier.__name__.rsplit('_')[0]

	        for j in range(self.MonteCarloSampSize):

	        	#Start time for model
	        	startTime = dt.datetime.utcnow()

	        	#Get train and test variables
	        	X_train, X_test, y_train, y_test = train_test_split(self.Sample.Sample.iloc[:,:-1],self.Sample.Sample.iloc[:,-1],test_size = self.TestRatio)

	        	#Get performance and other metadata
	        	self.EnsembleWts = [0,0,0,0,0]
	        	self.ModelPerf = classifier(X_train,X_test,y_train,y_test)
	        	self.ModelName = model_tag

	        	#Add record to results DF
	        	res_list.append(self.ModelPerf)

	        	self.ModelDurationSec = (dt.datetime.utcnow() - startTime).total_seconds()

	        	#Add record to Logger
	        	self.Log.addResultRecord(self)

	        # Add results to results dict
	        results_dict[model_tag] = res_list

	        #Save Results Log
	        self.Log.saveResultsLog(self.ResLogFilename)

	    #Format and store the average results
	    self.resultsDF = pd.DataFrame.from_dict(results_dict)
	    averageResults = self.resultsDF.apply(lambda col: tuple(map(np.mean, zip(*col))),axis = 0)

	    #Store average results for each model (order does not matter because of keys)
	    self.SVMPerf = averageResults['SVM']
	    self.RFPerf = averageResults['RF']
	    self.GNBPerf = averageResults['GNB']
	    self.KNNPerf = averageResults['KNN']
	    self.LOGPerf = averageResults['LOG']

	    #Default the values of the ensemble if no ensemble exists
	    if not self.EnsembleBool:
	    	self.EnsemblePerf = (0,0,0,0,0,0,0,0,0,0,0,0)
