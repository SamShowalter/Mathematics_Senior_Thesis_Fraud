import pandas as pd 
import numpy as np 
import datetime as dt
from Logger import Log
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

# Class takes a dataset sample and runs all ML analysis on it, storing the results
class Modeler():

	def __init__(self,  sample,
						log,
					    test_period_fold_size = 1,
					    test_ratio = 0.4,		
						n_neighbors = 5, 					
						SVMparams = ('rbf',1,5), 			
						n_estimators = 30,					
						monte_carlo = True,
						monte_carlo_samp_size = 30,
						prec_wt = 0.5,
						specific_model = None):

		self.Sample = sample
		self.KNNeighbors = n_neighbors
		self.SVMParams = SVMparams
		self.TestPeriodFoldSize = test_period_fold_size
		self.TestRatio = test_ratio
		self.RFEstimators = n_estimators
		self.SpecificModel = specific_model
		self.MonteCarlo = monte_carlo
		self.MonteCarloSampSize = monte_carlo_samp_size
		self.PrecisionWt = prec_wt
		self.RecallWt = (1 - prec_wt)
		self.Log = log

		classifiers = [self.RF_train_test_model,
		               self.LOG_train_test_model,
		               self.GNB_train_test_model,
		               self.KNN_train_test_model,
		               self.SVM_train_test_model]

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
		return (actual == predicted).value_counts()[True] / actual.size


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
		TP = ((actual == 1) & (predicted == 1)).value_counts()[True] 
		FP = ((actual == 1) & (predicted == 0)).value_counts()[True] 
		TN = ((actual == 0) & (predicted == 0)).value_counts()[True] 
		FN = ((actual == 0) & (predicted == 1)).value_counts()[True] 

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

		# Return all of these values to the dataset.
		print(TP, FP, TN, FN, posPrecision, negPrecision, posRecall, negRecall, precision, recall, accuracy, fMeasure)

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


	    results_dict = {}
	    for classifier in classifiers:
	        res_list = []
	        model_tag = classifier.__name__.rsplit('_')[0] + "_Test_Ratio_" + str(self.TestRatio)

	        for j in range(self.MonteCarloSampSize):

	        	#Start time for model
	        	startTime = dt.datetime.utcnow()

	        	#Get train and test variables
	        	X_train, X_test, y_train, y_test = train_test_split(self.Sample.Sample.iloc[:,:-1],self.Sample.Sample.iloc[:,-1],test_size = self.TestRatio)

	        	#Get performance
	        	self.ModelPerf = classifier(X_train,X_test,y_train,y_test)

	        	#Add record to results DF
	        	res_list.append(self.ModelPerf)

	        	self.ModelDurationSec = startTime - dt.datetime.utcnow()

	        	#Add record to Logger
	        	self.Log.addResultRecord(self)

	        # Add results to results dict
	        results_dict[model_tag] = res_list

	    self.resultsDF = pd.DataFrame.from_dict(results_dict)
	    print(self.resultsDF.describe())

	    # Update the metadata records with average performance

	    
	def saveResultsDF(self, resultsName):
		self.resultsDF.to_csv(logName + "_" + str(dt.datetime.now()) + "_ResultsDF.csv", sep = ",")