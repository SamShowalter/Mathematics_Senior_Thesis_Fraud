import pandas as pd 
import numpy as np 
import datetime as dt
from Logger import Log

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


class Modeler():

	def __init__(self,  sample,
						log,
					    test_period_fold_size = 1,			# Either 1 (day), 5 (week), 20 (month)
						n_neighbors = 5, 					# 1 = 100 day test period
						SVMparams = ('rbf',1,5), 			# 5 = 50 week (250 day) test period
						n_estimators = 80,					# 20 = 25 month (500 day) test period
						monteCarlo = True,
						monteCarloSampSize = 100,
						specificModel = None):

		self.sample = sample
		self.KNNeighbors = n_neighbors
		self.SVMParams = SVMparams
		self.TestPeriodFoldSize = test_period_fold_size
		self.RFEstimators = n_estimators
		self.SpecificModel = specificModel
		self.Log = log

		classifiers = [self.RF_train_test_model,
		               self.LOG_train_test_model,
		               self.GNB_train_test_model,
		               self.KNN_train_test_model,
		               self.SVM_train_test_model]

		if self.monteCarlo:
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
	def accuracy(actual, predicted):
		return (actual == predicted).value_counts().tolist()[1] / actual.size


	'''
	Positive precision as a function of True Positive and False Positive
	'''
	def posPrecision(TP, FP):
		return TP / (TP + FP)


	'''
	Negative precision as a function of True Negative and False Negative
	'''
	def negPrecision(TN, FP):
		return TN / (TN + FP)


	'''
	Positive recall as a function of True Positive and False Negative
	'''
	def posRecall(TP, FN):
		return TP / (TP + FN)


	'''
	Precision as a function of positive and negative precision
	'''
	def precision(pPrecision, nPrecision, pWeight, nWeight):
		return (pPrecision * pWeight) + (nPrecision * nWeight)

	
	'''
	Recall as a function of positive and negative recall
	'''
	def recall(pRecall, nRecall, pWeight, nWeight):
		return (pRecall * pWeight) + (nRecall * nWeight)


	'''
	Negative precision as a function of True Negative and False Positive
	'''
	def negRecall(TN, FP):
		return TN / (TN + FP)

	
	'''
	F-measure as a function of precision and recall
	'''
	def fMeasure(precision, recall):
		return (2 * precision * recall) / (precision + recall)

	'''
	Evaluate performance of a test.
	'''
	def evaluatePerformance(actual, predicted, countsOnly = False):
		resultsDF = pd.DataFrame(actual,predicted)

		accuracy = (actual == predicted).value_counts().tolist()[1] / actual.size 

		TP = (actual == 1 & predicted == 1).value_counts().tolist()[1] 
		FP = (actual == 1 & predicted == 0).value_counts().tolist()[1] 
		TN = (actual == 0 & predicted == 0).value_counts().tolist()[1] 
		FN = (actual == 0 & predicted == 1).value_counts().tolist()[1] 

		if countsOnly:
			return (TP, FP, TN, FN)

		posPrecision = posPrecision(TP,FP)
		negPrecision = negPrecision(TN,FN)
		posRecall = posRecall(TP, FN)
		negRecall = negRecall(TN, FP)

		precision = precision(posPrecision, negPrecision)
		recall = recall(posRecall, negRecall)

		fMeasure = fMeasure(precision, recall)


		return (posPrecision,negPrecision, posRecall, negRecall, precision, recall, fMeasure)

		

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
	def model_engine(self,classifiers,test_ratio,X_train, X_test, y_train, y_test):
	    results_dict = {}
	    for classifier in classifiers:
	        res_list = []
	        model_tag = model.__name__.rsplit('_')[0] + "_Test_Ratio_" + str(test_ratio)
	        for j in range(self.monteCarloSampSize):
	            performance = model(X_train,X_test,y_train,y_test)
	            res_list.append(performance)
	        results_dict[model_tag] = res_list

	    self.resultsDF = pd.DataFrame.from_dict(results_dict)


	def saveResultsDF(self, resultsName):
		self.resultsDF.to_csv(logName + "_" + str(dt.datetime.now()) + "_ResultsDF.csv", sep = ",")