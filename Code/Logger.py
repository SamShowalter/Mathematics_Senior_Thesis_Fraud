import datetime as dt 
import pandas as pd 

class Log():

	# Collection Log tracks the information about the datasets 
	# Being collected by API and ensures that all meta-data information 
	# is stored.
	def __init__(self, collectionLogName, 
					   resMetadataName, 
					   resLogName, 
					   collectionColNames, 
					   resMetadataColNames,
					   resColNames):



		#Two DataFrame logs for performance and data collection
		self.CollectionLog = pd.DataFrame(columns = collectionColNames)
		self.ResMetadataLog = pd.DataFrame(columns = resMetadataColNames)
		self.ResLog = pd.DataFrame(columns = resColNames)

	# Add a record of an API pull from Quandl to the log
	def addCollectionRecord(self, record):
		new_record_df = pd.DataFrame(
								      [[record.ExecutionDate,
								       record.ExecutionTimeStart,
								       record.Duration,
								       record.sampleSize,
									   record.StatusMessage]],

									   #Add the Collection Log Column Names
									   columns = self.CollectionLog.columns)

		self.CollectionLog = pd.concat([self.CollectionLog ,new_record_df], axis = 0)
		self.CollectionLog.reset_index(drop = True, inplace = True)

	def addResultRecord(self, MLModeler, modelDuration, modelTag, accuracy, modelSpecificInfo = ""):
		new_metadata_df = pd.DataFrame(
								      [[dt.datetime.now().date(),
								       	dt.datetime.now().time(),
								       	modelDuration,
										MLModeler.StockCollector.StockTicker,
										MLModeler.StockCollector.ActualDateStart,
										MLModeler.StockCollector.ActualDateEnd,
										modelTag,
										MLModeler.StockCollector.TrendSpecific,
										MLModeler.TestPeriodFoldSize,
										modelSpecificInfo,
										accuracy]],

									   #Add the Collection Log Column Names
									   columns = self.ResLog.columns)

		self.ResLog = pd.concat([self.ResLog ,new_metadata_df], axis = 0)
		self.ResLog.reset_index(drop = True, inplace = True)


	# Save the collection log as a csv
	def saveCollectionLog(self):
		self.CollectionLog.to_csv(logName + "_" + str(dt.datetime.now()) + "_CollectionLog.csv", sep = ",")

	# Save the results log as a csv
	def saveResultsLog(self):
		self.CollectionLog.to_csv(logName + "_" + str(dt.datetime.now()) + "_ResultsLog.csv", sep = ",")



