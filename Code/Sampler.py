import pandas as pd 
import numpy as np 
import copy
import datetime as dt
from scipy.spatial.distance import euclidean as euc
import random

class Sample():

	'''
	Sample method can be:
	 - SMOTE
	 - Standard
	 - Under

	 Other attributes involve the specifics of the sample composition and SMOTE parameters for analysis

	 NOTE: ANY SMOTE TESTS WITH LESS THAN 492 FRAUD RECORDS WILL ESSENTIALLY BE UNDER-SAMPLING. ALWAYS HAVE MORE
	'''
	def __init__(self, dataset, 
					   sample_method = 'SMOTE', 
					   SMOTE_Neighbors = 5, 
					   target_ratio = 0.20, 
					   total_size = 2000):
		utc_start = dt.datetime.utcnow()

		#Separate data into train and testing subsamples.
		self.FullData = dataset

		#Give the sample the same column names as the original dataset
		self.Sample = dataset[0:0]

		#Metadata about how (and how much) to sample
		self.SampleMethod = sample_method
		self.SMOTENeighbors = SMOTE_Neighbors
		self.TargetRatio = target_ratio
		self.TotalSize = total_size
		self.SMOTENeighbors = SMOTE_Neighbors

		#Run the sampler
		self.runSampler()

		#Record the duration of the sampler
		self.SampleDuration = (dt.datetime.utcnow() - utc_start).total_seconds()

	#Resample master sample
	def Resample(self):
		self.runSampler()

	#Resample master sample
	def runSampler(self):

		#Get Training and testing data separation
		self.Data = self.FullData.sample(frac = self.TargetRatio)
		self.TestData = self.FullData[~self.FullData.index.isin(self.Data.index)]

		#Orchestration package for sampler
		self.Sample = self.SamplerOrch()

		#Metadata about the composition of the sample
		if self.SampleMethod == "Standard":
			self.TotalRowNum = len(self.Sample.index)
			self.NonFraudRowNum = len(self.Data[self.Data.iloc[:,-1] != 1])
			self.FraudRowNum = len(self.Data[self.Data.iloc[:,-1] == 1].index)
			self.FraudSynthRowNum = 0
			self.FraudOrigRowNum = self.FraudRowNum

		#If sampling with SMOTE or Under-sampling
		else:
			self.TotalRowNum = len(self.Sample.index)
			self.NonFraudRowNum = int(self.TotalSize * (1-self.TargetRatio))
			self.FraudRowNum = self.TotalRowNum - self.NonFraudRowNum
			self.FraudSynthRowNum = max(self.FraudRowNum - len(self.Data[self.Data.iloc[:,-1] == 1].index), 0)
			self.FraudOrigRowNum = self.FraudRowNum - self.FraudSynthRowNum


	'''
	This is the SMOTE sampling engine. It synthetically creates records of the minority class
	(assumed to be class = 1) to bolster the robustness of the model. Its features can be 
	tuned from the constructor itself.

	ASSUMES that the sample of minority records is larger than the number truly existing.
	Otherwise, the model should simply use Undersampling.
	'''
	def SMOTESampler(self):

		#Separate records into fraudulent and non-fraudulent
		nonFraudRecords = self.Data[self.Data.iloc[:,-1] != 1]
		fraudRecords = self.Data[self.Data.iloc[:,-1] == 1]

		print(len(fraudRecords),len(nonFraudRecords))

		# Find the number of Non-fraudulent records and synthetic fraudulent records needed
		numNonFraudRecords = int(self.TotalSize * (1-self.TargetRatio))
		numSynthFraudRecords = max(int(self.TotalSize * self.TargetRatio) - len(fraudRecords.index), 0)

		#Add random sample of original records
		nonFraudSample = nonFraudRecords.sample(n = numNonFraudRecords)
		nonFraudRemaining = nonFraudRecords[~nonFraudRecords.index.isin(nonFraudSample.index)]

		#Test data
		self.TestData = pd.concat([self.TestData, nonFraudRemaining], ignore_index = True)
		self.Sample = self.Sample.append(nonFraudSample)

		#Randomly picks fraud records from the fraud records dataframe above (with replacement)
		randomFraudRecords = fraudRecords.loc[[random.choice(fraudRecords.index) for index in range(numSynthFraudRecords)],:]
		
		# Add a placeholder Index so that duplicates can still be compared after the index is dropped
		randomFraudRecords['placeholderIndex'] = randomFraudRecords.index
		randomFraudRecords.reset_index(drop = True, inplace = True)

		#For each randomly selected fraud record
		for index in range(len(randomFraudRecords)):

			#Select the reference record from the index
			referenceRecord = randomFraudRecords.iloc[index,:]

			#Removes any and all reference records from the reference DF to avoid self-matching
			referenceDF = randomFraudRecords[randomFraudRecords.placeholderIndex != referenceRecord.placeholderIndex]
			
			#Creates a new column in referenceDF (removing duplicates) that has the euclidean distance of the record from the 
			# reference record.
			referenceDF['euclid'] = referenceDF.drop_duplicates().apply(lambda row: euc(row.iloc[2:-2],referenceRecord.iloc[2:-2]), axis = 1)

			#Find the k nearest neighbors to the reference record
			NearestNeighbors = referenceDF.nsmallest(self.SMOTENeighbors,'euclid')
			
			#reset index for proper selection purposes (drop old one)
			NearestNeighbors.reset_index(drop = True, inplace = True)

			#Pick a random neighbor from the dataframe of k nearest neighbors
			randomNeighbor = NearestNeighbors.iloc[random.choice(NearestNeighbors.index),:-2]

			#Generate the differences between the neighbor and the reference record (multiplied by a 0-1 rand #)
			neighborDifferences = (randomNeighbor - referenceRecord[:-1]) * random.uniform(0,1)

			#Create the synthetic record by including these differences
			newSynthRecord = referenceRecord[:-1] + neighborDifferences

			#Add the record to the dataframe
			self.Sample = self.Sample.append(newSynthRecord, ignore_index = True)

		#Add all of the original fraud records to the dataframe
		self.Sample = self.Sample.append(fraudRecords, ignore_index = True)

		# Return Sample
		return self.Sample

	'''
	Undersampler assumes that the final sample will not need any synthetic fraud records. Therefore, it serves
	primarily the undersample (as the name implies) from the majority class. This way the impact
	of the fraudulent sample is much larger than it would be otherwise (possibly)
	'''		
	def UnderSampler(self):

		# Find the number of Non-fraudulent records and fraudulent records needed
		numNonFraudRecords = int(self.TotalSize * (1-self.TargetRatio))
		numFraudRecords = int(self.TotalSize * self.TargetRatio)

		#Create a random sample of the fraudulent and non-fraudulent data
		nonFraudRecords = self.Data[self.Data.iloc[:,-1] != 1].sample(numNonFraudRecords)

		#Remaining non-fraud records stored as testing data
		nonFraudRemaining = nonFraudRecords[~nonFraudRecords.index.isin(nonFraudSample.index)]
		self.TestData = pd.concat([self.TestData, nonFraudRemaining], ignore_index = True)

		#Use ALL fraud records
		fraudRecords = self.Data[self.Data.iloc[:,-1] == 1]

		#Add these random samples to the final Sample
		self.Sample = self.Sample.append(nonFraudRecords, ignore_index = True)
		self.Sample = self.Sample.append(fraudRecords, ignore_index = True)

		#Return sample
		return self.Sample
		

	'''
	Sampling orchestration package. This package takes input from the constructor and
	directs it to the correct sampling functions.
	'''
	def SamplerOrch(self):
		#Options for sampling
		if self.SampleMethod == "Standard":
			return self.Data
		elif self.SampleMethod == "SMOTE":
			return self.SMOTESampler()
		elif self.SampleMethod == "Under":
			return self.UnderSampler()

		#If someone asks for an unavailable sample
		else:
			print("ERROR: Bad sample method")


		


