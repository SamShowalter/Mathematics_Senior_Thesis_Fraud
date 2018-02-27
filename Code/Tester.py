import pandas as pd 
import numpy as np 
from Logger import Log
import datetime as dt
import copy

class Test():

	def __init__(self,sampler,modeler,Log):
		#Time metadata
		self.ExecutionDateStart = dt.datetime.now().date()
		self.ExecutionTimeStart = dt.datetime.now().time()
		utc_start = dt.datetime.utcnow()

		#Initialize log
		self.Log = Log

		#Initialize sampler
		self.Sample = sampler

		#Sample duration information
		self.SampleDuration = (dt.datetime.utcnow() - utc_start).total_seconds()

		#Initialize modeler
		self.Modeler = modeler

		#Add sample to modeler and run model
		self.Modeler.setSample(self.Sample)
		self.Modeler.run_model()

		#Duration metadata
		self.TestDuration = (dt.datetime.utcnow() - utc_start).total_seconds()
		self.ModelDuration = self.TestDuration - self.SampleDuration

		#Add masterlog record
		self.Log.addMasterLogRecord(self)











