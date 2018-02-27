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
		utc_exec_start = dt.datetime.utcnow()

		#Initialize log
		self.Log = Log

		#Initialize sampler
		self.Sample = sampler

		#Time the modeler
		utc_model_start = dt.datetime.utcnow()
		
		#Initialize modeler
		self.Modeler = modeler

		#Add sample to modeler and run model
		self.Modeler.setSample(self.Sample)
		self.Modeler.run_model()

		#Duration metadata
		self.ModelDuration = (dt.datetime.utcnow() - utc_model_start).total_seconds()
		self.TestDuration = (dt.datetime.utcnow() - utc_exec_start).total_seconds()
		

		#Add masterlog record
		self.Log.addMasterLogRecord(self)











