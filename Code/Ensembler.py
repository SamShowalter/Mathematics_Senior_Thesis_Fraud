import copy
import numpy as np; 
import pandas as pd
import random
import os
import time
from math import inf
import operator

#os.chdir("Tests")

class Ensemble():

	def __init__(self, modeler, 
					   generationSize = 10, 
					   numGenerations = 100, 
					   timeoutMin = 0.2, 
					   ceiling = 0.49,
					   numWts = 5):
		self.GenerationSize = generationSize
		self.NumGenerations = numGenerations
		self.Modeler = modeler
		self.TimeoutMin = timeoutMin
		self.Ceiling = ceiling
		self.Fitness = inf
		self.NumWts = numWts
		self.BestWts = np.array(np.random.uniform(0,2**40,self.NumWts)).astype(int)

		print("\n\nEVOLUTION COMMENCING\n\n")
		#Set random seed
		#np.random.seed(random.randint(0,100))

		#self.evolve()

		#self.printCoefficients()

	#Determine the fitness of the specific weights
	def fitness(self, Wts):

		#Get ensemble predictions column
		self.Modeler.EnsemblePreds['ensemble_predicted'] = self.Modeler.EnsemblePreds.iloc[:,:5].apply(lambda row: round(sum(row*self.convertWts(Wts))), axis = 1)

		#Get true positive and false positive values for 
		TP = ((self.Modeler.EnsemblePreds.actual == 1) & (self.Modeler.EnsemblePreds.ensemble_predicted == 1)).value_counts().get(True,0)
		FP = ((self.Modeler.EnsemblePreds.actual == 1) & (self.Modeler.EnsemblePreds.ensemble_predicted == 0)).value_counts().get(True,0) 
		FN = ((self.Modeler.EnsemblePreds.actual == 0) & (self.Modeler.EnsemblePreds.ensemble_predicted == 1)).value_counts().get(True,0)

		#Get the total cost of the model
		fraudChargesFound = self.Modeler.EnsemblePreds[(self.Modeler.EnsemblePreds.actual == 1) & (self.Modeler.EnsemblePreds.ensemble_predicted == 1)].loc[:,'amount'].sum()
		fraudChargesLost = self.Modeler.EnsemblePreds[(self.Modeler.EnsemblePreds.actual == 1) & (self.Modeler.EnsemblePreds.ensemble_predicted == 0)].loc[:,'amount'].sum()

		#Return the fraud cost
		return self.Modeler.fraudCost(TP,fraudChargesFound,FP,fraudChargesLost,FN)

	#Pads binary string so that it is always 8 characters
	def padBinString(self,num):
		temp_str = str(bin(num))[2 :] 							# String of number
		final_string = '0' * (40 - len(temp_str)) + temp_str 	# zero-padding plus string number

		return final_string

	#Converts an array of values to a binary string
	def convertToBinary(self,arr):

		#Initialize variables and empty final string
		final_var_string = ""

		#Iterate through variables
		for wt in arr:
			#Add the padded binary string to final
			final_var_string += self.padBinString(wt)

		return final_var_string

	#Convert a binary string to an integer array
	def convertBinToVars(self, kid):
		#Initialize array of all ones
		arr = np.array([1]*(self.NumWts))

		#For each variable
		for var in range(self.NumWts):
			arr[var] = int(kid[(var)*40:(var+1)*40], base = 2)

		return arr.astype(int)


	#Use combination to generate a child
	def generateChild(self, dad, mom):
		#Get middle index
		mid_index = len(dad) // 2

		# Get random number indices for each half
		mut_index_first_half = np.random.randint(0, mid_index)
		mut_index_second_half = np.random.randint(mid_index, len(dad))

		#Create two children with crossover
		kid1 = dad[: mut_index_first_half] + mom[mut_index_first_half:mut_index_second_half] + dad[mut_index_second_half:]
		kid2 = mom[: mut_index_first_half] + dad[mut_index_first_half:mut_index_second_half] + mom[mut_index_second_half:]

		#Store potential kids for future random choosing
		potential_kids = [kid1,kid2]

		#choose a random child from potential kids
		return potential_kids[np.random.randint(0,2)]

	#Mutate randomly in a population
	def mutate(self,spawn,oddsOfMutation):

		#For each child in the generation
		for offspring in range(self.GenerationSize):
			#Randomly choose a child
			randKid = np.random.randint(0,self.GenerationSize)

			#Potentially mutate that child based on the odds of mutation ()
			spawn[randKid] = self.mutateOne(spawn[randKid],oddsOfMutation)

		return spawn


	#Mutate random children
	def mutateOne(self, kid, oddsOfMutation):
		# randomly select how many times to mutate
		# during each time, randomly select a bit to mutate
		def subMutateOneV1(kid):
			#Random number of mutation changes
			mutNum = np.random.randint(len(kid))

			#Mutate the child "changeNum" times
			for change in range(mutNum):
				#Random index
				randIndex = np.random.randint(len(kid))

				if kid[randIndex] == "1":
					kid = kid[:randIndex] + "0" + kid[randIndex + 1 :]
				else:
					kid = kid[:randIndex] + "1" + kid[randIndex + 1:]

			return kid

		# randomly select a pair of indices
		# flip all bits between those two indices
		def subMutateOneV2(kid):
			index1 = np.random.randint(len(kid))
			index2 = np.random.randint(len(kid))

			index1, index2 = min(index1, index2), max(index1, index2)

			for index in range(index1, index2):
				if kid[index] == '0':
					kid = kid[: index] + '1' + kid[index + 1 :]
				else:
					kid = kid[: index] + '0' + kid[index + 1 :]

			return kid

		# randomly select a pair of indices
		# randomly decide whether to flip each bit
		# between those two indices
		def subMutateOneV3(kid):
			index1 = np.random.randint(len(kid))
			index2 = np.random.randint(len(kid))

			index1, index2 = min(index1, index2), max(index1, index2)

			for index in range(index1, index2):
				flip = np.random.randint(2)
				if flip == 0:
					if kid[index] == '0':
						kid = kid[: index] + '1' + kid[index + 1 :]
					else:
						kid = kid[: index] + '0' + kid[index + 1 :]

			return kid

		methods = [subMutateOneV1, subMutateOneV2, subMutateOneV3]

		#Turn kid into binary
		kid = self.convertToBinary(kid)

		randomChance = np.random.randint(0,oddsOfMutation)

		#Only do it if the random Chance equals 0
		if randomChance == 0:

			methodID = np.random.randint(len(methods))
			kid = methods[methodID](kid)

		#Put child back in population
		return self.convertBinToVars(kid)

	def createOffspring(self,spawn, fitness, generationSize):
		newspawn = []
		for kid in range(generationSize):
			dad, mom = self.getParents(spawn, fitness)
			newspawn.append(self.setCeiling(self.convertBinToVars(self.generateChild(dad,mom))))

		return newspawn

	def chooseParentIndex(self,fitChart):
		rNum = np.random.uniform(0,1)

		for i in range(len(fitChart)):
			if fitChart[i] >= rNum:
				return (i - 1)

		#If it was exactly one
		return len(fitChart) - 1

	#Cumulate (sum) distribution of fitnesses
	def getFitChart(self,fitnessPerc):
		fitChart = [fitnessPerc[0]]
		for i in range(1,len(fitnessPerc)):
			fitChart.append(sum(fitnessPerc[0:i+1]))

		return fitChart


	def setCeiling(self, kid):
		new_kid = np.array([1]*(self.NumWts))

		#Make sure everything abides by the new ceiling
		for num in range(len(kid)):
			new_kid[num] = self.putInBounds(kid[num])
		#Return the new kid
		return new_kid

	#Put a weight value in bounds (must be less than half)
	def putInBounds(self,num):
		if num < 0:
			return 0
		elif num > 2**40 - 1:
			return s**40 - 1

		#Return the ceiling number or less
		return min(num, int(sum(self.BestWts) * self.Ceiling))

	#Convert discrete weights into percentages
	def convertWts(self, Wts):
		sum_wts = sum(Wts)
		return Wts / sum_wts

	#Fix inbreeding if the spawn grow stale
	def fixInbreeding(self):
		new_spawn = []

		#Re-populate the spawn
		for kid in range(self.GenerationSize):

			#CHANGED WEIGHTS HERE
			new_kid = np.array([1]*(self.NumWts))
			for i in range(self.NumWts):
				new_kid[i] = self.putInBounds(np.random.normal(self.BestWts[i],10000000))
			new_spawn.append(new_kid)


		return new_spawn

	# Find the parents from a population probabilistically

	def getParents(self, spawn, fitness):

		#Get fitness percentages (all non negative)
		pos_fitness = [(fit + (abs(min(fitness)) + 1)) for fit in fitness]

		#Subtract fitness (want it low) from the max in the array + 1
		converted_fitness = [(max(pos_fitness) + 1) - fit for fit in pos_fitness]
		fitness_perc = (converted_fitness / sum(converted_fitness))

		#Get the fitness chart
		fit_chart = self.getFitChart(fitness_perc)
		#print(" ".join([str(i) for i in fitChart]))

		while(True):
			dadIndex, momIndex = self.chooseParentIndex(fit_chart), self.chooseParentIndex(fit_chart)
			if dadIndex != momIndex:
				return self.convertToBinary(spawn[dadIndex]), self.convertToBinary(spawn[momIndex])


	def evolve(self):

		#Start time for execution
		startTime = time.time()

		#How often mutations should occur
		mutateOccurence = 10

		#Count of how many generations had no change
		noChange = 0

		#Initialize container for spawn
		spawn = []

		#Create initial random Offspring
		for offspring in range(self.GenerationSize):
			spawn.append(np.array(np.random.randint(0,2**40, size = self.NumWts)))

		#Generate the fitness of each child
		fitness = [self.fitness(coeffs) for coeffs in spawn]

		#Pick parents randomly based on fitness
		dad, mom = self.getParents(spawn, fitness)

		#Print header information for output
		print('{:<5}  {:<10} {:<15}  {:<98}'.format("Gen", "Timer", "Fitness", "Variables"))

		#Evolve the sample
		for generation in range(self.NumGenerations):
			hyper_spawn = []

			#Generate children
			#print(spawn)
			spawn = self.createOffspring(spawn,fitness,self.GenerationSize)
			spawn = self.mutate(spawn,mutateOccurence)

		 	#Generate the fitness of each child
			fitness = [self.fitness(kid) for kid in spawn]

			#Update max fitness if better option is found
			if (min(fitness) < self.Fitness):

				#Update Fitness and variable values
				self.GensToBest = generation
				self.Fitness = min(fitness)
				self.BestWts = spawn[fitness.index(self.Fitness)]

			#Update spawn if population grows stale after awhile
			else:
				noChange += 1
				if noChange == 5:
					noChange = 0
					spawn = self.fixInbreeding()


			#Add best candidate to hyper_spawn
			hyper_spawn.append(spawn[fitness.index(max(fitness))])

			#Check to see if hyperspawn should replace spawn
			if len(hyper_spawn) == self.GenerationSize:
				spawn = hyperspawn

			#Calculate time left
			time_left = (self.TimeoutMin * 60) - int(time.time() - startTime)

			#Print out the fitness
			outputInfo = '{:<5} {:<10} {:<16}  {:<100}'.format(generation,
														str((self.TimeoutMin * 60) - int(time.time() - startTime)),
														"{0:.2f}".format(round(self.Fitness,2)),
														str(self.convertWts(self.BestWts)))
			print(outputInfo)

			#If the timer has run out, end the cycle
			if (time_left < 0):
				print("\nTIMEOUT REACHED BEFORE GENERATIONS FINISHED. EXITING.")
				break


		#Print output
		print("\nEVOLUTION FINISHED.\n\n" +
			  "Generations Until Optimum Found: " + str(self.GensToBest) +
			  "\nMaximum Fitness: " + str(self.Fitness) +
			  "\nBest Variables: " + str(self.convertWts(self.BestWts)))

		return self.convertWts(self.BestWts)

	

