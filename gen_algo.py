
import numpy as np
import pandas as pd
import random as rdm
import dmc
from dmc.classifiers import DecisionTree, Forest, NaiveBayes, SVM, NeuralNetwork
from dmc.classifiers import TreeBag, BayesBag, SVMBag
from dmc.classifiers import AdaTree, AdaBayes, AdaSVM

classifierSet = {'Forest' : Forest, 'NaiveBayes' : NaiveBayes, 'SVM' : SVM, 'TreeBag' : TreeBag, 'BayesBag' : BayesBag, 'AdaTree' : AdaTree, 'AdaBayes' : AdaBayes}

populationSize = 20
keepIn = 4
generate = populationSize - keepIn
mutationProb = 0.1
fitnessSetSize = 2
generations = 20

te_size = 5000
tr_size = 5000

dont_use = ['returnQuantity', 'orderID', 'orderDate', 'customerID', 'Unnamed: 0', 'Unnamed']

def getFeatureSet(fs: np.array, gene: np.array) -> np.array:	
	return fs[gene > 0]


def shuffle(df: pd.DataFrame) -> pd.DataFrame:
	return df.reindex(np.random.permutation(df.index))


def getAcc(test, train, classifier):
	clf = classifier(train[0], train[1])
	res = clf(test[0])
	return dmc.evaluation.precision(res, test[1])


def getEncodedSplit(data: pd.DataFrame):	
	data = shuffle(data)	
	data = data[:te_size + tr_size]
	X, Y = dmc.transformation.transform(data, scaler=dmc.normalization.scale_features, binary_target=True)
	train = X[:tr_size], Y[:tr_size]
	test = X[tr_size:tr_size + te_size], Y[tr_size:tr_size + te_size]
	return (train, test)

def adjustFitness(size, value):
	#dummy function but may be needed
	return value


def getFitness(genes: np.array, features: np.array, data: pd.DataFrame, classifier) -> np.array:
	fit = np.zeros((genes.shape[0], fitnessSetSize))
	for i in range(0, genes.shape[0]):	
		useFtS = getFeatureSet(features, genes[i])
		#data = shuffle(data)
		#rQ = data['returnQuantity']		
		usedData = data[useFtS]
		usedData['returnQuantity'] = data['returnQuantity']
		for j in range(0, fitnessSetSize):
			train, test = getEncodedSplit(usedData)
			#print(str(i) + "." + str(j))			
			fit[i, j] = getAcc(test, train, classifier)
	#print("fitness values:\n" + str(fit))	
	return np.array([adjustFitness(np.sum(f), np.mean(f)) for f in fit])	


def getRanking(genes: np.array, fitness: np.array) -> np.array:
	return (genes[np.argsort(fitness)][::-1], np.sort(fitness)[::-1])


def doCrossover(gene1: np.array, gene2: np.array):	
	transfer = np.zeros(gene1.size)
	transfer[rdm.sample(range(0, gene1.size), int(gene1.size/2))] = 1
	b = transfer == 1
	result1 = gene1
	result2 = gene2
	result1[b] = gene2[b]
	result2[b] = gene1[b]
	return (result1, result2)


def generateGenes(seeds: np.array) -> np.array:
	generated = []	
	while len(generated) < generate:
		useIndex = rdm.sample(range(0, keepIn), 2)	
		crossed = doCrossover(seeds[useIndex[0]], seeds[useIndex[1]])
		generated.append(crossed[0])
		generated.append(crossed[1])
	return np.array(generated)


def mutateGenes(genes: np.array) -> np.array:
	for i in range(0, genes.shape[0]):
		for j in range(0, genes.shape[1]):
			mutate = rdm.random()
			if mutate <= mutationProb:				
				genes[i,j] = 1-genes[i,j]
	return genes


def createStartPopulation(fs: np.array) -> np.array:
	startGenes = []
	while len(startGenes) < populationSize:
		geneSize = rdm.randint(1, fs.size)
		index = rdm.sample(range(0, fs.size), geneSize)
		gene = np.zeros(fs.size)
		gene[index] = 1		
		if not any((g == gene).all() for g in startGenes): 
			startGenes.append(gene)
	return np.array(startGenes)


def getData() -> pd.DataFrame: 
	return pd.read_csv('data/processed.csv', sep=',', na_values='\\N')


def prepareFeatureList(df: pd.DataFrame) -> np.array:
	return np.array([f for f in df.columns.tolist() if not f in dont_use])	


def main(number, classifierName, classifier, data, features):
	
	featureSetTable = pd.DataFrame(columns = np.append(features, ['accuracy']))
	featureScoreTable = pd.DataFrame({'feature' : features, 'score' : np.zeros(features.size), 'relative_score' : np.zeros(features.size)})	

	genes = createStartPopulation(features)
	print("genes: " + str(populationSize) + " generations: " + str(generations) + " keep: " + str(keepIn) + " mutationProb: " + str(mutationProb))
	
	for j in range(1, generations+1):
		currentFitness = getFitness(genes, features, data, classifier)
		(currentRanking, currentFitness) = getRanking(genes, currentFitness)
		genes[0:keepIn] = currentRanking[0:keepIn]
		genes[keepIn:populationSize] = mutateGenes(generateGenes(genes[0:keepIn]))
		for i in range(0, keepIn):
			featureScoreTable['score'] = featureScoreTable['score'] + currentRanking[i]*currentFitness[i]
		featureSetTable.loc[featureSetTable.shape[0]] = np.append(currentRanking[0], [currentFitness[0]]) 	
		print("classifier: " + classifierName + " test: " + str(number) + " generation: " + str(j) + " top-score: " + str(np.max(currentFitness)) + " top-size: " + str(np.sum(genes[0])))	

	featureScoreTable['relative_score'] = featureScoreTable['score']/ np.sum(featureScoreTable['score'])
	featureSetTable.to_csv('Sets_' + classifierName + '_' + str(number) + '.csv', sep=',')
	featureScoreTable.to_csv('Score_' + classifierName + '_' + str(number) + '.csv', sep=',')

pd.options.mode.chained_assignment = None 
data = getData()
print('data loaded')
features = prepareFeatureList(data)
print(str(features.size) + " features in pool")
for i in range(0,1):
	for name, classifier in classifierSet.items():
		main(i, name, classifier, data, features)
