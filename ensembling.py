from process import processed_data, split_data_at_id
from dmc.ensemble import ECEnsemble
from dmc.classifiers import Forest, SVM, TheanoNeuralNetwork
from dmc.classifiers import TensorFlowNeuralNetwork as TensorNetwork
from dmc.classifiers import NaiveBayes as Bayes
from dmc.transformation import scale_features as scaler
from dmc.transformation import scale_raw_features as raw_scaler


""" Train on 70%, Test on 30% and save test set """

data = processed_data(load_full=True)
data = data[~data.returnQuantity.isnull()]
train, test = split_data_at_id(data, 1527394)
# allocate Memory
del data

params = {
    'uuuu': {'sample': 200000, 'scaler': scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'uuuk': {'sample': None, 'scaler': scaler, 'ignore_features': None, 'classifier': Bayes},
    'uuku': {'sample': 200000, 'scaler': scaler, 'ignore_features': ['returnQuantity', 'orderID', 'orderDate'], 'classifier': TensorNetwork},
    'uukk': {'sample': 100000, 'scaler': raw_scaler, 'ignore_features': None, 'classifier': SVM},
    'ukuu': {'sample': 200000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'ukuk': {'sample': 200000, 'scaler': scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'ukku': {'sample': 200000, 'scaler': scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'ukkk': {'sample': 200000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'kuuu': {'sample': 200000, 'scaler': scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'kuuk': {'sample': 200000, 'scaler': raw_scaler, 'ignore_features': None, 'classifier': SVM},
    'kuku': {'sample': 200000, 'scaler': scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'kukk': {'sample': 300000, 'scaler': scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'kkuu': {'sample': 200000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'kkuk': {'sample': 300000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'kkku': {'sample': 300000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'kkkk': {'sample': 400000, 'scaler': None, 'ignore_features': None, 'classifier': Forest}
}

# use bayes for the zero-split
params['xa']['classifier'] = Bayes

# remove!
for p in params:
    params[p]['sample'] = 10000

ensemble = ECEnsemble(train, test, params)
print('transform for test')
ensemble.transform()
print('classify for test')
ensemble.classify(dump_results=True, dump_name='test')


""" Train on 100%, Classify on class data set """

data = processed_data(load_full=True)
train = data[data.returnQuantity.isnull()]
test = data[~data.returnQuantity.isnull()]

# allocate Memory
del data

params = {
    'uuuu': {'sample': 200000, 'scaler': scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'uuuk': {'sample': None, 'scaler': scaler, 'ignore_features': None, 'classifier': Bayes},
    'uuku': {'sample': 200000, 'scaler': scaler, 'ignore_features': ['returnQuantity', 'orderID', 'orderDate'], 'classifier': TensorNetwork},
    'uukk': {'sample': 100000, 'scaler': raw_scaler, 'ignore_features': None, 'classifier': SVM},
    'ukuu': {'sample': 200000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'ukuk': {'sample': 200000, 'scaler': scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'ukku': {'sample': 200000, 'scaler': scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'ukkk': {'sample': 200000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'kuuu': {'sample': 200000, 'scaler': scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'kuuk': {'sample': 200000, 'scaler': raw_scaler, 'ignore_features': None, 'classifier': SVM},
    'kuku': {'sample': 200000, 'scaler': scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'kukk': {'sample': 200000, 'scaler': scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'kkuu': {'sample': 200000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'kkuk': {'sample': 200000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'kkku': {'sample': 200000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'kkkk': {'sample': 200000, 'scaler': None, 'ignore_features': None, 'classifier': Forest}
}

# use bayes for the zero-split
params['xa']['classifier'] = Bayes

for p in params:
    params[p]['sample'] = 10000

ensemble = ECEnsemble(train, test, params)
print('transform for class')
ensemble.transform()
print('classify for class')
ensemble.classify(dump_results=True, dump_name='class')
