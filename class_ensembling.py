from process import processed_data
from dmc.ensemble import ECEnsemble
from dmc.classifiers import Forest, SVM, TheanoNeuralNetwork
from dmc.classifiers import TensorFlowNeuralNetwork as TensorNetwork
from dmc.classifiers import NaiveBayes as Bayes
from dmc.transformation import scale_features as scaler
from dmc.transformation import scale_raw_features as raw_scaler
from dmc.features import add_dependent_features


""" Train on 100%, Classify on class data set """

data = processed_data(load_full=True)
train = data[data.returnQuantity.isnull()]
test = data[~data.returnQuantity.isnull()]
train, test = add_dependent_features(train, test)

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
    'kuuk': {'sample': 200000, 'scaler': raw_scaler, 'ignore_features': None, 'classifier': TheanoNeuralNetwork},
    'kuku': {'sample': 200000, 'scaler': scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'kukk': {'sample': 300000, 'scaler': scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'kkuu': {'sample': 200000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'kkuk': {'sample': 250000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'kkku': {'sample': 250000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'kkkk': {'sample': 300000, 'scaler': None, 'ignore_features': None, 'classifier': Forest}
}

ensemble = ECEnsemble(train, test, params)
print('transform for class')
ensemble.transform()
print('classify for class')
ensemble.classify(dump_results=True, dump_name='class')
