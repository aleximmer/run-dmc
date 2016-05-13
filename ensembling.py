from process import processed_data, split_data_at_id
from dmc.ensemble import ECEnsemble
from dmc.classifiers import Forest, SVM, TheanoNeuralNetwork
from dmc.classifiers import TensorFlowNeuralNetwork as TensorNetwork
from dmc.classifiers import NaiveBayes as Bayes
from dmc.transformation import scale_features as scaler
from dmc.transformation import scale_raw_features as raw_scaler


data = processed_data(load_full=True)
train, test = split_data_at_id(data, 1527394)
# allocate Memory

params = {
    'uuuu': {'sample': 200000, 'scaler': raw_scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'uuuk': {'sample': 200000, 'scaler': raw_scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'uuku': {'sample': None, 'scaler': scaler, 'ignore_features': ['returnQuantity', 'orderID', 'orderDate'], 'classifier': Bayes},
    'uukk': {'sample': 100000, 'scaler': raw_scaler, 'ignore_features': None, 'classifier': SVM},
    'ukuu': {'sample': 200000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'ukuk': {'sample': 200000, 'scaler': raw_scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'ukku': {'sample': 200000, 'scaler': raw_scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'ukkk': {'sample': 200000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'kuuu': {'sample': 200000, 'scaler': raw_scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'kuuk': {'sample': 200000, 'scaler': raw_scaler, 'ignore_features': None, 'classifier': SVM},
    'kuku': {'sample': 200000, 'scaler': raw_scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'kukk': {'sample': 200000, 'scaler': raw_scaler, 'ignore_features': None, 'classifier': TensorNetwork},
    'kkuu': {'sample': 200000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'kkuk': {'sample': 200000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'kkku': {'sample': 200000, 'scaler': None, 'ignore_features': None, 'classifier': Forest},
    'kkkk': {'sample': 200000, 'scaler': None, 'ignore_features': None, 'classifier': Forest}
}

for p in params:
    params[p]['sample'] = 10000
    params[p]['scaler'] = None

ensemble = ECEnsemble(train, test, params)
print('transform')
ensemble.transform()
print('classify')
ensemble.classify(dump_results=True)
