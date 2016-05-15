from process import processed_data
from dmc.ensemble import ECEnsemble
from dmc.classifiers import Forest, SVM
from dmc.classifiers import TensorFlowNeuralNetwork as TensorNetwork
from dmc.transformation import scale_features as scaler
from dmc.transformation import scale_raw_features as raw_scaler
from dmc.features import add_dependent_features


quads = ['articleID', 'customerID', 'voucherID', 'productGroup']

""" Train on 100% (sampled), Classify on class data set """

data = processed_data(load_full=True)
train = data[~data.returnQuantity.isnull()]
test = data[data.returnQuantity.isnull()]
train, test = add_dependent_features(train, test)

# allocate Memory
del data

params = {
    # article, customer, productGroup
    'uuuu': {'sample': None, 'scaler': scaler, 'classifier': TensorNetwork},
    'uuuk': {'sample': 150000, 'scaler': raw_scaler, 'classifier': SVM},
    'uuku': {'sample': None, 'scaler': scaler, 'classifier': TensorNetwork},
    'uukk': {'sample': None, 'scaler': scaler, 'classifier': TensorNetwork},
    'ukuu': {'sample': 1000000, 'scaler': None, 'classifier': Forest},
    'ukuk': {'sample': 1200000, 'scaler': None, 'classifier': Forest},
    'ukku': {'sample': 1200000, 'scaler': None, 'classifier': Forest},
    'ukkk': {'sample': 1400000, 'scaler': None, 'classifier': Forest},
    'kuuk': {'sample': 150000, 'scaler': raw_scaler, 'classifier': SVM},
    'kukk': {'sample': None, 'scaler': scaler, 'classifier': TensorNetwork},
    'kkuk': {'sample': 1400000, 'scaler': None, 'classifier': Forest},
    'kkkk': {'sample': 1600000, 'scaler': None, 'classifier': Forest}
}

for k in params:
    params[k]['ignore_features'] = None

ensemble = ECEnsemble(train, test, params, quads)
print('transform for class')
ensemble.transform()
print('classify for class')
ensemble.classify(dump_results=True, dump_name='quadclass-ensemble-extreme')
