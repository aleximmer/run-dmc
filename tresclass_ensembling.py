from process import processed_data
from dmc.ensemble import ECEnsemble
from dmc.classifiers import Forest
from dmc.classifiers import TensorFlowNeuralNetwork as TensorNetwork
from dmc.transformation import scale_features as scaler
from dmc.features import add_dependent_features


trits = ['articleID', 'customerID', 'productGroup']

""" Train on 100% (sampled), Classify on class data set """

data = processed_data(load_full=True)
train = data[~data.returnQuantity.isnull()]
test = data[data.returnQuantity.isnull()]
train, test = add_dependent_features(train, test)

# allocate Memory
del data

params = {
    # article, customer, productGroup
    'uuu': {'sample': None, 'scaler': scaler, 'classifier': TensorNetwork},
    'uku': {'sample': 750000, 'scaler': None, 'classifier': Forest},
    'kuu': {'sample': None, 'scaler': scaler, 'classifier': TensorNetwork},
    'kuk': {'sample': None, 'scaler': scaler, 'classifier': TensorNetwork},
    'kku': {'sample': 750000, 'scaler': None, 'classifier': Forest},
    'kkk': {'sample': 750000, 'scaler': None, 'classifier': Forest}
}

for p in params:
    params[p]['ignore_features'] = None

ensemble = ECEnsemble(train, test, params, trits)
print('transform for class')
ensemble.transform()
print('classify for class')
ensemble.classify(dump_results=True, dump_name='tresclass-ensemble')
