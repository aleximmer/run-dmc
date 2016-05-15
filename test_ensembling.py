from process import processed_data, split_data_at_id, shuffle
from dmc.ensemble import ECEnsemble
from dmc.classifiers import Forest, SVM
from dmc.classifiers import TensorFlowNeuralNetwork as TensorNetwork
from dmc.classifiers import NaiveBayes as Bayes
from dmc.transformation import scale_features as scaler
from dmc.transformation import scale_raw_features as raw_scaler


quads = ['articleID', 'customerID', 'voucherID', 'productGroup']

""" Train on 70% (sampled), Test on 30% and save test set """

data = processed_data(load_full=True)
data = data[~data.returnQuantity.isnull()]
train, test = split_data_at_id(data, 1527394)

# allocate Memory
del data



params = {
    # article, customer, productGroup
    'uuuu': {'sample': 200000, 'scaler': scaler, 'classifier': TensorNetwork, 'optimize': True},
    'uuuk': {'sample': 5, 'scaler': None, 'classifier': SVM},
    'uuku': {'sample': 200000, 'scaler': scaler, 'classifier': TensorNetwork, 'optimize': True},
    'uukk': {'sample': 200000, 'scaler': scaler, 'classifier': TensorNetwork, 'optimize': True},
    'ukuu': {'sample': 5, 'scaler': None, 'classifier': Forest},
    'ukuk': {'sample': 5, 'scaler': None, 'classifier': Forest},
    'ukku': {'sample': 5, 'scaler': None, 'classifier': Forest},
    'ukkk': {'sample': 5, 'scaler': None, 'classifier': Forest},
    'kuuk': {'sample': 5, 'scaler': None, 'classifier': SVM},
    'kukk': {'sample': 200000, 'scaler': scaler, 'classifier': TensorNetwork, 'optimize': True},
    'kkuk': {'sample': 5, 'scaler': None, 'classifier': Forest},
    'kkkk': {'sample': 5, 'scaler': None, 'classifier': Forest}
}

for k in params:
    params[k]['ignore_features'] = None
    params[k]['optimize'] = False

params['uuuu']['optimize'] = True
params['uuku']['optimize'] = True
params['uukk']['optimize'] = True
params['kukk']['optimize'] = True

# use bayes for the zero-split
params['ukuu']['classifier'] = Bayes

ensemble = ECEnsemble(train, test, params, quads)
print('transform for test')
ensemble.transform()
print('classify for test')
ensemble.classify(dump_results=False)
