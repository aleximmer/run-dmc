from process import processed_data, split_data_at_id
from dmc.ensemble import ECEnsemble
from dmc.classifiers import Forest, SVM
from dmc.classifiers import TensorFlowNeuralNetwork as TensorNetwork
from dmc.classifiers import NaiveBayes as Bayes
from dmc.transformation import scale_features as scaler
from dmc.transformation import scale_raw_features as raw_scaler


""" Train on 70% (sampled), Test on 30% and save test set """

data = processed_data(load_full=True)
data = data[~data.returnQuantity.isnull()]
train, test = split_data_at_id(data, 1527394)

# allocate Memory
del data

params = {
    # article, customer, productGroup
    'uuuu': {'sample': 500000, 'scaler': scaler, 'classifier': TensorNetwork},
    'uuuk': {'sample': 100000, 'scaler': raw_scaler, 'classifier': SVM},
    'uuku': {'sample': 500000, 'scaler': scaler, 'classifier': TensorNetwork},
    'uukk': {'sample': 100000, 'scaler': raw_scaler, 'classifier': SVM},
    'ukuu': {'sample': 500000, 'scaler': None, 'classifier': Forest},
    'ukuk': {'sample': 500000, 'scaler': None, 'classifier': Forest},
    'ukku': {'sample': 500000, 'scaler': scaler, 'classifier': Forest},
    'ukkk': {'sample': 500000, 'scaler': None, 'classifier': Forest},
    'kuuk': {'sample': 100000, 'scaler': raw_scaler, 'classifier': SVM},
    'kukk': {'sample': 500000, 'scaler': None, 'classifier': Forest},
    'kkuk': {'sample': 500000, 'scaler': None, 'classifier': Forest},
    'kkkk': {'sample': 750000, 'scaler': None, 'classifier': Forest}
}

for k in params:
    params[k]['ignore_features'] = None

# use bayes for the zero-split
params['ukuu']['classifier'] = Bayes

ensemble = ECEnsemble(train, test, params)
print('transform for test')
ensemble.transform()
print('classify for test')
ensemble.classify(dump_results=True, dump_name='quadtest-ensemble')
