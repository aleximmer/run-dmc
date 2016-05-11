from process import processed_data, split_data_by_id
import pandas as pd
import numpy as np
from dmc.ensemble import Ensemble
from dmc.classifiers import Forest, DecisionTree, NaiveBayes


evaluation_sets = ['rawMirrored', 'rawLinearSample']

data = processed_data()


def shuffle(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(np.random.permutation(df.index))

for eval_set in evaluation_sets:
    print('================================')
    print('Train and evaluate', eval_set)
    train, test = split_data_by_id(data, eval_set)
    train = shuffle(train)[:250000]
    print('train', len(train), 'test', len(test))
    ensemble = Ensemble(train, test)
    ensemble.transform(binary_target=True)
    ensemble.classify(classifiers=[Forest] * len(ensemble.splits),
                      verbose=True)

for eval_set in evaluation_sets:
    print('================================')
    print('Train and evaluate', eval_set)
    train, test = split_data_by_id(data, eval_set)
    ensemble = Ensemble(train, test)
    ensemble.transform(binary_target=True)
    ensemble.classify(classifiers=[Forest] * len(ensemble.splits),
                      hyper_param=True, verbose=False)
