from process import processed_data, split_data_by_id
from dmc.ensemble import Ensemble
from dmc.classifiers import Forest, DecisionTree, NaiveBayes


evaluation_sets = ['rawSummerSale', 'rawNovDec', 'rawPopularCustomers', 'rawFirstOrders',
                   'rawLargestProductGroup', 'rawLinearSample', 'rawPopularCustomers']

data = processed_data()

for eval_set in evaluation_sets:
    print('================================')
    print('Train and evaluate', eval_set)
    train, test = split_data_by_id(data, eval_set)
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
