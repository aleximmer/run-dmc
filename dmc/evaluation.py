import numpy as np
import pandas as pd

from dmc.transformation import transform_feature_matrix_ph, transform_target_vector
from dmc.classifiers import Forest, DecisionTree


def dmc_cost(predicted: np.array, ground_truth: np.array) -> int:
    """Cost function defined for the DMC"""
    diff = np.abs(predicted - ground_truth)
    return np.sum(diff)


def dmc_cost_relative(predicted: np.array, ground_truth: np.array) -> float:
    cost = dmc_cost(predicted, ground_truth)
    return cost / len(predicted)


def precision(predicted: np.array, ground_truth: np.array) -> int:
    diff = predicted - ground_truth
    return 1 - np.count_nonzero(diff) / len(predicted)


def gini_ratio(arr: pd.Series) -> float:
    """Return impurity of array
    """
    _, counts = np.unique(arr, return_counts=True)
    squared_ratio = np.vectorize(lambda count: np.square(np.divide(count, len(arr))))
    return 1.0 - np.sum(squared_ratio(counts))


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a MultiIndex'd (col, val) DataFrame
    with gini impurity, average return quantity, and return probability
    """
    purities = pd.DataFrame(columns=['count', 'gini', 'retProb', 'avgRet', 'stdRet'],
                            index=pd.MultiIndex(labels=[[], []], levels=[[], []],
                                                names=['column', 'value']))
    feature_cols = df.drop('returnQuantity', axis=1).columns

    def column_info(labels: pd.Series) -> pd.Series:
        count = len(labels)
        gini = gini_ratio(labels)
        prob = labels.astype(bool).sum() / len(labels)
        avg = labels.mean()
        std = labels.std()
        return count, gini, prob, avg, std

    for col in feature_cols:
        value_infos = df.groupby(col)['returnQuantity'].apply(column_info)
        for i, row in value_infos.iteritems():
            purities.loc[(col, i), :] = row

    return purities


def column_purities(df: pd.DataFrame) -> pd.Series:
    feature_cols = df.drop('returnQuantity', axis=1).columns
    purities = pd.Series(None, index=feature_cols)

    def weighted_gini(group: pd.DataFrame) -> float:
        return len(group) / len(df) * gini_ratio(group['returnQuantity'])

    for col in feature_cols:
        summed_gini = df.groupby(col).apply(weighted_gini).sum()
        purities[col] = summed_gini

    return purities


def evaluate_features_by_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe giving each feature an importance factor"""
    X, fts = transform_feature_matrix_ph(df)
    Y = transform_target_vector(df)
    forest = Forest(X, Y).fit()
    tree = DecisionTree(X, Y).fit()
    ft_eval = pd.DataFrame({
        'feature': fts,
        'forest': forest.clf.feature_importances_,
        'tree': tree.clf.feature_importances_,
    })
    return ft_eval.groupby('feature').sum()


def evaluate_features_leaving_one_out(X_train, Y_train, X_class, Y_class,
                                      feature_header, classifier) -> pd.DataFrame:
    """Based on training and testing data features are left out and
    performance is measured without those features.

    A Dataframe with precision deltas and precision will be returned
    """
    clf = classifier(X_train, Y_train)
    baseline = precision(Y_class, clf(X_class))
    ftsl, seen = list(feature_header), set()
    res = pd.DataFrame(index=set(ftsl + ['all']),
                       data={'decrement': 0., 'precision': 0.})
    res.precision['all'] = baseline
    for ft in ftsl:
        if ft in seen:
            continue
        else:
            seen.add(ft)
        X_tr, X_cl = X_train.T[feature_header != ft].T, X_class.T[feature_header != ft].T
        clf = classifier(X_tr, Y_train)
        prec = precision(Y_class, clf(X_cl))
        res.decrement[ft] = baseline - prec
        res.precision[ft] = prec
    return res


def evaluate_without_one_feature(X_train, Y_train, X_class, Y_class, feature_header, ignore_features, classifier, n) -> pd.DataFrame:
    global_baseline = 0
    global_precision = 0
    for i in range(n):
        clf = classifier(X_train, Y_train)
        baseline = precision(Y_class, clf(X_class))
        ftsl, seen = list(feature_header), set()
        res = pd.DataFrame(index=set(ftsl + ['all']),
                           data={'decrement': 0., 'precision': 0.})
        res.precision['all'] = baseline

        X_tr, X_cl = X_train.T[feature_header != ignore_features].T, X_class.T[feature_header != ignore_features].T
        clf = classifier(X_tr, Y_train)
        prec = precision(Y_class, clf(X_cl))
        global_baseline += baseline
        global_precision += precision
    return global_baseline/n, global_precision/n
