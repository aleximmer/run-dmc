import process as p
import pandas as pd
import dmc

df = p.processed_data()
start, end, split = pd.Timestamp('2014-1-1'), pd.Timestamp('2014-12-31'), pd.Timestamp('2014-10-1')
df.orderDate = pd.to_datetime(df.orderDate)
mask = (df.orderDate >= start) & (df.orderDate <= end)
df_full = df[mask]
te_size = 10000
tr_size = 10000
X, Y, fts = dmc.transformation.transform_preserving_header(df_full, scaler=dmc.transformation.scale_features, binary_target=True)

n = 10
impact_map = dict()
feature_set = set(fts)
for feature in fts:
    impact_map[feature] = 0.

for j in range(n):
    print(j)
    df = p.shuffle(df)
    start, end, split = pd.Timestamp('2014-1-1'), pd.Timestamp('2014-12-31'), pd.Timestamp('2014-10-1')
    df.orderDate = pd.to_datetime(df.orderDate)
    mask = (df.orderDate >= start) & (df.orderDate <= end)
    df_full = df[mask]
    te_size = 10000
    tr_size = 10000
    X, Y, fts = dmc.transformation.transform_preserving_header(df_full, scaler=dmc.transformation.scale_features, binary_target=True)
    train = X[:tr_size], Y[:tr_size]
    test = X[tr_size:tr_size+te_size], Y[tr_size:tr_size+te_size]

    res_tree = dmc.evaluation.evaluate_features_leaving_one_out(train[0], train[1], test[0], test[1], fts, dmc.classifiers.DecisionTree)
    for feature in feature_set:
            impact_map[feature] = impact_map[feature] + res_tree.decrement[feature]
            
for key, value in impact_map.items():
    impact_map[key] = value / n

import operator
sorted_impact_list = sorted(impact_map.items(), key=operator.itemgetter(1))
print(sorted_impact_list)
