import pandas as pd
import dmc.cleansing
import dmc.preprocessing
import dmc.classifiers
import dmc.evaluation

data_train = pd.read_csv('data/orders_train.txt', sep=';')
data_class = pd.read_csv('data/orders_class.txt', sep=';')
