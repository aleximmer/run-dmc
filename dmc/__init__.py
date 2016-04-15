import pandas as pd
import dmc.cleansing
import dmc.preprocessing

data_train = pd.read_csv('data/orders_train.txt', sep=';')
data_class = pd.read_csv('data/orders_class.txt', sep=';')
