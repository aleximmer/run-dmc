
# coding: utf-8

# In[1]:

# In[2]:

import dmc
from process import *


# In[3]:

evaluation_sets = ['rawMirrored', 'rawLinearSample']
data = processed_data()


# In[4]:

train, test = split_data_by_id(data, 'rawMirrored')


# In[5]:

len(train), len(test)


# In[6]:

ensemble = dmc.ensemble.Ensemble(train[:4000], test[:1000])


# In[7]:
# pr


# In[8]:

import csv
import pandas as pd

file_name = 'ArticleProductgroupCustomer.csv'
with open('/home/team_c/fabian_p/Sets_SVM_' +
                  file_name) as csv_file:
    for index, row in enumerate(csv.reader(csv_file, delimiter=',')):
        if index == 0:

            series_a = pd.Series(row[1:])
        else:
            series_b = pd.Series(row[1:])
feature_table = pd.DataFrame({'feature': series_a, "value": series_b})
unimp_fts = feature_table[feature_table['value'] == '0.0']['feature'].tolist()
# print(unimp_fts)


# In[9]:

dropped_fts = ['t_singleItemPrice','t_isTypePants','t_isTypeBelt','t_isTypeTop'
,'t_isWeekend_A','t_isOneSize_A','t_voucher_is10PercentVoucher'
,'t_voucher_is15PercentVoucher','customerAvgUnisize', 'voucherID'] 
for drop in dropped_fts:
    unimp_fts.remove(drop)


# In[10]:

ensemble.transform(binary_target=True, scalers=[dmc.transformation.normalize_raw_features]*10, 
                   ignore_features=[unimp_fts] * 10)


# In[11]:

ensemble.classify([SVM] * len(ensemble.splits))



# In[ ]:




# In[ ]:



