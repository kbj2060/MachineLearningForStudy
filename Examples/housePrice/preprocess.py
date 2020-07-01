import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_sample = pd.read_csv('sample_submission.csv')

del train['Id']

dataframe = train.copy()
list(dataframe.columns.values)

def check(dataframe):
    cols = list(dataframe.columns.values)
    for col in cols:


check(dataframe)
