
import pandas as pd
import numpy as np
import statsmodels as sm
import preprocess
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
plt.interactive(False)

pre = preprocess.preprocess_data()
df = pre.get_data()

sns.plt.figure()
sns.pairplot(data=df, hue="sales")
sns.plt.show(block=True)