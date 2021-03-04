import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv('data/train.pkl', index_col=0)
df_test = pd.read_csv('data/test.pkl', index_col=0)