import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./dataset/allexoplanets.csv', low_memory=False)
print(df.columns)

