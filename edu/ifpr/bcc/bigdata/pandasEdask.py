import pandas as pd
import dask.dataframe as dd
from numpy.ma.extras import unique

series = pd.Series([10, 20, 30, 40])

data = {'Nome': ['Alice', 'Juca', 'Carlos'], 'Idade': [23, 30, 45]}
df = pd.DataFrame(data)
print(df)
print("\n")
print(series)

df = pd.read_csv('.\\data\\processed.hungarian.data')
print("\n")
print(df)

df_filtered = df[df.iloc[:,0] < 30]
print('\n')
print(df_filtered)

df_grouped = df.groupby('age')
print(df_grouped.describe())
#porque os dados ficaram assim?

print('\nDask')
ddf = dd.read_csv('.\\data\\processed.hungarian.data')
ddf_filtered = ddf[ddf['age'] < 30]
print(ddf_filtered)
