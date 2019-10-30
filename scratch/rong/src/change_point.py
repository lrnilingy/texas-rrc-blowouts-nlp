import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

raw_data = pd.read_csv('../data/blowouts_new_cleaned.csv')

raw_data.drop_duplicates(['Date', 'Operator', 'Lease/Facility Name', 'API#', 'Drill Permit #', 'Well #'], inplace=True)

raw_data.index = pd.to_datetime(raw_data.Date)

print(raw_data['2016'])
print(raw_data['2016'].shape[0])

year_rng = range(1969, 2020, 1)

i = 0
count_data = np.empty(len(year_rng))
for year in year_rng:
    count_data[i] = raw_data[str(year)].shape[0]
    i += 1

print(count_data)

plt.bar(year_rng, count_data, color="#348ABD")
plt.xlabel('Date')
plt.ylabel('Count of well control incidents')
plt.title('Did the safety records change over time?')
plt.show()

pass
