import pandas as pd

raw_data = pd.read_csv('../data/blowouts_new_cleaned.csv')

raw_data.drop_duplicates(['Date', 'Operator', 'Lease/Facility Name', 'API#', 'Drill Permit #', 'Well #'], inplace=True)

raw_data.index = pd.to_datetime(raw_data.Date)

print(raw_data['2016'])
print(raw_data['2016'].shape[0])

year_rng = range(1969, 2020, 1)
print(*year_rng)

pass
