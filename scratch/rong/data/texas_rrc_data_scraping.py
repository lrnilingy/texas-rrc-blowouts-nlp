# -*- coding: utf-8 -*-
"""Texas RRC - Data scraping.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13Jk58TwwqB9gHi4qpb5mftrUPKhDDjHr
"""

import pandas as pd

url_2016_2019 = "https://www.rrc.state.tx.us/oil-gas/compliance-enforcement/blowouts-and-well-control-problems/blowouts-and-well-control-problems-2016-2019/"
url_2011_2015 = "https://www.rrc.state.tx.us/oil-gas/compliance-enforcement/blowouts-and-well-control-problems/blowouts-and-well-control-problems-11-15/"
url_2006_2010 = "https://www.rrc.state.tx.us/oil-gas/compliance-enforcement/blowouts-and-well-control-problems/blowouts-and-well-control-problems-06-10/"
url_2001_2005 = "https://www.rrc.state.tx.us/oil-gas/compliance-enforcement/blowouts-and-well-control-problems/blowouts-and-well-control-problems-01-05/"
url_1996_2000 = "https://www.rrc.state.tx.us/oil-gas/compliance-enforcement/blowouts-and-well-control-problems/blowouts-and-well-control-problems-96-00/"
url_1995_1990 = "https://www.rrc.state.tx.us/oil-gas/compliance-enforcement/blowouts-and-well-control-problems/blowouts-and-well-control-problems-90-95/"
url_1980_1989 = "https://www.rrc.state.tx.us/oil-gas/compliance-enforcement/blowouts-and-well-control-problems/blowouts-and-well-control-problems-80-89/"
url_prior1980 = "https://www.rrc.state.tx.us/oil-gas/compliance-enforcement/blowouts-and-well-control-problems/blowouts-and-well-control-problems-80/"

url = [url_2016_2019, url_2011_2015, url_2006_2010, url_2001_2005, url_1996_2000, url_1995_1990, url_1980_1989, url_prior1980]

#  Creatings lists out of the tables in the URLs
data_2016_2019 = pd.read_html(url_2016_2019)
data_2011_2015 = pd.read_html(url_2011_2015)
data_2006_2010 = pd.read_html(url_2006_2010)
data_2001_2005 = pd.read_html(url_2001_2005)
data_1996_2000 = pd.read_html(url_1996_2000)
data_1995_1990 = pd.read_html(url_1995_1990)
data_1980_1989 = pd.read_html(url_1980_1989)
data_prior1980 = pd.read_html(url_prior1980)

data = [data_2016_2019, data_2011_2015, data_2006_2010, data_2001_2005, data_1996_2000, data_1995_1990, data_1980_1989, data_prior1980]

# Checking how many elements are in the lists:
for i in data:
  print(len(i))

header1 = ["District",	"Date",	"Operator",	"Lease/Facility Name",	"Lease/ID",	"API#",	"Drill Permit #",	"Well #",	"Field Name",	"County"	,"Fire",	"H2S"	,"Injuries",	"Deaths"	,"Remarks"]
header2 = ["District",	"Date",	"Operator",	"Lease/Facility Name",	"Lease/ID",	"Drill Permit #",	"Well #",	"Field Name",	"County"	,"Fire",	"H2S"	,"Injuries",	"Deaths"	,"Remarks"]

# The only difference between the two headings is the API# column. The data previous to 2015 misses this column.

# Converting the lists to a Pandas df
df1 = pd.DataFrame(data_2016_2019[0]) 
df2 = pd.DataFrame(data_2016_2019[1]) 
df_2016_2019 = pd.concat([df1, df2])

# Adjusting the header
df_2016_2019 = df_2016_2019.iloc[1:,]
df_2016_2019.columns = header1

df_2016_2019.head()

# Converting the list to a Pandas df
df_2011_2015 = pd.DataFrame(data_2011_2015[0]) 

# Adjusting the header
df_2011_2015 = df_2011_2015.iloc[1:,]
df_2011_2015.columns = header1

df_2011_2015.head()

# Converting the list to a Pandas df
df_2006_2010 = pd.DataFrame(data_2006_2010[0]) 
 
# Adjusting the header
df_2006_2010.columns = header2


df_2006_2010.head()

# Converting the list to a Pandas df
df_2001_2005 = pd.DataFrame(data_2001_2005[0]) 

# Adjusting the header
df_2001_2005.columns = header2

df_2001_2005.head()

# Converting the list to a Pandas df
df_1995_1990 = pd.DataFrame(data_1995_1990[0]) 

# Adjusting the header
df_1995_1990.columns = header2

df_1995_1990.head()

# Converting the list to a Pandas df
df_1996_2000 = pd.DataFrame(data_1996_2000[0]) 

# Adjusting the header
df_1996_2000.columns = header2

df_1996_2000.head()

# Converting the list to a Pandas df
df_1980_1989 = pd.DataFrame(data_1980_1989[0]) 

# Adjusting the header
df_1980_1989.columns = header2

df_1980_1989.head()

# Converting the list to a Pandas df
df_prior1980  = pd.DataFrame(data_prior1980 [0]) 

# Adjusting the header
df_prior1980.columns = header2

df_prior1980.head()

# Checking number of rows in the dfs
dataframes = [df_2016_2019, df_2011_2015, df_2006_2010, df_2001_2005, df_1996_2000, df_1995_1990, df_1980_1989, df_prior1980]
for dataframe in dataframes:
  print("____________________________________________________")
  print(dataframe.count())

# Checking number of rows in the dfs
dataframes = [df_2016_2019, df_2011_2015, df_2006_2010, df_2001_2005, df_1996_2000, df_1995_1990, df_1980_1989, df_prior1980]
rows = 0
for dataframe in dataframes:
  count = dataframe['District'].count()
  rows = rows+count
  
print(rows)

#  Concatenating the dfs that have the API# row
df_1 = [df_2016_2019, df_2011_2015]
df_1 = pd.concat(df_1)
df_1.count()

#  Concatenating the dfs that DO NOT have the API# row
df_2 = [df_2006_2010, df_2001_2005, df_1996_2000, df_1995_1990, df_1980_1989, df_prior1980]
df_2 = pd.concat(df_2)
df_2.count()

df_1.head()

# Including a ghost API# row to facilitate the concatenation process
df_2.insert(5, "API#","")
df_2.head()

# Appending the two dfs
dataset = pd.concat([df_1,df_2])
dataset.head()

# Sanity check on the number of rows
dataset.describe()

from google.colab import drive
drive.mount('drive')

dataset.to_csv('blowouts_new.csv', index = False)
!cp blowouts_new.csv drive/My\ Drive/
drive.mount("drive", force_remount = True)

newdataset_cleaned = dataset.dropna()

newdataset_cleaned.drop_duplicates(subset ="Remarks", 
                     keep = False, inplace = True)

from google.colab import drive
drive.mount('drive')

dataset.to_csv('blowouts_new_cleaned.csv', index = False)
!cp blowouts_new_cleaned.csv drive/My\ Drive/
drive.mount("drive", force_remount = True)

