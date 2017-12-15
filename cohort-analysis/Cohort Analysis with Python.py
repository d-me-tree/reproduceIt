
# coding: utf-8

# Article source: http://www.gregreda.com/2015/08/23/cohort-analysis-with-python/

# A **cohort** is a group of users who share something in common, be it their sign-up date, first purchase month, birth date, acquisition channel, etc.
# 
# **Cohort analysis** is the method by which these groups are tracked over time, helping you spot trends, understand repeat behaviors (purchases, engagement, amount spent, etc.), and monitor your customer and revenue retention.

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

pd.set_option('max_columns', 50)
mpl.rcParams['lines.linewidth'] = 2

get_ipython().magic(u'matplotlib inline')


# In[4]:

df = pd.read_excel('data/cohort_analysis/chapter-12-relay-foods.xlsx', sheetname='Purchase Data - Full Study')
df.head()


# In[6]:

df.info()


# ### 1. Create a period column based on the OrderDate

# In[14]:

df['OrderPeriod'] = df.OrderDate.map( lambda x: x.strftime('%Y-%m') )
df.head()


# ### 2. Determine the user's cohort group (based on their first order)

# In[15]:

df.set_index('UserId', inplace=True)

df['CohortGroup'] = df.groupby(level=0)['OrderDate'].min().apply( lambda x: x.strftime('%Y-%m') )
df.reset_index(inplace=True)
df.head()


# ### 3. Rollup data by CohortGroup & OrderPeriod

# In[23]:

grouped = df.groupby(['CohortGroup', 'OrderPeriod'])

# Count the unique users, orders and total revenue per Group + Period
cohorts = grouped.agg( {'UserId': pd.Series.nunique,
                        'OrderId': pd.Series.nunique,
                        'TotalCharges': np.sum} )

# Make the column names more meaningful
cohorts.rename( columns={'UserId': 'TotalUsers',
                         'OrderId': 'TotalOrders'}, inplace=True )
cohorts.head()


# ### 4. Label the CohortPeriod for each CohortGroup

# In[24]:

def cohort_period(df):
    """
    Creates a 'CohortPeriod' column, which is the Nth period based on the user's first purchase.
    
    Example
    -------
    Say you want to get the 3rd month for every user:
        df.sort(['UserId', 'OrderTime'], inplace=True)
        df = df.groupby('UserId').apply(cohort_period)
        df[df.CohortPeriod == 3]
    """
    df['CohortPeriod'] = np.arange(len(df)) + 1
    return df

cohorts = cohorts.groupby(level=0).apply(cohort_period)
cohorts.head()


# ### 5. Make sure we did all that right

# In[25]:

x = df[(df.CohortGroup == '2009-01') & (df.OrderPeriod == '2009-01')]
y = cohorts.ix[('2009-01', '2009-01')]

assert x['UserId'].nunique() == y['TotalUsers']
assert x['TotalCharges'].sum().round(2) == y['TotalCharges'].round(2)
assert x['OrderId'].nunique() == y['TotalOrders']

x = df[(df.CohortGroup == '2009-01') & (df.OrderPeriod == '2009-09')]
y = cohorts.ix[('2009-01', '2009-09')]

assert(x['UserId'].nunique() == y['TotalUsers'])
assert(x['TotalCharges'].sum().round(2) == y['TotalCharges'].round(2))
assert(x['OrderId'].nunique() == y['TotalOrders'])

x = df[(df.CohortGroup == '2009-05') & (df.OrderPeriod == '2009-09')]
y = cohorts.ix[('2009-05', '2009-09')]

assert(x['UserId'].nunique() == y['TotalUsers'])
assert(x['TotalCharges'].sum().round(2) == y['TotalCharges'].round(2))
assert(x['OrderId'].nunique() == y['TotalOrders'])


# ## User Retention by Cohort Group

# We want to look at the percentage change of each `CohortGroup` over time - not the absolute change.
# 
# To do this, we'll first need to create a pandas **Series** containing each CohortGroup and its size.

# In[26]:

# Reindex the DataFrame
cohorts.reset_index(inplace=True)
cohorts.set_index(['CohortGroup', 'CohortPeriod'], inplace=True)

# Create a Series holding the total size of each CohortGroup
cohort_group_size = cohorts['TotalUsers'].groupby(level=0).first()
cohort_group_size.head()


# Now, we'll need to divide the `TotalUsers` values in `cohorts` by `cohort_group_size`. Since DataFrame operations are performed based on the indices of the objects, we'll use **`unstack`** on our `cohorts` DataFrame to create a matrix each column represents a CohortGroup and each row is the CohortPeriod corresponding to that group.
# 
# To illustrate what `unstack` does, recall the first five `TotalUsers` values:

# In[31]:

cohorts['TotalUsers'].head()


# And here's what they look like when we `unstack` the CohortGroup level from the index:

# In[32]:

cohorts['TotalUsers'].unstack(0).head()


# Now, we can utilize **`broadcasting`** to divide each column by the corresponding `cohort_group_size`.
# 
# The resulting DataFrame, `user_retention`, contains the percentage of users from the cohort purchasing within the given period. For instance, 38.4% of users in the 2009-03 purchased again in month 3 (which would be May 2009).

# In[33]:

user_retention = cohorts['TotalUsers'].unstack(0).divide(cohort_group_size, axis=1)
user_retention.head(10)


# In[36]:

user_retention[['2009-06', '2009-07', '2009-08']].plot(figsize=(10, 5))
plt.title('Cohorts: User Retention')
plt.xticks(np.arange(1, 12.1, 1))
plt.xlim(1, 12)
plt.ylabel('% of Cohort Purchasing');


# In[37]:

import seaborn as sns
sns.set(style='white')

plt.figure(figsize=(12, 8))
plt.title('Cohorts: User Retention')
sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt='.0%');


# Unsurprisingly, we can see from the above chart that fewer users tend to purchase as time goes on.
# 
# However, we can also see that the 2009-01 cohort is the strongest, which enables us to ask targeted questions about this cohort compared to others -- what other attributes (besides first purchase month) do these users share which might be causing them to stick around? How were the majority of these users acquired? Was there a specific marketing campaign that brought them in? Did they take advantage of a promotion at sign-up? The answers to these questions would inform future marketing and product efforts.

# In[ ]:



