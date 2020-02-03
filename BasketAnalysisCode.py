# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 08:51:19 2019

@author: daniel.lopez
"""

# for basic operations
import numpy as np
import pandas as pd

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# for defining path
import os
os.chdir(PATH)

# for market basket analysis
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# reading the dataset


df = pd.read_csv(LINK_TO_CODE',
                 sep = '|',dtype=str)
# let's check the shape of the dataset
data.shape

# Visualise the data

from wordcloud import WordCloud

prod_list = list(df['pa'])

plt.rcParams['figure.figsize'] = (15, 15)
wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(str(prod_list))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Most Popular Items',fontsize = 20)
plt.show()

# looking at the frequency of most popular items 

plt.rcParams['figure.figsize'] = (18, 7)
color = plt.cm.copper(np.linspace(0, 1, 40))
df['department'].value_counts().head(40).plot.bar(color = color)
plt.title('frequency of most popular items', fontsize = 20)
plt.xticks(rotation = 90 )
plt.grid()
plt.show()

df['DEPARTMENT'] = 'DEPARTMENT'
df = df.truncate(before = -1, after = 15)


import networkx as nx

graph = nx.from_pandas_edgelist(df,source='department',target='DEPARTMENT',edge_attr=True)

plt.rcParams['figure.figsize'] = (20, 20)
pos = nx.spring_layout(graph)
nx.draw_networkx_nodes(graph, pos, node_size = 1200, node_color = 'lightgreen')
nx.draw_networkx_edges(graph, pos, width = 6, alpha = 0.6, edge_color = 'black')
nx.draw_networkx_labels(graph, pos, font_size = 20, font_family = 'sans-serif')
plt.axis('off')
plt.grid()
plt.title('Top 15 First Choices', fontsize = 40)
plt.show()

# making each customers shopping items an identical list

# Creating the basket 

basket = (df.groupby(['ticket_id','department'])['department'].count().unstack().reset_index().fillna(0).set_index('ticket_id','date'))

# Remove 0 and negative numbers

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

"""
------------------------------------------------------------------------------------------------------------------------------------
'antecedent support' computes the proportion of transactions that contain the antecedent A
'consequent support' computes the support for the itemset of the consequent C. 
'support' metric then computes the support of the combined itemset A ∪ C --
note that 'support' depends on 'antecedent support'and 'consequent support' via min('antecedent support', 'consequent support').
--------------------------------------------------------------------------------------------------------------------------------
The confidence of a rule A->C is the probability of seeing the consequent in a transaction given that it also contains the antecedent. 
Note that the metric is not symmetric or directed; for instance, the confidence for A->C is different than the confidence for C->A. 
The confidence is 1 (maximal) for a rule A->C if the consequent and antecedent always occur together.
----------------------------------------------------------------------------------------------------------------------------------
The lift metric is commonly used to measure how much more often the antecedent and consequent of a rule A->C occur together than we 
would expect if they were statistically independent. If A and C are independent, the Lift score will be exactly 1.
-----------------------------------------------------------------------------------------------------------------------------------
Leverage computes the difference between the observed frequency of A and C appearing together and the frequency that would be 
expected if A and C were independent. An leverage value of 0 indicates independence.
------------------------------------------------------------------------------------------------------------------------------------
A high conviction value means that the consequent is highly depending on the antecedent. For instance, in the case of a perfect
confidence score, the denominator becomes 0 (due to 1 - 1) for which the conviction score is defined as 'inf'. Similar to lift,
if items are independent, the conviction is 1.
-------------------------------------------------------------------------------------------------------------------------------------
"""

# Filter rules
rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]
