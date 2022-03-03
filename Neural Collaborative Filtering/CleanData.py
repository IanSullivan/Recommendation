import pandas as pd
import numpy as np
import random
import time

src = "D:\\data\\h&m images\\h-and-m-personalized-fashion-recommendations\\transactions_train.csv"
# src = "dummy.csv"
final_df_name = 'indexCustomersLabeled20.csv'
df_size = 20000
df = pd.read_csv(src)
df = df[:df_size]

# Looking for all unique values to map them to an index, embedding layer requires indexes
customerSet = set()
custormer2Idx = dict()

itemSet = set()
item2Idx = dict()
print(df_size)
print(df.columns)

[customerSet.add(i) for i in df['customer_id']]
print(len(customerSet), ' customer set size')
[itemSet.add(i) for i in df['article_id']]
print(len(itemSet), ' item set size')

for i, customer in enumerate(customerSet):
    custormer2Idx[customer] = i
for i, item in enumerate(itemSet):
    item2Idx[item] = i

n_negatives = 2
time1 = time.time()
print("real values")
# loop through the data frame with relevant columns, label of 1 indicates it is a real customer item pair
result = np.array([(custormer2Idx[x], item2Idx[y], z, 1.0) for x, y, z in zip(df['customer_id'], df['article_id'],
                                                                              df['price'])])
print(abs(time1 - time.time()))
print("fake values")
time1 = time.time()
setList = list(customerSet)
# label of 0 indicates it is a fake customer item pair ie; the customer never purchased the item in the data set
for i in range(n_negatives):
    fake_results = np.array([(custormer2Idx[random.choice(setList)], item2Idx[y], z, 0.0) for y, z in
                            zip(df['article_id'], df['price'])])

    result = np.vstack((result, fake_results))
print(abs(time1 - time.time()))
# Save to csv
df = pd.DataFrame(data=result, columns=['customer_id', 'article_id', 'price', 'label'], index=None)
df.to_csv(final_df_name)
