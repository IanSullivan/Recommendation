from sklearn.cluster import KMeans
from RandomData import get_random_data

data = get_random_data()
kmeans = KMeans(n_clusters=10)
kmeans.fit(data)   #data is of shape [1000,]
#learn the labels and the means
labels = kmeans.predict(data)  #labels of shape [1000,] with values 0<= i <= 9
centroids = kmeans.cluster_centers_  #means of shape [10,]