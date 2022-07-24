import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from random import sample
from numpy.random import uniform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from pyclustertend import hopkins


def hopkins_statistic(X):
    X = X.values  # convert dataframe to a numpy array
    sample_size = int(X.shape[0] * 0.05)  # 0.05 (5%) based on paper by Lawson and Jures

    # a uniform random sample in the original data space
    X_uniform_random_sample = uniform(X.min(axis=0), X.max(axis=0), (sample_size, X.shape[1]))

    # a random sample of size sample_size from the original data X
    random_indices = sample(range(0, X.shape[0], 1), sample_size)
    X_sample = X[random_indices]

    # initialise unsupervised learner for implementing neighbor searches
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(X)

    # u_distances = nearest neighbour distances from uniform random sample
    u_distances, u_indices = nbrs.kneighbors(X_uniform_random_sample, n_neighbors=2)
    u_distances = u_distances[:, 0]  # distance to the first (nearest) neighbour

    # w_distances = nearest neighbour distances from a sample of points from original data X
    w_distances, w_indices = nbrs.kneighbors(X_sample, n_neighbors=2)
    # distance to the second nearest neighbour (as the first neighbour will be the point itself, with distance = 0)
    w_distances = w_distances[:, 1]

    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)

    # compute and return hopkins' statistic
    H = u_sum / (u_sum + w_sum)
    return H


df = pd.read_csv('A://Repositories/ProgettoDM/dataset/final_dataset/clustering_filled.csv', on_bad_lines='skip')
df.drop(columns=['Unnamed: 0'], inplace=True)

print(df.columns)
print('\n')

# df.loc[len(df.index)] = [1, 1, 255, 365.2]
# df.loc[len(df.index)] = [11.2, 318, 110, 4331]

data = {'pl_rade': [1, 11.2], 'pl_bmasse': [1, 318], 'pl_eqt': [255, 110], 'pl_orbper': [365.2, 4331]}
ej = pd.DataFrame(data, index=[0, 1])

si = df[['pl_rade', 'pl_bmasse', 'pl_eqt', 'pl_orbper']]

sidf = si

prt = []
for c in si.columns:
    prt.append(c + ' ')
print(str(prt))

stat = 1 - hopkins(si, 340)

print(stat)
print('\n')

for col in si.columns:
    print(col + ' ' + str(si[col].count()))

print("\n")

for col in sidf.columns:
    print(col + ' ' + str(sidf[col].count()))

print(ej)

hopkins(si, 340)

scaler = StandardScaler()
si = scaler.fit_transform(si)

kmeans_kwargs = {
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

# A list holds the SSE values for each k
sse = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(si)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(range(1, 8), sse)
plt.xticks(range(1, 8))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

kl = KneeLocator(
    range(1, 8), sse, curve="convex", direction="decreasing"
)

print(kl.elbow)

# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(si)
    score = silhouette_score(si, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.figure(figsize=(10, 10))
plt.plot(range(2, 8), silhouette_coefficients)
plt.xticks(range(2, 8))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

stdsi = pd.DataFrame(si, columns=['pl_rade', 'pl_bmasse', 'pl_eqt', 'pl_orbper'])
print(stdsi)

# Find epsilon for DBSCAN
neighbors = NearestNeighbors(n_neighbors=len(stdsi.columns) * 2)
neighbors_fit = neighbors.fit(stdsi)
distances, indices = neighbors_fit.kneighbors(stdsi)

distances = np.sort(distances, axis=0)
distances = distances[:, 1]

plt.figure(figsize=(10, 10))
plt.title("MinPts: " + str(len(stdsi.columns) * 2))
plt.xlabel("K-Distance")
plt.ylabel("Sorted object distance")
plt.xlim(300, 350)
plt.ylim(0, 1.5)
plt.plot(distances)
plt.show()

# Instantiate k-means
kmeans = KMeans(n_clusters=4)

# Fit the algorithms to the features
kmeans.fit(si)

# Compute the silhouette scores for each algorithm
kmeans_silhouette = silhouette_score(
    si, kmeans.labels_
).round(2)

print(kmeans.labels_)

print(kmeans_silhouette)

plt.figure(figsize=(10, 10))
plt.title("Exo-Planets Dendrogram")

clusters = shc.linkage(si,
                       method='ward',
                       metric="euclidean")
shc.dendrogram(Z=clusters)
# plt.axhline(y = 17, color = 'r', linestyle = '-')
plt.show()

# filter rows of original data
sidf['label'] = kmeans.labels_

filtered_label0 = sidf[sidf.label == 0]

filtered_label1 = sidf[sidf.label == 1]

filtered_label2 = sidf[sidf.label == 2]

filtered_label3 = sidf[sidf.label == 3]

filtered_label4 = sidf[sidf.label == 4]

filtered_label5 = sidf[sidf.label == 5]

sidf.drop(columns=['label'], inplace=True)

visited = []

for col1 in sidf.columns:
    visited.append(col1)
    for col2 in sidf.columns:
        if col2 not in visited:
            plt.figure(figsize=(10, 10))
            plt.scatter(filtered_label0[col1], filtered_label0[col2], color='yellow')
            plt.scatter(filtered_label1[col1], filtered_label1[col2], color='red')
            plt.scatter(filtered_label2[col1], filtered_label2[col2], color='green')
            plt.scatter(filtered_label3[col1], filtered_label3[col2], color='blue')
            plt.scatter(filtered_label4[col1], filtered_label4[col2], color='green')
            plt.scatter(filtered_label5[col1], filtered_label5[col2], color='black')
            earthx = ej.at[0, col1]
            earthy = ej.at[0, col2]
            jupiterx = ej.at[1, col1]
            jupitery = ej.at[1, col2]
            plt.plot([earthx, jupiterx], [earthy, jupitery], marker='*', ls='none', ms=20)
            plt.annotate("Earth", (earthx, earthy), fontsize=15)
            plt.annotate("Jupiter", (jupiterx, jupitery), fontsize=15)
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.xscale('log')
            plt.yscale('log')
            plt.show()

col1 = 'pl_orbper'
col2 = 'pl_eqt'

plt.figure(figsize=(10, 10))
plt.scatter(filtered_label0[col1], filtered_label0[col2], color='yellow')
plt.scatter(filtered_label1[col1], filtered_label1[col2], color='red')
plt.scatter(filtered_label2[col1], filtered_label2[col2], color='blue')
plt.scatter(filtered_label3[col1], filtered_label3[col2], color='green')
plt.scatter(filtered_label4[col1], filtered_label4[col2], color='orange')
plt.scatter(filtered_label5[col1], filtered_label5[col2], color='black')
earthx = ej.at[0, col1]
earthy = ej.at[0, col2]
jupiterx = ej.at[1, col1]
jupitery = ej.at[1, col2]
plt.plot([earthx, jupiterx], [earthy, jupitery], marker='*', ls='none', ms=20)
plt.annotate("Earth", (earthx, earthy), fontsize=15)
plt.annotate("Jupiter", (jupiterx, jupitery), fontsize=15)
plt.xlabel(col1)
plt.ylabel(col2)
plt.xscale('log')
plt.yscale('log')
plt.show()

clustering_model = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
clustering_model.fit(si)
clustering_model.labels_

# filter rows of original data
sidf['label'] = clustering_model.labels_

filtered_label0 = sidf[sidf.label == 0]

filtered_label1 = sidf[sidf.label == 1]

filtered_label2 = sidf[sidf.label == 2]

filtered_label3 = sidf[sidf.label == 3]

filtered_label4 = sidf[sidf.label == 4]

filtered_label5 = sidf[sidf.label == 5]

sidf.drop(columns=['label'], inplace=True)

visited = []

for col1 in sidf.columns:
    visited.append(col1)
    for col2 in sidf.columns:
        if col2 not in visited:
            plt.figure(figsize=(10, 10))
            plt.scatter(filtered_label0[col1], filtered_label0[col2], color='yellow')
            plt.scatter(filtered_label1[col1], filtered_label1[col2], color='red')
            plt.scatter(filtered_label2[col1], filtered_label2[col2], color='green')
            plt.scatter(filtered_label3[col1], filtered_label3[col2], color='blue')
            plt.scatter(filtered_label4[col1], filtered_label4[col2], color='orange')
            plt.scatter(filtered_label5[col1], filtered_label5[col2], color='black')
            earthx = ej.at[0, col1]
            earthy = ej.at[0, col2]
            jupiterx = ej.at[1, col1]
            jupitery = ej.at[1, col2]
            plt.plot([earthx, jupiterx], [earthy, jupitery], marker='*', ls='none', ms=20)
            plt.annotate("Earth", (earthx, earthy), fontsize=15)
            plt.annotate("Jupiter", (jupiterx, jupitery), fontsize=15)
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.xscale('log')
            plt.yscale('log')
            plt.show()

col1 = 'pl_orbper'
col2 = 'pl_eqt'

plt.figure(figsize=(10, 10))
plt.scatter(filtered_label0[col1], filtered_label0[col2], color='red')
plt.scatter(filtered_label1[col1], filtered_label1[col2], color='green')
plt.scatter(filtered_label2[col1], filtered_label2[col2], color='blue')
plt.scatter(filtered_label3[col1], filtered_label3[col2], color='yellow')
plt.scatter(filtered_label4[col1], filtered_label4[col2], color='orange')
plt.scatter(filtered_label5[col1], filtered_label5[col2], color='black')
earthx = ej.at[0, col1]
earthy = ej.at[0, col2]
jupiterx = ej.at[1, col1]
jupitery = ej.at[1, col2]
plt.plot([earthx, jupiterx], [earthy, jupitery], marker='*', ls='none', ms=20)
plt.annotate("Earth", (earthx, earthy), fontsize=15)
plt.annotate("Jupiter", (jupiterx, jupitery), fontsize=15)
plt.xlabel(col1)
plt.ylabel(col2)
plt.xscale('log')
plt.yscale('log')
plt.show()

dbscan = DBSCAN(eps=0.5, min_samples=8)
dbscan.fit(stdsi)
dbscan.labels_

# filter rows of original data
sidf['label'] = dbscan.labels_

filtered_label0 = sidf[sidf.label == 0]

filtered_label1 = sidf[sidf.label == 1]

filtered_label2 = sidf[sidf.label == 2]

filtered_label3 = sidf[sidf.label == 3]

filtered_label4 = sidf[sidf.label == 4]

filtered_label5 = sidf[sidf.label == -1]

sidf.drop(columns=['label'], inplace=True)

visited = []

print(len(filtered_label5))
print(len(filtered_label0))

for col1 in sidf.columns:
    visited.append(col1)
    for col2 in sidf.columns:
        if col2 not in visited:
            plt.figure(figsize=(10, 10))
            plt.scatter(filtered_label0[col1], filtered_label0[col2], color='green')
            plt.scatter(filtered_label1[col1], filtered_label1[col2], color='red')
            plt.scatter(filtered_label2[col1], filtered_label2[col2], color='yellow')
            plt.scatter(filtered_label3[col1], filtered_label3[col2], color='blue')
            plt.scatter(filtered_label4[col1], filtered_label4[col2], color='orange')
            plt.scatter(filtered_label5[col1], filtered_label5[col2], color='black')
            earthx = ej.at[0, col1]
            earthy = ej.at[0, col2]
            jupiterx = ej.at[1, col1]
            jupitery = ej.at[1, col2]
            plt.plot([earthx, jupiterx], [earthy, jupitery], marker='*', ls='none', ms=20)
            plt.annotate("Earth", (earthx, earthy), fontsize=15)
            plt.annotate("Jupiter", (jupiterx, jupitery), fontsize=15)
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.xscale('log')
            plt.yscale('log')
            plt.show()

clustering_model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
clustering_model.fit(si)
clustering_model.labels_

sidf['label'] = clustering_model.labels_

filtered_label0 = sidf[sidf.label == 0]

filtered_label1 = sidf[sidf.label == 1]

filtered_label2 = sidf[sidf.label == 2]

filtered_label3 = sidf[sidf.label == 3]

filtered_label4 = sidf[sidf.label == 4]

filtered_label5 = sidf[sidf.label == 5]

sidf.drop(columns=['label'], inplace=True)

visited = []

for col1 in sidf.columns:
    visited.append(col1)
    for col2 in sidf.columns:
        if col2 not in visited:
            plt.figure(figsize=(10, 10))
            plt.scatter(filtered_label0[col1], filtered_label0[col2], color='red')
            plt.scatter(filtered_label1[col1], filtered_label1[col2], color='green')
            plt.scatter(filtered_label2[col1], filtered_label2[col2], color='blue')
            plt.scatter(filtered_label3[col1], filtered_label3[col2], color='yellow')
            plt.scatter(filtered_label4[col1], filtered_label4[col2], color='orange')
            plt.scatter(filtered_label5[col1], filtered_label5[col2], color='black')
            earthx = ej.at[0, col1]
            earthy = ej.at[0, col2]
            jupiterx = ej.at[1, col1]
            jupitery = ej.at[1, col2]
            plt.plot([earthx, jupiterx], [earthy, jupitery], marker='*', ls='none', ms=20)
            plt.annotate("Earth", (earthx, earthy), fontsize=15)
            plt.annotate("Jupiter", (jupiterx, jupitery), fontsize=15)
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.xscale('log')
            plt.yscale('log')
            plt.show()