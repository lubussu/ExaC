# importing the required libraries
import math

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


k2k = pd.read_csv('A://Repositories/Git/ProgettoDM/dataset/final_dataset/k2-kepler_old.csv', on_bad_lines='skip')
k2k.drop(columns=['Unnamed: 0'], inplace=True)

si = k2k.drop(k2k.index[k2k['disposition'] == 0])

pl_ratdom = ((si.pl_ratdor * si.st_rad) / si.st_mass)
pl_ratdot = ((si.pl_ratdor * si.st_rad) / si.st_teff)
pl_ratdokm = ((si.pl_ratdor * si.st_rad) / si.sy_kepmag)

si.insert(9, 'pl_ratdom', pl_ratdom, allow_duplicates=True)
si.insert(9, 'pl_ratdot', pl_ratdot, allow_duplicates=True)
si.insert(9, 'pl_ratdokm', pl_ratdokm, allow_duplicates=True)

si.drop(columns=['pl_name', 'disposition'], inplace=True)

stat = hopkins_statistic(si)

print(stat)

si.drop(columns=['pl_orbper', 'dec', 'pl_trandep', 'pl_trandur', 'st_teff', 'st_rad', 'sy_kepmag'],
        inplace=True)

print(si.columns)

stat = hopkins_statistic(si)

sidf = si

print(stat)

scaler = StandardScaler()
si = scaler.fit_transform(si)

kmeans_kwargs = {
    "init": "random",
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

plt.style.use("fivethirtyeight")
plt.plot(range(2, 8), silhouette_coefficients)
plt.xticks(range(2, 8))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

# Find epsilon for DBSCAN
neighbors = NearestNeighbors(n_neighbors=len(sidf.columns) * 2)
neighbors_fit = neighbors.fit(sidf)
distances, indices = neighbors_fit.kneighbors(sidf)

distances = np.sort(distances, axis=0)
distances = distances[:, 1]

plt.xlim(2800, 3300)
plt.ylim(0, 20)
plt.plot(distances)
plt.show()

# Instantiate k-means
kmeans = KMeans(n_clusters=3)

# Fit the algorithms to the features
kmeans.fit(si)

# Compute the silhouette scores for each algorithm
kmeans_silhouette = silhouette_score(
    si, kmeans.labels_
).round(2)

print(kmeans.labels_)

print(kmeans_silhouette)

label = kmeans.fit_predict(sidf)

print(label)

# filter rows of original data
filtered_label0 = sidf[label == 0]

filtered_label1 = sidf[label == 1]

filtered_label2 = sidf[label == 2]

# Plotting the results
plt.scatter(filtered_label0['pl_ratdor'], filtered_label0['pl_ratdom'], color='red')
plt.scatter(filtered_label1['pl_ratdor'], filtered_label1['pl_ratdom'], color='green')
plt.scatter(filtered_label2['pl_ratdor'], filtered_label2['pl_ratdom'], color='blue')
plt.show()

dbscan = DBSCAN(eps=8, min_samples=len(sidf.columns) * 2)

labeldb = dbscan.fit_predict(sidf)

print(labeldb)

# filter rows of original data
filtered_label0 = sidf[labeldb == 0]

filtered_label1 = sidf[labeldb == 1]

filtered_label2 = sidf[labeldb == -1]

# Plotting the results
plt.scatter(filtered_label0['pl_ratdor'], filtered_label0['pl_ratdom'], color='red')
plt.scatter(filtered_label1['pl_ratdor'], filtered_label1['pl_ratdom'], color='green')
plt.scatter(filtered_label2['pl_ratdor'], filtered_label2['pl_ratdom'], color='blue')
plt.show()
