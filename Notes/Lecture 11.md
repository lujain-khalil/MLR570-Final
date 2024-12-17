# Lecture 11: K-Means Clustering

**Unsupervised Learning**
- **Definition**: A machine learning branch used to identify structures among unlabeled data.
- **Applications**: 
    - News clustering (e.g., Google News)
    - Customer segmentation
    - Anomaly detection
    - Image quantization

## Clustering
Grouping similar data points together based on a similarity measure. The distance and cost are calculated as:

$$dist(x_i, z_j) = ||x_i - z_j||^2_2$$

$$cost(C_1, \dots, C_k, z_1, \dots, z_k) = \sum_{j=1}^k \sum_{i \in C_j} ||x_i - z_j||^2_2 $$

The objective is: 

$$\min_z \sum_{j=1}^k \sum_{i \in C_j} ||x_i - z_j||^2_2 $$

**Hard Clustering**: Each data point belongs to one exclusive cluster.

$$ C_i \cap C_j = \emptyset \, (i \neq j) $$

**Soft Clustering**: Data points can belong to multiple clusters with probabilities.


## K-Means Clustering
**Goal**: Partition data into $k$ clusters, minimizing intra-cluster distance to centroids.

**Steps**:
1. Specify $k$ (number of clusters).
2. Randomly select initial centroids $z_1, \dots, z_k$.
3. Assign each data point $x_i$ to the nearest centroid $z_j$:

$$ \argmin_{j=1,\dots,k} ||x_i - z_j||^2_2 $$

4.  Update centroids:

$$ z_j = \frac{1}{|C_j|} \sum_{i \in C_j} x_i $$

5. Repeat until convergence 

**Key Points**:
- Uses squared Euclidean distance for simplicity.
- Converges to a **local minimum**.


## Limitations of K-Means
1. **Initialization**:
    - Random initialization can result in different results.
    - **Solution**: K-Means++ initialization ensures centroids are well-separated.

2. **Spherical Clusters**:
    - K-Means assumes spherical clusters; fails on non-linear or anisotropic data.
    - **Example**: Concentric circles or moon-shaped datasets.

3. **Distance Metric**:
    - K-Means uses only squared Euclidean distance.
    - **Solution**: Use **K-Medoids** clustering, which supports other distance metrics.


## K-Means++ Initialization
Ensures robust centroid selection:
1. Choose the first centroid randomly.
2. Select the next centroid probabilistically based on squared distances.
3. Repeat until all $k$ centroids are chosen.

## K-Medoids Clustering
A generalization of K-Means that uses other distance metrics.  **Medoid**: The most central data point in a cluster.

1. Specify $k$ (clusters).
2. Initialize medoids randomly.
3. Iteratively update medoids to minimize total distance:

$$ \text{cost}(z_1, \dots, z_k) = \sum_{j=1}^k \sum_{i \in C_j} \text{dist}(x_i, z_j) $$


## Kernel K-Means
Applies a kernel function $\phi(x)$ to map data to a higher-dimensional space. Can handle non-spherical clusters effectively.

1. Transform data: $\phi(x)$.
2. Compute distances and centroids in the transformed space.
3. Assign clusters and update centroids iteratively.


The squared distance between a point $\phi(x_i)$ in the transformed space and the cluster center $z_j$ is given by:

$$||\phi(x_i) - z_j||_2^2 = \phi(x_i)^T \phi(x_i) - 2 \phi(x_i)^T z_j + z_j^T z_j$$


Reformulating with Cluster Mean $z_j$:

$$z_j = \frac{\sum_{t \in C_j} \phi(x_t)}{|C_j|}$$

Substitute into the distance equation:

$$\|\phi(x_i) - z_j\|_2^2 = \phi(x_i)^T \phi(x_i) - 2 \phi(x_i)^T \frac{\sum_{t \in C_j} \phi(x_t)}{|C_j|} + \frac{\sum_{t \in C_j} \phi(x_t)^T \phi(x_t)}{|C_j|^2}$$

Final Simplified Form:

$$\|\phi(x_i) - z_j\|_2^2 = \phi(x_i)^T \phi(x_i) - 2 \frac{\sum_{t \in C_j} \phi(x_i)^T \phi(x_t)}{|C_j|} + \frac{\sum_{t \in C_j} \phi(x_t)^T \phi(x_t)}{|C_j|^2}$$

### Kernel Trick:
The dot products $\phi(x_i)^T \phi(x_t)$ can be replaced with the kernel function $K(x_i, x_t)$:

$$K(x_i, x_t) = \phi(x_i)^T \phi(x_t)$$

Thus, the above equations can be expressed purely in terms of the kernel $K$, avoiding the need to explicitly compute $\phi(x)$.

- Polynomial: $K(x, z) = (x \cdot z + 1)^d$
- RBF: $K(x, z) = \exp\left(-\gamma||x - z||^2\right)$


## Summary Table: K-Means vs. K-Medoids

| **Feature**              | **K-Means**                  | **K-Medoids**              |
|--------------------------|-----------------------------|---------------------------|
| Distance Metric          | Squared Euclidean Distance   | Any distance metric       |
| Cluster Center           | Mean of points (centroid)    | Most central point        |
| Robustness to Outliers   | Low                          | High                      |
| Computational Cost       | Lower                        | Higher                    |
