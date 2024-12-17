# Computational Complexities of ML Models

| **Model**                          | **Computational Complexity**                           | **Notes**                                              |
|------------------------------------|-------------------------------------------------------|-------------------------------------------------------|
| **Perceptron Algorithm**           | $O(N \cdot d \cdot T)$                            | - $N$: Number of data points.<br>- $d$: Features.<br>- $T$: Iterations. |
| **Weighted Least Squares**   | $O(N \cdot d^2 + d^3)$                           | - Solves $(X^T W X)^{-1} X^T W y$.<br>- Involves matrix inversion. |
| **Ordinary Least Squares**   | $O(N \cdot d^2 + d^3)$                           | - Similar to WLS but without weighting.              |
| **Ridge Regression**               | $O(N \cdot d^2 + d^3)$                           | - Adds regularization to OLS. Matrix inversion needed. |
| **SVM with Linear Kernel**         | $O(N^2 \cdot d)$ to $O(N^3)$                 | - Depends on the solver (e.g., SMO).<br>- $N$: Data points, $d$: Features. |
| **SVM with Polynomial Kernel**     | $O(N^2 \cdot d^q + N^3)$                         | - $q$: Degree of the polynomial kernel.          |
| **SVM with RBF Kernel**            | $O(N^2 \cdot d + N^3)$                           | - RBF kernel requires computing pairwise distances.  |
| **Kernel Least Squares**           | $O(N^3)$                                         | - Involves kernel matrix inversion.                  |
| **k-NN**     | $O(N \cdot d)$ per query                         | - $N$: Data points, $d$: Features.<br>- Brute force search. |
| **Decision Trees**                 | $O(N \cdot d \cdot \log N)$                      | - Each split takes $O(N \cdot d)$.               |
| **FNNs**     | $O(L \cdot N \cdot d \cdot h)$                   | - $L$: Layers.<br>- $d$: Input size.<br>- $h$: Hidden units. |
| **CNNs**  | $O(L \cdot (H \cdot W \cdot C \cdot K^2))$       | - $L$: Layers.<br>- $H, W$: Feature map size.<br>- $C$: Channels.<br>- $K$: Filter size. |
| **RNNs**| $O(T \cdot N \cdot d^2)$                         | - $T$: Time steps.<br>- $d$: Hidden units.   |
| **GRU**     | $O(T \cdot N \cdot d^2)$                         | - Similar to RNN but with gating mechanisms.         |
| **K-Means Clustering**             | $O(N \cdot k \cdot d \cdot T)$                   | - $k$: Clusters.<br>- $T$: Iterations.       |
| **GMM: Soft Clustering**           | $O(N \cdot k \cdot d^2 \cdot T)$                 | - Includes E-step and M-step.<br>- $k$: Components. |
| **Vanilla Autoencoders**           | $O(L \cdot N \cdot d \cdot h)$                   | - $L$: Layers.<br>- $h$: Hidden layer size.  |
| **Sparse Autoencoders**            | $O(L \cdot N \cdot d \cdot h)$                   | - Similar to vanilla but with sparsity constraints.  |
| **Variational Autoencoders**| $O(L \cdot N \cdot d \cdot h + M \cdot d)$       | - Adds KL-divergence regularization.<br>- $M$: Latent space dimension. |


### Notes on Notation:
- $N$: Number of data points.
- $d$: Number of features or input size.
- $T$: Number of iterations or time steps.
- $L$: Number of layers (for neural networks).
- $k$: Number of clusters (K-means) or components (GMM).
- $h$: Number of hidden units (neural networks).
- $C$: Number of channels in a CNN.
- $K$: Kernel size in a CNN.