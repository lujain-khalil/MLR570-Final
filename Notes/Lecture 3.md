# Lecture 3: Data Preprocessing

Some other data preprocessing techniques include the following:

- **Dimensionality Reduction** reduces data size while retaining essential information.
- **Feature Selection** improves model performance and efficiency by selecting the most relevant features.
- **Feature Creation** (not discussed)

## **1. Dimensionality Reduction**
Transforms high-dimensional data into a lower-dimensional space to improve interpretability and computational efficiency. Techniques include:
- **Principal Component Analysis (PCA)**: Captures directions of maximum variance.
- **Linear Discriminant Analysis (LDA)**: Maximizes class separability.
- **t-SNE**: Focuses on preserving pairwise distances in lower dimensions.
- **UMAP**: Similar to t-SNE but faster and scalable.
- **Random Projection**: Projects data randomly to reduce dimensions.

> **_Note:_**  Only PCA is discussed in this lecture.

### **Principal Component Analysis (PCA)**

Let $X \in \mathbb{R}^{n \times d}$ be a dataset with $n$ samples and $d$ features. The goal here is to convert $X$ to a $k$-dimensional space, where $k < d$. The steps to do that are as follows:

1. **Center the Data**

Subtract the mean of each feature to center the data. Let $\mu_j$ be the mean of feature $j$ s.t. $j \in [0, d)$. Then:

$$\mu_j = \frac{1}{n} \sum_{j=0}^{j=d} x_j $$
    
Subtract the mean $\mu_j$ from every $x_j$. Let the matrix $\mu \in \mathbb{R}^{n \times d}$ s.t. every column contains $\mu_j$ for it's respective feature (better understanding of this is given in the works example document [here](https://github.com/lujain-khalil/MLR570-Final/blob/main/Notes/Practice/Lecture%203.md)). Then:

$$\begin{align*}
    \mu & = \begin{bmatrix}
           \mu_{1} & \mu_{2} & \dots & \mu_{d} \\
           \mu_{1} & \mu_{2} & \dots & \mu_{d}\\
           \vdots & \vdots & \dots & \vdots \\
           \mu_{1} & \mu_{2} & \dots & \mu_{d}
         \end{bmatrix} \in \mathbb{R}^{n \times d}
  \end{align*}$$

$$X_{\text{center}} = X - \mu$$

2. **Calculate the covariance matrix**

The covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$ is defined as follows:

$$\Sigma = \frac{1}{n-1} X_{\text{center}}^\top X_{\text{center}}$$


3. **Eigenvalue Decomposition**

The eigenvalues extracted here will basically be your principal components. Solve the eigenvalue decomposition:

$$\Sigma v_i = \lambda_i v_i$$

where:
- $v_i$ are your eigenvectors (principal components).
- $\lambda_i$: Eigenvalues (variance explained by each component).

To put it very briefly, solve for the eigenvalues $\lambda$ by solving:

$$det(\Sigma - \lambda I) = 0$$
    
Finding the actual eigenvectors $v \in \mathbb{R}^{d \times 1}$ is a very long linear algebra problem so it's not worth going over it, but it works.

3. **Project the data onto the new lower-dimensional space**

Use the eigenvectors that correspond to the top $k$ eigenvalues to form $W_k \in \mathbb{R}^{d \times k}$, s.t. every column in $W_k$ is an eigenvector (i.e. principal component):

$$\begin{align*}
    W_k & = \begin{bmatrix}
           v_{1} & v_{2} & \dots & v_{k} \\
         \end{bmatrix} \in \mathbb{R}^{d \times k}
  \end{align*}$$

Project $X_{center}$ onto the new dimensional space as follows:

$$Y = X_{\text{center}} W_k$$

where $Y \in \mathbb{R}^{n \times k}$ represents the "new" data in the lower-dimensional subspace.

#### **Singular Value Decomposition (SVD)**

The process of extracting the principal components here slightly differs, where they're going to correspond to the right singular vectors in $V$, rather than the eigenvectors in the process described above. Decomposing $X$ using SVD:

$$X = U \Sigma V^\top$$

where:
- $U \in \mathbb{R}^{n \times n}$ are the left singular vectors.
- $\Sigma \in \mathbb{R}^{n \times d}$ are the singular values (i.e. square root of the eigenvalues of $X$, $\sqrt{\lambda}$).
- $V \in \mathbb{R}^{d \times d}$ are the right singular vectors (your principal components).

#### **Optimization Objective**
We can look at PCA from two different perspectives:

1. **Maximizing Variance**: Maximizing the variance of the projected data $Y$. Deriving $Var(Y)$:

$$Var(Y) = \frac{1}{n-1} Y^{\top} Y$$

$$Var(Y) = \frac{1}{n-1} W_k^{\top} X_{\text{center}}^{\top} X_{\text{center}} W_k$$
    
$$Var(Y) = W_k^{\top} \Sigma W_k$$
    
The objective would be as follows:
    
$$\max_W \text{tr}(W_k^\top \Sigma W_k)$$

2. **Minimizing Reconstruction Error**: This is just a fancy way to say minimizing the Frobenius norm of the difference between original data and it's reconstruction. Using $W_k \in \mathbb{R}^{d \times k}$ to reconstruct $Y \in \mathbb{R}^{n \times k}$ back to it's original dimension ($X_{\text{reconstruct}} \in \mathbb{R}^{n \times d}$):
    
$$X_{\text{reconstruct}} = Y W_k^{\top}$$
    
$$X_{\text{reconstruct}} = X_{\text{center}} W_k W_k^{\top}$$
    
The objective function would be:
    
$$\min_W \|X_{\text{center}} - X_{\text{center}} W_k W_k^\top\|_F^2$$
    
Expanding this, we would get:
    
$$\min_W \text{tr}(X_{\text{center}}^{\top} X_{\text{center}}) - 2 \text{tr}(W_k^{\top} X_{\text{center}}^{\top} X_{\text{center}} W_k) + \text{tr}(W_k^{\top} X_{\text{center}}^{\top} X_{\text{center}} W_k W_k^{\top} W_k)$$

#### **Choosing Number of Components**:
- **Elbow Method**: Plot variance vs. components and find the "elbow" point.
- **Cumulative Explained Variance**: Choose components covering a target cumilative variance (e.g., 95%).
- **Cross-Validation**: Evaluate model accuracy with different components.

#### **Limitations**:
- Assumes linear relationships.
- Hard to interpret principal components.
- Maximizing variance may not always capture important features.
- Sensitive to feature scaling.

## **2. Feature Selection**
Selects the most relevant features to improve model performance and reduce dimensionality. Methods include:

### **Filter Methods**
- **Variance Threshold**: Removes features with low variance. For a feature column $X_j \in \mathbb{R}^{n}$:
     
$$\text{Var}(X_j) = \frac{1}{n} \sum_{i=1}^{n} (x_{ij} - \mu_j)^2$$

- **Pearson's Correlation Coefficient**: Measures linear relationship between a feature and the target. Higher magniture of $\rho$ indicates strong linear relationship. Sign implies direction of relationship. For a feature $X_j$ and target $y$:
    
$$\rho(X_j, y) = \frac{\text{Cov}(X_j, y)}{\sigma_{X_j} \sigma_y}$$

- **Mutual Information**: Captures both linear and non-linear dependencies. Higher MI indicated stronger relationship. For a feature $X_j$ and target $y$:
    
$$I(X_j; y) = \sum_{X, Y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}$$
    

### **Wrapper Methods**

"Wrapped" by a model:

- **Recursive Feature Elimination (RFE)**: Iteratively removes least important features based on model importance.
- **Forward Feature Selection**: Starts with no features, adds features one by one, maximizing performance at each step.

### **Embedded Methods**
Feature selection during the model training process. An example of this include **Lasso Regression (L1 Regularization)**, whcih adds an $L_1$ penalty to force some feature coefficients to zero:
    
$$\min_w \frac{1}{2n} \| Xw - y \|_2^2 + \lambda \| w \|_1$$
    
where the regularization paramter $\lambda$ indicates strength of regularization. More embedded methods are discussed in [Lecture 6](https://github.com/lujain-khalil/MLR570-Final/blob/main/Notes/Lecture%206.md).
