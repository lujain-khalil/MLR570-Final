## Lecture 3. Data Preprocessing

### Principal Component Analysis
- Both eigenvalue decomposition and SVD implicitly maximize variance and minimize reconstruction error 
- By selecting the largest $k$ eigenvalues, you're explicitly maximizing variance, since eigenvalues represent variance explained by their corresponding eigenvectors

> STILL UNSURE

### Connecting PCA to Autoencodes 
- If Autoencoders use linear activations, they work the same as PCAs, where the new features are linear combinations of other features
- If non-linear activations are used, they dont "do" the same thing anymore

## Lecture 4. Similarity measures

**Manhattan VS Euclidean**
- Euclidian is less robust to outliers because of the square, large/small outliers become much larger/smaller
- Euclidian is no longer meaningful in higher dimensions

**Jaccard and Cosine**
- Jaccard and Cosine similarity are not affected by magnitude 

## Lecture 5. Model evaluation

- **Expression power** of a model is how much information a model can capture
- **Harmonic mean** provides a more balanced understanding of the performance of a model
- **MAE** handles outlies better than **MSE** (similar logic to euclidean and manhattan)

## Lecture 6. Linear Methods

### Perceptron Algorithm
- Weight matrix in $WX + b$ is orthogonal to the decision boundary
- Perceptron algorithms takes each datapoint as an iteration
- Most algorithms use **local updates** because that's more computationally efficient:
    - Batch size = 1 $\implies$ most **local** you can get
    - Batch size = Sample size $\implies$ most **global** you can get

### SVMs
- Fails when variance is high in one class, because an unseen datapoint from the class with high variance could easily exceed the decision boundary 

> did we mention anything else?

### Linear Regression

- In $y = \beta X + \epsilon$, the error term $\epsilon$
- Linear regression is optimal when $\epsilon$ is normally distributed (recall STA 301 and the residual plot thing, when the residuals are centered around 0 with constant variance, it's a visual way to decide that this model is performing well)
- Why? Normal distribution of $\epsilon \implies$ normal distribution of $\beta X$. 
- WLS solves the problem of 

### Looking at $\beta = (X^\top X)^{-1} X^\top y$

When can you take the inverse inside? $(X^\top)^{-1} X^{-1} X^\top y$
- When $X^\top X$ is full rank (invertible)
- When $X^\top X$ is a square matrix (invertible)
    
Why $(X^\top X)^{-1}$ rather than $(X X^\top)^{-1}$?
- When $X X^\top$ is full rank (invertible)
- When $X X^\top$ is a square matrix (invertible)

Comments about rank:
- If $X \in \mathbb{R}^{m \times n}$, then $\text{rank}(X^\top X) \leq \min(m, n)$
- Ridge ($L2$) regularization ensures than we are always at full rank, since the weights are never 0 - just very close


#### WLS $\beta = (X^\top W X)^{-1} X^\top W y$

#### Ridge Regression $\beta = (X^T X + \lambda I)^{-1} X^T y$
- Perturbing $\lambda I$ increases the rank, avoiding chances of singularity (non-invertibility)

## Lecture 12: Density Based Clustering and Soft Clustering

### DBSCAN

- DBSCAN better than k-means clustering because:
    - Can classify clusters of arbitrary shapes properly
    - Recognizes noise
- When counting the points in the $\epsilon$ neighbourhood of a point $p$, do we count the point $p$ itself as well?
    - Depends on the algorithm
- It decides the number of clusters through training, unlike k-means
- DBSCAN assumes denisty is consistent throughout the dataset (homogeneous density)
- *H-DBSCAN* allows for varying density 

### Soft Clustering
- We're specifically studying Gaussian Mixture Model
- If we change the distrubution assumption from Gaussian to something else, what happens?
    - The objective changes
    - Other distributions **might** not have a closed form solution 
- Similar to what the variational autoencoder does in the latent space

> return to the autoencoder point