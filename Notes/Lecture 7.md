# Lecture 7: Kernel Methods

## Non-Linear Classification

### **Simple Transformation Example**
Linear classifiers fail for non-linearly separable data. To address this, **transform the input data**  $ x \in \mathbb{R}^d  $ to a higher-dimensional space $\phi(x) : \mathbb{R}^d \to \mathbb{R}^{d'}$ where $d' > d $. In the transformed space, the data becomes linearly separable.

Example transformation for $x = [x_1, x_2]$:

$$\phi(x) = \begin{bmatrix} x_1 \\ x_2 \\ x_1^2 + x_2^2 \end{bmatrix} \in \mathbb{R}^3$$

The original prediction function $f(x) = \text{sign}(w \cdot x)$ is now :

$$f(x) = \text{sign}(w \cdot \phi(x))$$

$$f(x) = \text{sign}(w_0 + w_1x_1 + w_2x_2 + w_3 (x_1^2 + x_2^2))$$

### **Kernel Trick**
Directly transforming data to higher dimensions is computationally expensive. To showcase this, let's look at the SVM dual formation of $w$, derived in [Lecture 6](https://github.com/lujain-khalil/MLR570-Final/blob/main/Notes/Lecture%206.md):

$$w = \sum_{i=1}^n \alpha_i y_i x_i$$

To make a prediction, let's substitute $w$ in $f(x)$:

$$f(x) = \text{sign}( (\sum_{i=1}^n \alpha_i y_i x_i) \cdot x)$$

$$f(x) = \text{sign}(\sum_{i=1}^n \alpha_i y_i (x_i \cdot x))$$

Computing the inner product ($x_i \cdot x$) in the original space is fine. However, in the transformed space, $\phi(x_i) \cdot \phi(x)$ becomes way too computationally expensive, especially in transformations such as RBF, where the data is mapped to infinite dimensions.

The solution? Find a kernel function $K(x, z)$, such that:

$$K(x, z) = \langle \phi(x), \phi(z) \rangle$$

Common kernels compute this efficiently without explicitly calculating $\phi(x)$. Some examples include the polynomial kernel and the RBF kernel, further discussed and derived below.

## Kernel Methods

### **1. Polynomial Kernel**
Maps data into a higher $d$-dimensional space using polynomial terms. Polynomial kernels are suitable for data with interactions between features. The mapping function $\phi : \mathbb{R}^{n} \rightarrow \mathbb{R}^{d}$ s.t. $d > n$ is defined as follows:

$$\phi(x) = \left[\frac{\sqrt{d!}}{\sqrt{j_1! j_2! \dots j_{n+1}!}} x_1^{j_1} \dots x_n^{j_n} 1^{j_{n+1}} \right]_{j_1 + j_2 + \dots + j_{n+1} = d}$$

The equation above is used to compute each term in the polynomial, where each term used a different combination of $j$'s that satsify the constraint $\sum_{i = 1}^{n+1} j_i = d$

Let's run it for a simple example. For a datapoint $x = [x_1 x_2]$ and $n = 2, d = 2$, the transformation $\phi(x)$ is computed as:

$$
\phi(x) = 
    \begin{bmatrix} 
        \frac{\sqrt{2!}}{\sqrt{0! 0! 2!}} \cdot x_1^0 \cdot x_2^0 \cdot 1^2 = 1 \\
        \frac{\sqrt{2!}}{\sqrt{1! 0! 1!}} \cdot x_1^1 \cdot x_2^0 \cdot 1^1 = \sqrt{2}x_1 \\
        \frac{\sqrt{2!}}{\sqrt{0! 1! 1!}} \cdot x_1^0 \cdot x_2^1 \cdot 1^1 = \sqrt{2}x_2 \\
        \frac{\sqrt{2!}}{\sqrt{1! 1! 0!}} \cdot x_1^1 \cdot x_2^1 \cdot 1^0 = \sqrt{2}x_1 x_2 \\
        \frac{\sqrt{2!}}{\sqrt{2! 0! 0!}} \cdot x_1^2 \cdot x_2^0 \cdot 1^0 = x_1^2 \\
        \frac{\sqrt{2!}}{\sqrt{0! 2! 0!}} \cdot x_1^0 \cdot x_2^2 \cdot 1^0 = x_2^2 \\
    \end{bmatrix}
$$

$$\phi(x) = \begin{bmatrix} 1, \, \sqrt{2}x_1, \, \sqrt{2}x_2, \, \sqrt{2}x_1x_2, \, x_1^2, \, x_2^2\end{bmatrix}$$

We can see that, if we were to take the inner product of two vectors $\langle \phi(x), \phi(z) \rangle$, the result would be equivalent to the binomial exapansion of $(x \cdot z + 1)^d$. Therefore, we define our polynomial kernel function as follows:

$$K(x, z) = (x \cdot z + 1)^d$$

### **2. Radial Basis Function (RBF) Kernel**
RBF can be defined as the polynomial mapping $\phi : \mathbb{R}^{n} \rightarrow \mathbb{R}^{\infty}$. This, obviously, is way too computationally complex. Instead, we have the following kernel function:

$$K(x, z) = \exp\left(-\gamma||x - z||^2\right)$$

where $\gamma = \frac{1}{2\sigma^2}$ controls the "roughness" of the kernel. The RBF kernel implicitly maps data to an **infinite-dimensional space**.


## Properties of Kernel Functions

1. **Additivity**: The sum of two valid kernels is also a valid kernel.
2. **Scalar Multiplication**: Multiplying a valid kernel by a positive scalar yields a valid kernel.
3. **Product of Kernels**: The product of two valid kernels is also a valid kernel.
4. **Exponentiation**: Raising a valid kernel to a positive power gives a valid kernel.



## Kernel Least Squares

Extending linear least squares to non-linear relationships using kernel methods, we define the objective as follows:

$$\min_{\beta} || y - \phi(X)\beta ||^2$$

We do the whole $\frac{\partial}{\partial \beta} = 0$ then find $\beta$ in terms of $\phi$ and $y$, and we get:

$$\beta = (\phi(X)^{\top} \phi(X))^{-1} \phi(X)^{\top} y$$

If we were to add a regularization term:

$$\beta = (\phi(X)^{\top} \phi(X) + \lambda I)^{-1} \phi(X)^{\top} y$$


## Summary

| **Method**          | **Kernel Function**                                  | **Feature Space**       |
|       -|                 --|        -|
| **Polynomial Kernel**|  $K(x, z) = (x \cdot z + 1)^d$                  | Finite-dimensional      |
| **RBF Kernel**       |  $K(x, z) = \text{exp}(-\gamma \|\|x - z\|\|^2)$  | Infinite-dimensional    |

 
## Limitations of Kernel Methods
- **Complex Patterns**: Might fail to capture intricate relationships.
- **Overfitting**: Flexible kernels can overfit noisy datasets.
- **Curse of Dimensionality**: Performance degrades in very high-dimensional spaces.
- **Parameter Tuning**: Selecting the right kernel and tuning parameters (e.g.,  $ \sigma  $ in RBF) can be challenging.

 

