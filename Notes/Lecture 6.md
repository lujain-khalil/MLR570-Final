# Lecture 6: Linear Methods

Supervised learning involves learning a mapping function $f: X \to Y$, where $X$ are input features and $Y$ are output labels. A dataset is **linearly separable** if there exists a hyperplane such that, given a single datapoint $x$:
  
$$w \cdot x + b = 0$$
  
- $w$ is the weight vector (orientation).
- $b$ is the bias term (position).

Correctly classified points satisfy:
  
$$y_i (w \cdot x_i + b) > 0, \, \forall i$$

where $y \in \{-1, 1\}$. A correct classification would always result in a positive value.

## **1. Perceptron Algorithm**
Foundational binary classification algorithm. The hypothesis is as follows:

$$f(x) = \text{sign}(w \cdot x + b)$$

where $w \in \mathbb{R}^{d}$ and $b \in \mathbb{R}$. Perceptron is based on the **hinge loss** function is defined as:

$$L(w, b) = \text{max}(0, -y_i (w \cdot x_i + b))$$

Deriving $L(w, b)$ w.r.t $w$ and $b$:

$$\frac{\partial L}{\partial w} = \begin{cases} -y_i x_i & \text{if } y_i (w \cdot x_i + b) \leq 0 \\ 0 & \text{otherwise} \end{cases}$$

$$\frac{\partial L}{\partial b} = \begin{cases} -y_i & \text{if } y_i (w \cdot x_i + b) \leq 0 \\ 0 & \text{otherwise} \end{cases}$$

Using gradient descent, the update rule for both $w$ and $b$ are:

$$w \gets w - \eta \frac{\partial L}{\partial w} = w + \eta y_i x_i$$

$$b \gets b - \eta \frac{\partial L}{\partial b} = b + \eta y_i$$

where $\eta$ is the learning rate.

### **Algorithm**

Weights and biases are updated iteratively based on misclassified points. A detailed example is given in the practice notes [here](https://github.com/lujain-khalil/MLR570-Final/blob/main/Notes/Practice/Lecture%202.pdf).

1. Initialize $w$ and $b$. Could be $0$'s or random or whatever
2. Evaluate $f(x) = \text{sign}(w \cdot x_0 + b)$
3. If **correct**: No update. If **misclassified**, use above update rule
4. Move on to next sample


### **Limitations of Perceptron**
- Only converges for **linearly separable data**.
- Does not optimize for **maximum margin**, leading to poor generalization.


## **2. Support Vector Machines (SVMs)**
SVMs aim to find the **maximum margin hyperplane**, improving generalization. The goal is to make sure the data points lie at least 1 unit away from the decsiion boundary. In other words, we have the following constraint:

$$y_i (w \cdot x_i + b) \geq 1, \, \forall i$$

The distance from any point $x$ to the hyperplane/decision boundary is given as:

$$d = \frac{|w \cdot x + b|}{||w||}$$

Since we're defining the margin to be one unit away on either side of the boundary, our margin width is defined as:

$$\text{Margin Width} = 2 \times \frac{1}{||w||} = \frac{2}{||w||}$$

The goal for SVM's is to minimize margin width w.r.t $w$. In other words, the optimization problem is formulated as:

$$\min_{w} \frac{1}{2} ||w||^2$$

The $\frac{1}{2}$ factor is purely for mathematical conveience when differentiating w.r.t $w$, and the squared operation is to guarantee a global minima (convex) without losing our constraint $y_i (w \cdot x_i + b) \geq 1, \, \forall i$.

### **Dual Formulation**
Turns out the best (or only? no clue) way to solve a constrained optimization problem is to introduce a Langrange multiplier $\alpha_i \geq 0$ to the objective function like this:
  
$$L(w, b, \alpha) = \frac{1}{2} ||w||^2 - \sum_{i=1}^n \alpha_i \big(y_i (w \cdot x_i + b) - 1\big)$$

Apparently that's called the primal form, which is not very primal because now, we have 3 different variables: Minimize $w$ and $b$, but maximize $\alpha$. All to minimize $L(w, b, \alpha)$. To solve this, we're gonna change it to "dual formation", where we set $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial w}$ to 0. Let's start:

$$\frac{\partial L}{\partial w} = w - \sum_{i=1}^{n} \alpha_i y_i x_i = 0$$

$$w = \sum_{i=1}^n \alpha_i y_i x_i$$

$$\frac{\partial L}{\partial b} = - \sum_{i=1}^{n} \alpha_i y_i = 0$$

$$\sum_{i=1}^{n} \alpha_i y_i = 0$$

Now we have a new way to represent $w$ and a new constraint. Substituting both these formulations into the original $L(w, b, \alpha)$, we get:

$$\max_{\alpha} L(\alpha) = \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j (x_i \cdot x_j)$$

subject to:
  - $\sum_{i=1}^n \alpha_i y_i = 0$,
  - $\alpha_i \geq 0$.

Tada. As usual, mroe detailed hand-written walkthrough can be found [here](https://github.com/lujain-khalil/MLR570-Final/blob/main/Notes/Practice/Lecture%202.pdf).


## **3. Linear Regression**

Let $X \in \mathbb{R}^{n \times d}$ be a dataset with $n$ samples and $d$ features, $y \in \mathbb{R}^{n \times 1}$ be the target vector, and $W \in \mathbb{R}^{d \times 1}$ be the weight vector.

### **Ordinary Least Squares (OLS)**
- Models relationship between input $X$ and target $y$:

$$y = X \beta + \epsilon$$

- Coefficients $\beta$ are estimated by minimizing the residual sum of squares:

$$\min_{\beta} ||y - X \beta||^2$$

$$\min_{\beta} (y - X\beta)^{\top} (y - X\beta)$$

**Deriving the normal equation**:

1. Expand the cost function:
   
$$J(\beta) = y^{\top} y - 2\beta^{\top} X^{\top} y + \beta^{\top} X^{\top} X \beta$$

2. Take the gradient with respect to $\beta$:
   
$$\frac{\partial J(\beta)}{\partial \beta} = -2 X^{\top} y + 2 X^{\top} X \beta$$

3. Set the gradient to zero:

   $$X^{\top} X \beta = X^{\top} y$$

4. Solve for $ \beta $:
   
   $$\beta = (X^{\top} X)^{-1} X^{\top} y$$

**Limitations**:
- Assumes **linear relationship** between features and target
- Assumes **constant variance** (homoscedasticity)
- Singularity in $(X^{\top} X)^{-1}$ (non-invertible). Could be handled by perturbing it with $(X^{\top}X + \epsilon I)^{-1}$.


### **Weighted Least Squares (WLS)**
Accounts for **heteroscedasticity** (non-constant variance) by assigning smaller weights to observations with higher variance:

$$\min_{\beta}  W ||(y - X\beta) || ^2$$

where $W \in \mathbb{R}^{n \times n}$ is a diagonal matrix. Each element in the diagonal $w = \frac{1}{\sigma_i^2}$ corresponds to inverse of the variance $\sigma_i^2$ of observation $i$. Higher variance $\implies$ lower weight. 

**Deriving the normal equation**:

1. Expand the cost function:
   
$$J(\beta) = y^{\top} W y - 2\beta^{\top} X^{\top} W y + \beta^{\top} X^{\top} W X \beta$$

2. Take the gradient with respect to $\beta$:
   
$$\frac{\partial J(\beta)}{\partial \beta} = -2 X^{\top} W y + 2 X^{\top} W X \beta$$

3. Set the gradient to zero:

   $$X^{\top} W X \beta = X^{\top} W y$$

4. Solve for $ \beta $:
   
   $$\beta = (X^{\top} W X)^{-1} X^{\top} W y$$

### **Ridge Regression**
Adds an $L_2$-penalty to OLS to handle multicollinearity and instability:

$$\min_{\beta} ||y - X \beta||^2 + \lambda ||\beta||^2_2$$

where $\lambda \geq 0$ controls regularization strength.

**Deriving the normal equation**:

1. Expand the cost function:
   
$$J(\beta) = y^T y - 2\beta^T X^T y + \beta^T X^T X \beta + \lambda \beta^T \beta$$

2. Take the gradient with respect to $\beta$:
   
$$\frac{\partial J(\beta)}{\partial \beta} = -2 X^T y + 2 X^T X \beta + 2 \lambda \beta$$

3. Set the gradient to zero:

   $$X^T X \beta + \lambda \beta = X^T y$$

4. Solve for $ \beta $:
   
   $$\beta = (X^T X + \lambda I)^{-1} X^T y$$

### **Summary of normal equations**

| Method             | Normal Equation                          |
|--------------------|-------------------------------------------|
| **OLS**            | $\beta = (X^T X)^{-1} X^T y$         |
| **WLS**            | $\beta = (X^T W X)^{-1} X^T W y$     |
| **Ridge Regression**| $\beta = (X^T X + \lambda I)^{-1} X^T y$ |

### **Lasso Regression**
Selects a subset of features by driving some coefficients to zero. This is achieved through an $L_1$-penalty for **sparse solutions**:

$$\min_{\beta} ||y - X \beta||^2 + \lambda ||\beta||_1$$

### **Elastic Net**
Combines Ridge and Lasso:
$$\min_{\beta} ||y - X \beta||^2 + \lambda_1 ||\beta||^2 + \lambda_2 ||\beta||_1$$

### **Comparision of linear regression algorithms**

| **Method**        | **Penalty**           | **Feature Selection** | **Multicollinearity Handling** | **Bias-Variance Tradeoff**   |
|--------------------|-----------------------|------------------------|--------------------------------|------------------------------|
| **OLS**           | None                 | No                     | Poor                          | Low bias, high variance     |
| **Ridge**         | $L_2$             | No                     | Reduces multicollinearity     | Low bias, moderate variance |
| **Lasso**         | $L_1$             | Yes                    | Arbitrarily selects features  | High bias, low variance     |
| **Elastic Net**   | $L_1 + L_2$        | Yes                    | Better than Lasso             | Flexible                   |
