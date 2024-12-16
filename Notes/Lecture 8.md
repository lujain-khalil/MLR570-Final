# Lecture 8: k-Nearest Neighbors (k-NNs) and Decision Trees


## k-Nearest Neighbors (k-NNs)

- **Non-parametric** and **instance-based** algorithm.
- Predictions rely on distances to stored training examples.
- Works for **classification** and **regression** tasks.

### Distance Metrics:
| Metric              | Formula                                   |
|---------------------|-------------------------------------------|
| Euclidean Distance  | $ d(x_q, x_i) = \sqrt{\sum_{j=1}^n (x_{qj} - x_{ij})^2} $ |
| Manhattan Distance  | $ d(x_q, x_i) = \sum_{j=1}^n |x_{qj} - x_{ij}| $         |

---

### k-NN for Classification
Class label assigned by the **majority vote** of $ k $-nearest neighbors:

$$\hat{y}_q = \text{mode}(\{y_1, y_2, \dots, y_k\})$$

#### Tie-Breaking Strategies:
1. **Random selection**: Pick randomly among tied classes.
2. **Preference for closest neighbor**: Use the closest neighbor's class.
3. **Weighted voting**: Assign weights inversely proportional to distances.


### k-NN for Regression

Predicted value is the mean of $k$-nearest neighbors:

$$\hat{y}_q = \frac{1}{k} \sum_{i=1}^{k} y_i$$

To improve performance, predicted value could also be the weighted mean of $k$-nearest neighbors:

$$\hat{y}_q = \frac{\sum_{i=1}^k w_i y_i}{\sum_{i=1}^k w_i}, \quad \text{where } w_i = \frac{1}{d(x_q, x_i)}$$

The denominator $\sum_{i=1}^k w_i$ is to scale the weights. 

### Choosing $k$

- High $k$ values could cause underfitting
- Low $k$ values could cause overfitting
- **Elbow method** for choosing $k$:
  - Plot error as a function of $ k $.
  - Choose $ k $ at the "elbow point," where error stops decreasing significantly.


### Limitations of k-NN
| Problem                        | Solution                      |
|--------------------------------|--------------------------------|
| **Curse of Dimensionality**: Distances become meaningless in higher dimensions    | Dimensionality reduction (PCA, LDA) |
| **Computational Complexity**: $O(N \cdot n)$  | Use KD-trees or ball trees.   |
| **Class Imbalance**            | Distance-weighted voting.     |


## Decision Trees

- Simple, interpretable model for **supervised classification**.
- A tree is built using:
  - **Nodes**: Represent decisions or splits on features.
  - **Edges**: Outcomes of decisions.
  - **Leaf Nodes**: Class labels.


### Splitting Criteria
1. **Information Gain (ID3)**

The goal is to maximize information gain by calculating the change in (shannon) entropy after a split.

$$\text{Entropy: } H(t) = -\sum_{i=1}^C p_i \log_2(p_i)$$

where $t$ is the node, $C$ is the number of classes, $p_i$ is the proportion of class $i$ in node $t$.

$$\text{Information Gain: } I(t) = H_{\text{root}} - \sum_{i=1}^l w_i H_i$$

where $l$ is the number of leaf nodes, $w_i$ is the proportion of samples in a leaf node (out of the number of samples in the root node $t$). 

2. **Gini Index (CART)**

Similar goal as ID#, but uses Gini Index instead of Entropy. They're both measures of impurity.

$$\text{Gini Index: } G(t) = 1 - \sum_{i=1}^C p_i^2$$

where $t$ is the node, $C$ is the number of classes, $p_i$ is the proportion of class $i$ in node $t$.

| Criterion           | Advantages                  | Disadvantages               |
|---------------------|-----------------------------|-----------------------------|
| **Information Gain**| Effective for imbalanced data | Computationally expensive   |
| **Gini Index**      | Computationally efficient   | Sensitive to class imbalance |

3. **Tsallis Entropy**

The generalization of Entropy and Gini Index, defined as:

$$\text{Tsallis Entropy: } S_q(X) = \frac{1}{q-1} (1 - \sum_{i=1}^C p_i^q)$$

where $q \in \mathbb{R}$, $t$ is the node, $C$ is the number of classes, $p_i$ is the proportion of class $i$ in node $t$. Gini Index is achieved when $q=2$, while Shannon Entropy is achieved when $q \rightarrow 1$.


### Overfitting in Decision Trees:

**Smaller, shallower trees** generalize better. Some strategies to avoid overfitting:

1. **Pre-pruning**:
    - Stop growing the tree early based on:
        - Maximum depth.
        - Minimum number of samples at a node.
        - Minimum information gain.
2. **Post-pruning**:
    - Grow the full tree and trim it afterward if information gain is low.


## 3. Comparison of k-NN and Decision Trees

| Aspect                     | k-NN                              | Decision Tree                    |
|----------------------------|-----------------------------------|----------------------------------|
| **Type**                   | Non-parametric, lazy learner     | Parametric, model-based          |
| **Training Complexity**    | None                             | Depends on tree depth            |
| **Prediction Complexity**  | $ O(N \cdot n) $               | $ O(\text{log}_2N) $            |
| **Interpretability**       | Low                              | High                             |
| **Overfitting**            | Common for small $ k $         | Prevented via pruning            |
