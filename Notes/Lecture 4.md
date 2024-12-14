# Lecture 4: Similarity Measures

- Understanding relationships between data points is essential for many machine learning and data mining tasks. 
- Choose the similarity or distance measure based on the data type, application, and dimensionality.
- Some measures are better suited for specific tasks (e.g., Cosine for high-dimensional vectors, Manhattan for grid-based problems).

Key similarity and distance measures include:

## **1. Pearson Correlation**
- **Definition**: Measures the linear relationship between two variables.
- **Formula**:

  $$ \rho(A, B) = \frac{\text{cov}(A, B)}{\sigma_A \sigma_B} $$

  - $ \text{cov}(A, B) $: Covariance of $ A $ and $ B $.
  - $ \sigma_A, \sigma_B $: Standard deviations of $ A $ and $ B $.
- **Applications**:
  - Feature selection.
  - Linear regression.


## **2. Euclidean Distance**
- **Definition**: The straight-line distance between two points in Euclidean space.
- **Formula**:

  $$ d(A, B) = \sqrt{\sum_{i=1}^n (A_i - B_i)^2} $$

- **Applications**:
  - K-means clustering.
  - Nearest-neighbor algorithms.

## **3. Manhattan Distance**
- **Definition**: The sum of the absolute differences between the coordinates of two points.
- **Formula**:

  $$ d(A, B) = \sum_{i=1}^n |A_i - B_i| $$

- **Key Points**:
  - Measures grid-like distance (e.g., city blocks).
  - More robust to outliers and effective in high-dimensional spaces.


## **4. Cosine Similarity**
- **Definition**: Measures the cosine of the angle between two vectors, focusing on direction rather than magnitude.

- **Formula**:

  $$ \text{Cosine Similarity} = \frac{A \cdot B}{\|A\| \|B\|} $$
  
  - $ A \cdot B $: Dot product of $ A $ and $ B $.
  - $ \|A\|, \|B\| $: Magnitudes of $ A $ and $ B $.

- **Applications**:
  - Text mining.
  - Recommendation systems.

## **5. Jaccard Similarity**
- **Definition**: Measures the overlap between two sets.
- **Formula**:
  $$ \text{Jaccard Similarity} = \frac{|A \cap B|}{|A \cup B|} $$
- **Applications**:
  - Comparing binary or categorical data.
  - Document similarity in text mining.


## **Comparison of Measures**

| Measure              | Strengths                                      | Weaknesses                                | Applications                         |
|----------------------|-----------------------------------------------|------------------------------------------|--------------------------------------|
| **Pearson Correlation** | Detects linear relationships.                 | Fails for non-linear relationships.      | Feature selection, linear regression.|
| **Euclidean Distance** | Simple, intuitive for spatial problems.        | Sensitive to outliers, less effective in high dimensions. | Clustering, k-NN.                   |
| **Manhattan Distance** | Robust to outliers, good for grid-like data.   | Ignores diagonal relationships.          | Pathfinding, clustering.             |
| **Cosine Similarity**  | Effective in high-dimensional spaces.          | Ignores magnitude of vectors.            | Text mining, recommendations.        |
| **Jaccard Similarity** | Ideal for categorical or binary data.          | Loses information about magnitude.       | Text mining, clustering.             |

