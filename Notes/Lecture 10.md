# Lecture 10: Convolutional Neural Networks and Recurrent Neural Networks

## Convolutional Neural Networks

Input images are represented as pixel intensity values ranging from $[0, 255]$. For an RGB image, it is represented as a **matrix** of shape $H \times W \times 3$ (e.g., $1080 \times 1080 \times 3$). CNNs Exploit spatial structures by learning local patches and hierarchically build up features. A basic CNN would look like this:

![CNN architecture](https://github.com/lujain-khalil/MLR570-Final/blob/main/Notes/Figures/CNN.png)

Some key advantages include:

1. **Parameter Sharing**: Reduces the number of parameters.
2. **Local Connectivity**: Learns spatial hierarchies.
3. **Translation Invariance**: Recognizes features anywhere in the input.

### Feature Extraction with Convolution

A **filter** (or kernel) is applied to local patches. Multiple filters extract different features. **Parameter sharing** is when filters are reused across the input. A visual of this can be seen in the [Lecture 10 Practice](https://github.com/lujain-khalil/MLR570-Final/blob/main/Notes/Practice/Lecture%2010.pdf) document.

For an $N \times N$ input, $F \times F$ kernel, $P$-padded rows/columns (if you add one 'border' of 0's, you're adding 1 row and 1 column to the image, hence $P = 1$), and stride = 1 (how much you're 'shifitng' the kernel across the input), the output size of a convolution is formulated as:

$$\text{Output size} = \frac{(N + 2P - F)}{\text{stride}} + 1$$  

**Example:** Input = $5 \times 5$, Filter = $3 \times 3$, Padding = $1$, Stride = $1$:

$$\text{Output Size} = \frac{(5 + 2(1) - 3)}{1} + 1 = 5 \implies 5 \times 5$$


### Pooling Layer

Pooling reduces spatial dimensions (downsampling) and provides spatial invariance. The two examples are:

| **Pooling Type**     | **Method**                       |
|-----------------------|----------------------------------|
| **Max Pooling**       | Retains the maximum value in a patch. |
| **Average Pooling**   | Computes the average of the values.   |

**Example**: For a $2 \times 2$ kernel with stride $2$, the output of a max pooling layer is:

$$
\begin{bmatrix}
1 & 1 & 4 & 2 \\
5 & 6 & 7 & 8 \\
3 & 2 & 0 & 1 \\
1 & 2 & 3 & 4
\end{bmatrix}
\quad \to \quad
\begin{bmatrix}
6 & 8 \\
3 & 4
\end{bmatrix}
$$

### Padding
- **Same Convolution**: Pads all inputs so that the features are the same size and the input throughout the network.
- **Valid-only Convolution**: Standard convolution concept but with strictly no padding in the network.

### CNN Architectures

| **Architecture** | **Year** | **Key Features**                                     | **Depth**      | **FC Layers**      | **Parameters**         |
|------------------|----------|------------------------------------------------------|----------------|------------------|------------------------|
| **LeNet-5**      | 1998     | - First successful CNN.<br>- Replaced manual feature extraction.<br>- Two conv and pooling layers. | 5 layers       | :heavy_check_mark:            | ~60,000                |
| **AlexNet**      | 2012     | - First use of ReLU.<br>- First breakthrough w/ ImageNet<br>- Dropout for regularization.<br>- Data augmentation. | 8 layers       | :heavy_check_mark:         | ~60 million            |
| **VGGNet**       | 2014     | - Smaller convolutions.<br>- Deeper network.<br> | 16–19 layers   | :heavy_check_mark:         | ~138 million           |
| **GoogleNet**    | 2014     | - Introduced **Inception module**.<br>- Reduced parameters.<br>- Much deeper than AlexNet<br>- Computationally efficient. | 22 layers      | :x:         | ~5 million             |
| **ResNet**       | 2015     | - Introduced **skip connections** (residual blocks).<br>- Much much deep networks.<br>- Solves vanishing/exploding gradient problem. | 50–152 layers  | :x:         | ~25 million (ResNet-50) |

### Inception module

Used in **GoogleNet**, the inception module is **computationally efficient** due to it's parallel CNN structure. It be explained by the following example (using Same Convolutions for the sake of simplicity):

- **Input features** of size $28 \times 28 \times 196$ (i.e. $196$ channels)
- **Parallel convolutions** of size $28 \times 28 \times C_i$, where $C_i$ is the size of subset of channels used in branch $i$. A pooling layer would take all the channels ($28 \times 28 \times 196$)
- **Filter Concatentation**: $28 \times 28 \times \sum_{i=0}^B C_i$, where $B$ is the number of branches. This final layer just concatenates the outputs of all parallel convolution branches

![Inception module](https://github.com/lujain-khalil/MLR570-Final/blob/main/Notes/Figures/inception-module.png)

### Skip Connections

Used in **ResNet**, skip connections allow deeper networks to perform better during training. Here's a visual example that uses ReLU as the activation function, where $f(z) = \max(0, z)$:

![Skip Connections](https://github.com/lujain-khalil/MLR570-Final/blob/main/Notes/Figures/skip-connections.png)


$$z_1 = W_1 \cdot a_0 + b_1$$

$$a_1 = f(z_1)$$

$$z_2 = W_2 \cdot a_1 + b_2$$


$$a_2 = f(z_2)$$

$$\implies a_2 = W_2 \cdot a_1 + b_2$$

If we introcude a skip connection:

$$a_2 = f(z_2 + a_0)$$

$$\implies a_2 = W_2 \cdot a_1 + b_2 + a_0$$

