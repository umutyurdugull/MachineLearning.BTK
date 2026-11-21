# The Role of Vectors in Machine Learning

## Basic Properties of Vector Spaces

In machine learning, data points are represented as vectors within a vector space.

* **2-Dimensional Space:** A vector $A = (2, 5)$ resides in a 2D space, denoted as $A \in \mathbb{R}^2$.
* **$n$-Dimensional Space:** A general vector $A = (a_1, \dots, a_n)$ has $n$ dimensions, denoted as $A \in \mathbb{R}^n$.

---

## Data and Feature Representation

A dataset is mathematically represented as a collection of input-output pairs:

$$X = \{ (x_i, y_i) \mid i = 1, 2, \dots, n \}$$

**Process Flow:**
$$\text{Training Set} \rightarrow \text{Learning Process} \rightarrow f(x) = y$$

Each input instance $x$ is mapped to a feature vector $\varphi(x)$:

$$\varphi(x) = [\varphi_1(x), \dots, \varphi_d(x)], \quad \varphi(x) \in \mathbb{R}^d$$

> **Note:** The number of features ($d$) determines the dimensionality of the vector.

### Tabular Data to Vector Example

| Id | Age | Income | Education | Duration (sec) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 25 | 40000 | Uni | 180 |
| 2 | 48 | - | - | - |
| 3 | 32 | - | - | - |
| 4 | 61 | - | - | - |
| 5 | 28 | - | - | - |

From the first row of the table, we can construct a feature vector as follows:

$$x = (\text{Id}, \text{Age}, \text{Income}, \text{Education}, \text{Duration})$$

$$x = (1, 25, 40000, \text{Uni}, 180)$$

*(Note: Categorical features like `Uni` or missing values in the table require preprocessing before mathematical operations.)*

### Text Representation (Bag of Words)

The content of a text document can be represented as a vector based on word frequencies. By building a vocabulary from the entire corpus, each sentence or document can be expressed as a vector indicating how many times each word appears. 

* **Reference:** See `BagOfWords.py` for the implementation example.

---

## Basic Operations on Vectors

Let two vectors be defined as:

$$a = (3, 4, 5), \quad b = (1, 2, 9)$$

### 1. Vector Addition

$$a + b = (3+1, 4+2, 5+9) = (4, 6, 14)$$

### 2. Scalar (Dot) Product

$$a \cdot b = (3 \cdot 1) + (4 \cdot 2) + (5 \cdot 9) = 3 + 8 + 45 = 56$$

### 3. Cross Product

For 3D vectors (e.g., taking $z = 0$ as in the example):

$$a \times b = \begin{vmatrix} i & j & k \\ 3 & 4 & 0 \\ 1 & 2 & 0 \end{vmatrix}$$

---

## Fundamental Distance Metrics

### When is a Function a Metric?

For a distance function $d(A, B)$ to be considered a valid mathematical metric, it must satisfy three conditions:

1. **Positivity (Non-negativity):** $d(A, B) \ge 0$, and $d(A, B) = 0 \iff A = B$
2. **Symmetry:** $d(A, B) = d(B, A)$
3. **Triangle Inequality:** $d(A, B) \le d(A, C) + d(C, B)$

---

### 1. Euclidean Distance

Measures the straight-line (direct) distance between two points in space.

$$d(A, B) = \sqrt{\sum_{i=1}^{k} (a_i - b_i)^2}$$

### 2. Manhattan Distance

Measures the distance between two points by only moving along horizontal and vertical axes (similar to navigating a city grid).

$$d(A, B) = \sum_{i=1}^{k} |a_i - b_i|$$

### 3. Minkowski Distance

A generalized metric form that encompasses other distance metrics depending on the parameter $q$:

$$d(A, B) = \left( \sum_{i=1}^{k} |a_i - b_i|^q \right)^{\frac{1}{q}}$$

* $q = 1 \rightarrow \text{Manhattan Distance}$
* $q = 2 \rightarrow \text{Euclidean Distance}$
* $q \rightarrow \infty \text{ (or higher orders)} \rightarrow \text{Chebyshev Distance}$

### 4. Chebyshev Distance

Defined as the maximum absolute difference between the coordinates of two vectors across any single dimension.

$$d(A, B) = \max_i (|a_i - b_i|)$$

### 5. Hamming Distance

Measures how many bits (or elements) differ between two binary vectors of equal length. It is particularly useful when the dataset contains a large number of binary/categorical features.

$$d(A, B) = \sum_{i=1}^{k} I(a_i, b_i)$$

Where the indicator function is defined as:

$$I(a_i, b_i) = \begin{cases} 1, & a_i \neq b_i \\ 0, & a_i = b_i \end{cases}$$




# Tokenization, Vectorization, and Similarity Metrics

## 1. What is Tokenization?

**Tokenization** is the process of breaking down raw text data into smaller, meaningful units called **tokens** (such as words, subwords, or characters).

### Why Do We Need Tokenization?
* **Simplifies Analysis:** Breaks complex raw text into manageable, meaningful pieces.
* **Granular Control:** Enables analysis at the word level or lower levels (character-level, subword-level).
* **Foundation for Vectorization:** Serves as the mandatory preprocessing step before converting text into numerical representations.

---

## 2. Word-Based Tokenization

In word-based tokenization, text is split into individual words and symbols based on whitespace and punctuation marks.

**Example:**
* **Raw Text:** `"Bunlar gerçek hayatta ne işimize yarayacak?"` *(What use will these be to us in real life?)*
* **Tokenized Output:** `["Bunlar", "gerçek", "hayatta", "ne", "işimize", "yarayacak", "?"]`

> **Key Concept (From Tokens to Vectors):**  
> While tokenization prepares text for analysis, machine learning algorithms cannot process raw string tokens directly. These tokens must be converted into **numerical vectors** ($\vec{v} \in \mathbb{R}^d$). This vectorization process allows algorithms to understand text data, discover patterns, and make predictions.

---

## 3. Overview of Text Vectorization Methods

Once the text is tokenized, several techniques can be used to map tokens into numerical vectors:

### 1. One-Hot Encoding
* Creates a $1$-dimensional binary vector for each unique token in the vocabulary (where vector length equals vocabulary size $V$).
* The index corresponding to the specific token is set to $1$, and all other indices are set to $0$.
* *Example:* $v_i = [0, 0, 1, 0, \dots, 0]$

### 2. Bag-of-Words (BoW)
* Represents a document as a vector of word frequencies (counts of how many times each token appears in the document).
* **Limitation:** Word order and grammatical context are completely lost.

### 3. TF-IDF (Term Frequency - Inverse Document Frequency)
* A statistical weighting method that evaluates how important a word is to a specific document relative to a collection of documents (corpus).
* Increases proportionally to the number of times a word appears in the document, but is offset by the frequency of the word across the entire corpus.

### 4. Word Embeddings
* Converts words into low-dimensional, **dense** real-valued vectors (unlike sparse One-Hot or BoW vectors where most elements are $0$).
* Capable of capturing **semantic relationships** between words (words with similar meanings have closer vector representations in space).
* **Popular Algorithms:** Word2Vec, GloVe, FastText.

---

## 4. Cosine Similarity

**Cosine Similarity** measures the similarity between two non-zero vectors in an inner product space by calculating the cosine of the angle ($\theta$) between them. Unlike Euclidean distance, it focuses on the **orientation (direction)** of the vectors rather than their magnitude, making it ideal for text frequency vectors.

### Mathematical Formulation

$$
\cos \theta = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}
$$

### Numerical Example

Consider three vectors in $\mathbb{R}^2$:
* $A = [48, 12]$
* $B = [50, 10]$
* $C = [10, 50]$

#### 1. Similarity between $A$ and $B$ (Angle $\theta_1$):
* **Dot Product:** $A \cdot B = (48 \cdot 50) + (12 \cdot 10) = 2400 + 120 = 2520$
* **Magnitudes:**
  $$\|A\| = \sqrt{48^2 + 12^2} = \sqrt{2304 + 144} = \sqrt{2448} \approx 49.477$$
  $$\|B\| = \sqrt{50^2 + 10^2} = \sqrt{2500 + 100} = \sqrt{2600} \approx 50.990$$
* **Cosine Similarity:**
  $$\cos \theta_1 = \frac{2520}{49.477 \cdot 50.990} \approx 0.998$$
  *(Vectors $A$ and $B$ point in almost the exact same direction; high similarity.)*

#### 2. Similarity between $B$ and $C$ (Angle $\theta_2$):
* **Dot Product:** $B \cdot C = (50 \cdot 10) + (10 \cdot 50) = 500 + 500 = 1000$
* **Magnitudes:** $\|B\| = \sqrt{2600}$, $\|C\| = \sqrt{10^2 + 50^2} = \sqrt{2600}$
* **Cosine Similarity:**
  $$\cos \theta_2 = \frac{1000}{\sqrt{2600} \cdot \sqrt{2600}} = \frac{1000}{2600} \approx 0.385$$
  *(Larger angle $\theta_2$ between $B$ and $C$ indicates low similarity.)*

---

## 5. Categorical Data Encoding & Deep Dive into One-Hot Encoding

### 1. Encoding Ordinal Categorical Data
When encoding **ordinal categorical data** (where categories have a natural, intrinsic order), the fundamental hierarchical relationship within the data must be preserved during numerical conversion (Label/Ordinal Encoding).

* **Dataset:** $A = \{\text{Good (İyi)}, \text{Medium (Orta)}, \text{Bad (Kötü)}\}$
* **Numerical Mapping:**
  $$\text{Good} \rightarrow 3, \quad \text{Medium} \rightarrow 2, \quad \text{Bad} \rightarrow 1$$
* Here, the mathematical inequality $3 > 2 > 1$ accurately reflects the true characteristic of the data ($\text{Good} > \text{Medium} > \text{Bad}$).

### 2. The Problem with Nominal Categorical Data
What if the data characteristic does **not** contain a natural mathematical hierarchy (**nominal categorical data**)?
* **Example 1:** $A = \{\text{The Good (İyi)}, \text{The Bad (Kötü)}, \text{The Ugly (Çirkin)}\}$
* **Example 2:** Geometric shapes $\{\text{Circle}, \text{Triangle}, \text{Square}\}$

If we naively assign integer labels:
$$\text{Circle} = 1, \quad \text{Triangle} = 2, \quad \text{Square} = 3$$

Since numbers inherently possess an ordering relation ($3 > 2 > 1$), machine learning algorithms will falsely assume an ordinal hierarchy where none exists:
$$\text{Square} > \text{Triangle} > \text{Circle}$$

### 3. Solution: One-Hot Encoding & Linear Independence
To prevent artificial ordering in nominal data, **One-Hot Encoding** represents each category as a separate binary dimension:

| Category | Circle | Triangle | Square | Vector ($\vec{v} \in \mathbb{R}^3$) |
| :--- | :---: | :---: | :---: | :---: |
| **Circle** | $1$ | $0$ | $0$ | $[1, 0, 0]$ |
| **Triangle** | $0$ | $1$ | $0$ | $[0, 1, 0]$ |
| **Square** | $0$ | $0$ | $1$ | $[0, 0, 1]$ |

#### Mathematical Properties of One-Hot Vectors:
* **Linear Independence:** Vectors generated via One-Hot Encoding are **linearly independent** (orthogonal standard basis vectors $e_1, e_2, \dots, e_n$).
* **No Linear Combination:** None of these vectors can be written in terms of the others (e.g., $c_1 [1,0,0] + c_2 [0,1,0] \neq [0,0,1]$ for any scalars $c_1, c_2$).
* **Equidistant:** The dot product between any two distinct One-Hot vectors is $0$ ($e_i \cdot e_j = 0$ for $i \neq j$), and the Euclidean distance between any pair is constant ($\sqrt{2}$), completely eliminating any false superiority or ordering among categories.





## Linear Independence Theorem

Let $V = \{v_1, v_2, \dots, v_n\}$ be a finite set of non-zero vectors in a vector space, and let $c_1, c_2, \dots, c_n$ be $n$ scalars. The vectors in $V$ are **linearly independent** if and only if the vector equation:

$$
c_1 v_1 + c_2 v_2 + \dots + c_n v_n = 0
$$

is satisfied exclusively by the trivial solution:

$$
c_1 = c_2 = \dots = c_n = 0
$$

### Example with One-Hot Encoded Vectors

Consider the three One-Hot encoded basis vectors representing categorical shapes in $\mathbb{R}^3$:

* $\text{Circle} = (1, 0, 0)$
* $\text{Triangle} = (0, 1, 0)$
* $\text{Square} = (0, 0, 1)$

To verify that these vectors are **linearly independent**, we set their linear combination equal to the zero vector:

$$
\begin{aligned}
c_1(1, 0, 0) + c_2(0, 1, 0) + c_3(0, 0, 1) &= (0, 0, 0) \\
(c_1, 0, 0) + (0, c_2, 0) + (0, 0, c_3) &= (0, 0, 0) \\
(c_1, c_2, c_3) &= (0, 0, 0)
\end{aligned}
$$

$$
\implies c_1 = 0, \quad c_2 = 0, \quad c_3 = 0
$$

Since the equation holds true only when all scalars are zero, the vectors $(1, 0, 0)$, $(0, 1, 0)$, and $(0, 0, 1)$ are proven to be **linearly independent**.



# Representation of Datasets with Matrices

## 1. Dataset Hierarchy and Structure

In machine learning, data is organized hierarchically from the entire dataset down to individual attributes:

$$\text{Dataset} \xrightarrow{\text{Observations}} \text{Samples (Rows)} \xrightarrow{\text{Features}} \text{Attributes (Columns)}$$

* **Dataset (Veri Seti):** The complete collection of data points collected for a machine learning task.
* **Observations / Samples (Gözlemler):** Each individual sample, instance, or record (represented as rows in tabular data).
* **Features / Attributes (Özellikler):** The distinct measurable properties or variables observed for each sample (represented as columns in tabular data).

---

## 2. What is a Matrix?

A **matrix** is a two-dimensional rectangular array of numbers arranged in rows and columns. 

A general matrix $A$ of size $m \times n$ ($m$ rows and $n$ columns) is denoted as:

$$A = \begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\ 
a_{21} & a_{22} & \cdots & a_{2n} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
a_{m1} & a_{m2} & \cdots & a_{mn} 
\end{bmatrix}_{m \times n}$$

* **Matrix Element ($a_{ij}$ or $a_{mn}$):** Represents the scalar value located at the $i$-th row and $j$-th column ($m$. satır, $n$. sütundaki eleman).

### Matrix as a Set of Column Vectors

A matrix can be interpreted as an ordered collection of $n$ column vectors:

$$A = \begin{bmatrix} \mathbf{v}_1 & \mathbf{v}_2 & \cdots & \mathbf{v}_n \end{bmatrix} = \begin{bmatrix} 
\begin{bmatrix} a_{11} \\ a_{21} \\ \vdots \\ a_{m1} \end{bmatrix} & 
\begin{bmatrix} a_{12} \\ a_{22} \\ \vdots \\ a_{m2} \end{bmatrix} & 
\cdots & 
\begin{bmatrix} a_{1n} \\ a_{2n} \\ \vdots \\ a_{mn} \end{bmatrix} 
\end{bmatrix}$$

* In the context of machine learning, each column vector represents a **feature** spanning across all $m$ observations.
* Conversely, each row can be viewed as a row vector representing a single **observation** containing $n$ feature values.

---

## 3. Tabular Dataset to Matrix Representation

Consider the following customer dataset:

| Customer ID | Monthly Spending (₺) | Visit Frequency (Monthly) | Product Variety (Count) | Loyalty Score (0–100) |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 2500 | 8 | 12 | 85 |
| **2** | 800 | 2 | 5 | 40 |
| **3** | 1200 | 4 | 7 | 55 |
| **4** | 3200 | 10 | 15 | 92 |
| **5** | 600 | 1 | 4 | 30 |
| **6** | 1500 | 5 | 8 | 60 |
| **7** | 2800 | 9 | 13 | 88 |
| **8** | 1000 | 3 | 6 | 50 |
| **9** | 500 | 1 | 3 | 25 |
| **10** | 2200 | 7 | 11 | 80 |

### Mathematical Formulation (Design Matrix)

When feeding this tabular data into machine learning algorithms:
1. Identifier columns (such as `Customer ID`) are dropped as they carry no generalizable predictive signal.
2. The remaining numerical features form a feature matrix (often called the **Design Matrix**) $X \in \mathbb{R}^{10 \times 4}$:

$$X = \begin{bmatrix} 
2500 & 8 & 12 & 85 \\ 
800 & 2 & 5 & 40 \\ 
1200 & 4 & 7 & 55 \\ 
3200 & 10 & 15 & 92 \\ 
600 & 1 & 4 & 30 \\ 
1500 & 5 & 8 & 60 \\ 
2800 & 9 & 13 & 88 \\ 
1000 & 3 & 6 & 50 \\ 
500 & 1 & 3 & 25 \\ 
2200 & 7 & 11 & 80 
\end{bmatrix}_{10 \times 4}$$

### Vector Decomposition

* **Row Vector (Observation $i$):**
  $$x^{(i)} \in \mathbb{R}^4 \quad \text{e.g., } x^{(1)} = [2500, 8, 12, 85]$$

* **Column Vector (Feature $j$):**
  $$x_j \in \mathbb{R}^{10} \quad \text{e.g., Feature 1 (Monthly Spending): } x_1 = \begin{bmatrix} 2500 \\ 800 \\ \vdots \\ 2200 \end{bmatrix}$$

---

## 4. Supervised Learning Formulation: Features ($X$) and Labels ($y$)

In supervised learning tasks (such as customer segmentation / classification), each observation is paired with an output label:

$$\text{Customer } (x_i) \longrightarrow \text{Segment } (y_i), \quad \text{for } i = 1, \dots, 10$$

* **Feature Vector ($x_i \in \mathbb{R}^4$):** The row vector containing all attributes for the $i$-th customer.
* **Target / Label ($y_i$):** The category/class assigned to the $i$-th customer (`Sadık`, `Riskli`, `Normal`).

### Dataset as Input-Output Pairs

The entire supervised dataset is formalized as:

$$\mathcal{D} = \{ (x_i, y_i) \mid i = 1, \dots, 10 \}$$

$$\underbrace{\begin{bmatrix} 
2500 & 8 & 12 & 85 \\ 
800 & 2 & 5 & 40 \\ 
1200 & 4 & 7 & 55 \\ 
\vdots & \vdots & \vdots & \vdots \\ 
2200 & 7 & 11 & 80 
\end{bmatrix}_{10 \times 4}}_{\text{Input / Feature Matrix } X} 
\quad \longrightarrow \quad
\underbrace{\begin{bmatrix} 
\text{Sadık} \\ 
\text{Riskli} \\ 
\text{Normal} \\ 
\vdots \\ 
\text{Sadık} 
\end{bmatrix}_{10 \times 1}}_{\text{Target / Label Vector } y}$$

> **Key Dimensions:**
> - **Number of Samples / Observations ($m$):** $10$ rows
> - **Number of Features ($n$):** $4$ columns
> - **Feature Matrix:** $X \in \mathbb{R}^{m \times n} = \mathbb{R}^{10 \times 4}$
> - **Target Vector:** $y \in \mathbb{R}^{m \times 1} = \mathbb{R}^{10 \times 1}$

---

# Matrix Operations in Machine Learning Processes

## 1. Linear Systems and Feature-Target Relationships

In machine learning workflows (e.g., linear regression), we often seek to uncover the relationship between input features and target continuous values via systems of linear equations.

### Example: Predicting House Prices

Consider a sample housing dataset with two observations:

| Record No | Number of Rooms (Count) | Size ($m^2$) | Price (Million ₺) |
| :---: | :---: | :---: | :---: |
| **1** | 2 | 180 | 3 |
| **2** | 4 | 300 | 8 |

From the table, house prices depend directly on the **number of rooms** and **house size ($m^2$)**.

### Algebraic System of Equations

Assuming a linear relationship where $x_1$ is the unit price/weight per room and $x_2$ is the unit price/weight per $m^2$:

$$
\begin{aligned}
2 x_1 + 180 x_2 &= 3 \\
4 x_1 + 300 x_2 &= 8
\end{aligned}
$$

### Matrix-Vector Representation ($A\mathbf{x} = \mathbf{b}$)

Using matrix multiplication, this system of linear equations is compactly formulated as:

$$\begin{bmatrix} 2 & 180 \\ 4 & 300 \end{bmatrix} \cdot \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 3 \\ 8 \end{bmatrix}$$

* **Feature / Coefficient Matrix ($A \in \mathbb{R}^{2 \times 2}$):**
  $$A = \begin{bmatrix} 2 & 180 \\ 4 & 300 \end{bmatrix}$$
* **Unknown Weights / Parameters ($\mathbf{x} \in \mathbb{R}^{2 \times 1}$):**
  $$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$
* **Target / Price Vector ($\mathbf{b} \in \mathbb{R}^{2 \times 1}$):**
  $$\mathbf{b} = \begin{bmatrix} 3 \\ 8 \end{bmatrix}$$

---

## 2. Image Representation via Matrices: The RGB Model

Digital images are a core data modality in computer vision and machine learning. An image is fundamentally structured as a 2D grid of pixels arranged horizontally and vertically.

### Pixel Grid and Multi-Channel Structure
* An image of dimensions $m \times n$ (height $\times$ width) contains $m \cdot n$ individual pixels.
* In color imaging, the color of each pixel is determined by its intensity across three primary channels: **Red (R)**, **Green (G)**, and **Blue (B)**.
* Consequently, an $m \times n$ color image is represented as a stack of three $m \times n$ matrices (a **3D tensor** of shape $m \times n \times 3$):
  $$\mathcal{I} = [R, G, B], \quad \text{where } R, G, B \in \mathbb{R}^{m \times n}$$
* Each pixel intensity value typically ranges from $0$ to $255$ in standard 8-bit digital representation ($0 \le p_{ij} \le 255$).

### Example: Channel Decomposition of an Image Patch

When inspecting a localized patch of an image, the values across the three color channels form three distinct matrices:

$$
\mathbf{R}_{\text{channel}} = \begin{bmatrix} 
240 & 241 & 241 \\ 
240 & 237 & 238 \\ 
239 & 240 & 240 \\ 
238 & 237 & 240 \\ 
240 & 240 & 239 \\ 
239 & 240 & 240 
\end{bmatrix}, \quad
\mathbf{G}_{\text{channel}} = \begin{bmatrix} 
207 & 199 & 196 \\ 
183 & 163 & 195 \\ 
183 & 166 & 184 \\ 
176 & 172 & 181 \\ 
184 & 167 & 176 \\ 
182 & 180 & 170 
\end{bmatrix}, \quad
\mathbf{B}_{\text{channel}} = \begin{bmatrix} 
234 & 231 & 225 \\ 
223 & 213 & 225 \\ 
219 & 211 & 195 \\ 
176 & 205 & 189 \\ 
168 & 141 & 117 \\ 
160 & 142 & 117 
\end{bmatrix}
$$

### Applications of Matrix Operations on Images
Formulating images as matrices unlocks linear algebra operations for computer vision:
1. **Transformations & Enhancement:** Adjusting brightness and contrast via scalar multiplication and addition ($\alpha \mathbf{M} + \beta$).
2. **Convolutional Filters (CNNs):** Feature extraction, edge detection, and blurring through kernel-matrix operations.
3. **Normalization:** Scaling pixel ranges from $[0, 255]$ down to $[0, 1]$ ($\frac{\mathbf{M}}{255.0}$) for numerical stability in neural networks.

---

# Eigenvalues and Eigenvectors (Özdeğer ve Özvektör)

In linear algebra and machine learning (e.g., Principal Component Analysis - PCA, dimensionality reduction, PageRank), **eigenvalues** and **eigenvectors** reveal the fundamental axes along which a linear transformation acts merely by stretching or scaling vectors without altering their direction.

## 1. Fundamental Definition and Derivation

### Conceptual and Geometric Interpretation

* **Eigenvector (Özvektör):** A vector whose **direction does not change** when a matrix transformation is applied to it; only its **scale** (length/magnitude) changes.
* **Eigenvalue (Özdeğer):** The **scaling factor** ($\lambda$) by which the eigenvector is stretched, compressed, or reversed (if negative).

$$\underbrace{A \cdot x}_{\text{Matrix Transformation}} = \underbrace{\lambda \cdot x}_{\text{Pure Scalar Scaling}}$$

### Mathematical Formulation

Let $A$ be a square matrix ($n \times n$). A non-zero vector $x \in \mathbb{R}^n$ ($x \neq 0$) is an **eigenvector** of $A$, and a scalar $\lambda \in \mathbb{R}$ is its corresponding **eigenvalue**, if:

$$A x = \lambda x$$

To solve for $\lambda$ and $x$:
1. Introduce the identity matrix $I$:
   $$A x = \lambda I x$$
2. Move all terms to one side:
   $$A x - \lambda I x = 0 \implies (A - \lambda I) x = 0$$
3. Since an eigenvector must be non-trivial ($x \neq 0$), the matrix $(A - \lambda I)$ must be singular (non-invertible), meaning its determinant must be zero:
   $$\det(A - \lambda I) = 0$$
   This is known as the **Characteristic Equation**.

---

## 2. Step-by-Step Example

Let matrix $A$ be:
$$A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$$

### Step 1: Characteristic Equation & Eigenvalues ($\lambda$)

$$\det(A - \lambda I) = \begin{vmatrix} 0 - \lambda & 1 \\ -2 & -3 - \lambda \end{vmatrix} = 0$$

Compute the determinant:
$$(-\lambda)(-3 - \lambda) - (1)(-2) = 0$$
$$\lambda(\lambda + 3) + 2 = 0$$
$$\lambda^2 + 3\lambda + 2 = 0$$

Factor the quadratic equation:
$$(\lambda + 2)(\lambda + 1) = 0$$

Eigenvalues:
$$\lambda_1 = -2, \quad \lambda_2 = -1$$

---

### Step 2: Finding Eigenvectors

#### Case 1: For $\lambda_1 = -2$

Substitute $\lambda_1 = -2$ into $A x = \lambda x$:

$$\begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = -2 \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} -2x_1 \\ -2x_2 \end{bmatrix}$$

Applying matrix-vector multiplication (Row $\times$ Column):
1. **Row 1 $\times$ Column:**
   $$(0 \cdot x_1) + (1 \cdot x_2) = -2x_1 \implies x_2 = -2x_1$$
2. **Row 2 $\times$ Column:**
   $$(-2 \cdot x_1) + (-3 \cdot x_2) = -2x_2 \implies -2x_1 - 3x_2 = -2x_2 \implies -2x_1 = x_2$$

Both equations produce the identical linear dependency:
$$x_2 = -2x_1$$

Choosing $x_1 = 1$, we get $x_2 = -2$. The corresponding eigenvector $\mathbf{v}_1$ is:
$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ -2 \end{bmatrix} \quad \left(\text{or generally } c \begin{bmatrix} 1 \\ -2 \end{bmatrix}, \forall c \neq 0\right)$$

#### Case 2: For $\lambda_2 = -1$

Substitute $\lambda_2 = -1$ into $A x = \lambda x$:

$$\begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = -1 \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} -x_1 \\ -x_2 \end{bmatrix}$$

1. **Row 1 $\times$ Column:**
   $$(0 \cdot x_1) + (1 \cdot x_2) = -x_1 \implies x_2 = -x_1$$
2. **Row 2 $\times$ Column:**
   $$-2x_1 - 3x_2 = -x_2 \implies -2x_1 = 2x_2 \implies x_2 = -x_1$$

Choosing $x_1 = 1$, we get $x_2 = -1$. The corresponding eigenvector $\mathbf{v}_2$ is:
$$\mathbf{v}_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix} \quad \left(\text{or generally } c \begin{bmatrix} 1 \\ -1 \end{bmatrix}, \forall c \neq 0\right)$$

---

### Geometric Interpretation of the Result

* When matrix $A$ acts on the space, ordinary vectors rotate and change direction.
* However, any vector pointing along $\mathbf{v}_1 = [1, -2]^T$ stays along that same line—it is only scaled by $\lambda_1 = -2$ (flipped and doubled in length).
* Similarly, any vector pointing along $\mathbf{v}_2 = [1, -1]^T$ stays on its line and is scaled by $\lambda_2 = -1$ (direction inverted, same magnitude).





