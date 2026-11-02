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



