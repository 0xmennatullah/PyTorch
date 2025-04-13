## Funds

### 1. Traditional vs ML

#### Traditional Approach: Input + Rules = Output

- **Explainability**: Easier to understand and interpret.
- **Simplicity**: A straightforward approach.
- **Error Handling**: Less tolerant to errors.
- **Data Limitation**: Insufficient data to extract rules effectively.


#### Machine Learning (ML) / Deep Learning (DL): Input + Output = Rules

- **Complex Systems**: Ideal for scenarios with millions of rules that are hard to manually define.
- **Adaptability**: Handles environmental changes well.
- **Large Data Insights**: Extracts insights from vast datasets (e.g., differentiating 100 types of food, each with unique rules).

---

### 2. ML vs DL

#### Machine Learning (ML)

- **Data Type**: Structured data (e.g., rows and columns).
- **Applications**: Production systems.
- **Example**: XGBoost.


#### Deep Learning (DL)

- **Data Type**: Unstructured data (e.g., posts, wikis, images, audio files).
- **Example**: Neural networks.
- ![**Usage Scenarios**:]
(./img/Pasted image ٢٠٢٥٠٤١٢٢٣٠٤١٥.png)

---

### 3. Neural Networks

#### Process Flow:

1. Input: Unstructured data.
2. Numerical Encoding: Convert input into numbers.
3. Neural Network (NN): Learns representations (weights).
4. Representation Outputs: Features or classes.
5. Outputs: Presented in a human-understandable form.
6. ![[Pasted image ٢٠٢٥٠٤١٢٢٢٥٤١٤.png]]

---

### 4. Types of Learning

1. **Supervised Learning**: Data with labels for each example.
2. **Unsupervised Learning**: Learns representations and differences without labels.
3. **Transfer Learning**: Transfers patterns learned in one model to another for faster training.
4. **Reinforcement Learning**: Interaction between an agent and its environment, aiming for rewards or avoiding penalties.
5. ![[Pasted image ٢٠٢٥٠٤١٢٢٢٥٨٣٥.png]]

---

## PyTorch

### 1. GPU/TPU

#### CUDA:

- Allows ML code to run on GPUs.


#### TPU (Tensor Processing Unit):

- Not as popular as GPUs.


#### Tensor Basics:

1. Any numerical representation, primarily used in ML/DL workflows.
2. [Dan's Explanation](https://youtu.be/f5liqUk0ZTw?si=ERNNCMsPsRMAYRY6):
    - Vectors represent areas; their length can be proportional to square meters of an area and perpendicular to it.
    - Vector components are projections on axes, expressed as column vectors.
    - Scalars are rank-zero tensors (no directions required).
    - Higher-dimensional tensors:
        - Rank 1 Tensor = Vectors (one direction per axis).
        - Rank 2 Tensor = Forces/areas represented in a matrix (e.g., 2×2 matrix).
        - Rank 3 Tensor = 3D matter represented in a matrix (e.g., 3×3 matrix).

---

### 2. Workflow:

![[Pasted image ٢٠٢٥٠٤١٢٢٣٣٤١٩.png]]

---

### 3. Introduction to Tensors

1. **Definition**:
    - Tensors encode data into numerical formats using brackets (`[]`).
    - Nesting levels determine dimensions:
        - 2D → `[]`
        - 3D → `[[]]`
        - 4D → `[[[]]]`, etc.
2. Example:
    - A tensor with one matrix, three rows, and three columns → `(1,3,3)` shape.
3. Structure:
    - Outer `[]`: Contains batches.
    - Inner `[]`: Contains matrices.
    - Innermost `[]`: Contains vectors/elements.
4. Notation:
    - `tensor['batch'['matrix[vector[]]]]` → `(batch_size, matrix_size, vector_size)`.
5. Tensor Creation:
    - Use `torch.tensor` for encoding data into numbers.
6. Tensor-Like Structures:
    - Example: `tensor1 = torch.zeros_like(tensor2` creates a tensor with the same shape as tensor2 but filled with zeros.
7. Data Types:
    - Precision in computing refers to the number of digits used to represent a number.
    - Single float point precision → 32-bit; half precision → 16-bit (uses less memory and faster calculations).

