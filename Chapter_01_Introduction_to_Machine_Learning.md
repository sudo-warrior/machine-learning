# Chapter 1: Introduction to Machine Learning

## 1.1 What is Machine Learning?

### 1.1.1 Formal Definition

Machine Learning (ML) is a field of artificial intelligence that gives computers the ability to learn without being explicitly programmed. According to Tom Mitchell's widely accepted definition:

> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

This definition encapsulates the essence of machine learning: a system that improves its performance on a specific task through experience or data.

### 1.1.2 Learning from Data vs. Explicit Programming

Traditional programming follows a rule-based approach where developers explicitly code instructions for every scenario:

```
IF condition THEN action
```

In contrast, machine learning systems learn patterns from data:

```python
# Traditional programming approach
def classify_email(email_content):
    if "lottery" in email_content and "winner" in email_content and "claim" in email_content:
        return "spam"
    # Many more rules...
    else:
        return "not_spam"

# Machine learning approach
# Train a model on thousands of labeled emails
model = train_classifier(labeled_emails)
# Use the model to classify new emails
prediction = model.predict(new_email)
```

The key difference is that ML systems can:
- Identify patterns that humans might miss
- Adapt to new patterns in data
- Handle complex relationships that would be impractical to code manually
- Improve over time with more data

### 1.1.3 Key Terminology

- **Features (X)**: The input variables or attributes used to make predictions
- **Labels (y)**: The output or target variable we're trying to predict
- **Models**: Mathematical representations that capture patterns in data
- **Training**: The process of learning patterns from data
- **Inference**: Using a trained model to make predictions on new data
- **Dataset**: Collection of examples (instances) used for training and evaluation
- **Parameters**: The values learned during training that define the model
- **Hyperparameters**: Configuration settings that are not learned from data

### 1.1.4 Real-world Examples

#### Spam Detection
- **Task**: Classify emails as spam or not spam
- **Features**: Words in the email, sender information, email structure
- **Model**: Text classifier (e.g., Naive Bayes, SVM, or neural network)
- **Impact**: Gmail reports over 99.9% accuracy in spam detection

#### Recommendation Systems
- **Task**: Suggest products, movies, or content to users
- **Features**: User behavior, item characteristics, contextual information
- **Models**: Collaborative filtering, content-based filtering, hybrid approaches
- **Impact**: Netflix estimates their recommendation system saves $1 billion per year through customer retention

#### Image Recognition
- **Task**: Identify objects, people, or scenes in images
- **Features**: Pixel values, extracted visual patterns
- **Models**: Convolutional Neural Networks (CNNs)
- **Impact**: Modern systems achieve >98% accuracy on ImageNet, enabling applications from self-driving cars to medical diagnostics

#### Medical Diagnosis
- **Task**: Detect diseases from medical images or patient data
- **Features**: Medical images (X-rays, MRIs), patient history, lab results
- **Models**: Deep learning models, ensemble methods
- **Impact**: AI systems now match or exceed human performance in detecting certain conditions like diabetic retinopathy and lung cancer

## 1.2 Types of Machine Learning

### 1.2.1 Supervised Learning

Supervised learning involves training models on labeled data, where each example has a known output or target value. The model learns to map inputs to outputs based on the provided examples.

#### Concept: Mapping Inputs (X) to Outputs (Y)

The goal is to learn a function f that maps input features X to output labels Y:

Y = f(X)

During training, the model adjusts its parameters to minimize the difference between its predictions and the true labels.

#### Tasks

**Classification**: Predicting discrete categories or classes
- **Binary Classification**: Two possible outcomes (e.g., spam/not spam)
- **Multi-class Classification**: More than two classes (e.g., digit recognition)
- **Multi-label Classification**: Multiple labels per instance (e.g., image tagging)

**Regression**: Predicting continuous numerical values
- **Simple Linear Regression**: One input feature, linear relationship
- **Multiple Linear Regression**: Multiple input features, linear relationship
- **Polynomial Regression**: Non-linear relationships using polynomial features
- **Other regression types**: Ridge, Lasso, Elastic Net, etc.

#### Examples

**Classification Example: Email Spam Detection**

```python
# Using scikit-learn for a simple spam classifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Sample data
emails = ["Win a free lottery", "Meeting at 2pm tomorrow", "Claim your prize now", "Project deadline reminder"]
labels = [1, 0, 1, 0]  # 1 for spam, 0 for not spam

# Create a pipeline with feature extraction and classifier
model = Pipeline([
    ('vectorizer', CountVectorizer()),  # Convert text to numerical features
    ('classifier', MultinomialNB())     # Naive Bayes classifier
])

# Train the model
model.fit(emails, labels)

# Make predictions
new_emails = ["Free prize waiting for you", "Team meeting notes"]
predictions = model.predict(new_emails)
print(f"Predictions: {predictions}")  # Expected: [1, 0]
```

**Regression Example: House Price Prediction**

```python
# Using scikit-learn for house price prediction
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample data: [house_size, num_bedrooms, house_age] and prices
X = np.array([
    [1400, 3, 10],
    [1800, 4, 15],
    [1200, 2, 5],
    [2100, 4, 8],
    [1600, 3, 12]
])
y = np.array([250000, 320000, 180000, 380000, 290000])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Predict price for a new house
new_house = np.array([[1750, 3, 7]])
predicted_price = model.predict(new_house)
print(f"Predicted price for new house: ${predicted_price[0]:.2f}")
```

### 1.2.2 Unsupervised Learning

Unsupervised learning involves training models on unlabeled data, where the goal is to discover hidden patterns, structures, or relationships within the data.

#### Concept: Finding Patterns in Unlabeled Data

Unlike supervised learning, there are no target outputs to guide the learning process. The model must identify inherent structures in the data based on internal patterns and relationships.

#### Tasks

**Clustering**: Grouping similar data points together
- **K-means**: Partitioning data into k clusters based on distance
- **Hierarchical Clustering**: Building nested clusters in a tree structure
- **DBSCAN**: Density-based clustering for arbitrary shapes

**Dimensionality Reduction**: Simplifying data while preserving important information
- **Principal Component Analysis (PCA)**: Linear transformation to uncorrelated components
- **t-SNE**: Non-linear technique for visualization
- **Autoencoders**: Neural network-based approach

**Anomaly Detection**: Finding outliers or unusual patterns
- **Isolation Forest**: Isolating anomalies through random partitioning
- **One-Class SVM**: Learning a boundary around normal data
- **Autoencoders**: Detecting points with high reconstruction error

#### Examples

**Clustering Example: Customer Segmentation**

```python
# Using K-means for customer segmentation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample customer data: [annual_income, spending_score]
X = np.array([
    [45000, 85],
    [35000, 35],
    [70000, 75],
    [30000, 40],
    [80000, 90],
    [42000, 45],
    [68000, 30],
    [55000, 60],
    [25000, 25],
    [75000, 80]
])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using the Elbow method
wcss = []  # Within-Cluster Sum of Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Apply K-means with the chosen number of clusters (let's use 4)
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='X', label='Centroids')
plt.title('Customer Segments')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.colorbar(label='Cluster')
plt.tight_layout()
# plt.show()  # Uncomment to display the plot

# Interpret the clusters
for i in range(4):
    cluster_points = X[clusters == i]
    print(f"Cluster {i}:")
    print(f"  Number of customers: {len(cluster_points)}")
    print(f"  Average income: ${np.mean(cluster_points[:, 0]):.2f}")
    print(f"  Average spending score: {np.mean(cluster_points[:, 1]):.2f}")
```

**Dimensionality Reduction Example: Visualizing High-Dimensional Data**

```python
# Using PCA and t-SNE for dimensionality reduction
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the digits dataset (8x8 images of handwritten digits)
digits = load_digits()
X, y = digits.data, digits.target

# Apply PCA to reduce to 50 dimensions first (for efficiency)
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)
print(f"Explained variance ratio with 50 PCA components: {sum(pca.explained_variance_ratio_):.2f}")

# Apply t-SNE to reduce to 2 dimensions for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# Visualize the result
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.8)
plt.colorbar(scatter, label='Digit')
plt.title('t-SNE Visualization of Handwritten Digits')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.show()  # Uncomment to display the plot
```

### 1.2.3 Reinforcement Learning

Reinforcement learning involves training agents to make sequences of decisions by interacting with an environment and receiving feedback in the form of rewards or penalties.

#### Concept: Learning through Interaction and Feedback

The agent learns to take actions that maximize cumulative rewards over time, often involving delayed feedback and trade-offs between immediate and long-term rewards.

#### Terminology

- **Agent**: The learner or decision-maker
- **Environment**: The world with which the agent interacts
- **State**: The current situation or configuration
- **Action**: What the agent can do in a given state
- **Reward**: Feedback signal indicating the desirability of an action
- **Policy**: The agent's strategy for selecting actions
- **Value Function**: Estimation of future rewards from a state
- **Q-Function**: Estimation of future rewards from a state-action pair

#### Examples

**Game Playing: AlphaGo**
- Developed by DeepMind (Google)
- Combined reinforcement learning with deep neural networks
- Learned by playing millions of games against itself
- Defeated world champion Lee Sedol in 2016, a milestone in AI

**Robotics Control Example**

```python
# Simple Q-learning example for a grid world
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap

# Define a simple grid world environment
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.grid = np.zeros((size, size))
        # Set goal position (reward +1)
        self.goal = (size-1, size-1)
        self.grid[self.goal] = 1
        # Set trap positions (reward -1)
        self.traps = [(1, 1), (2, 3), (3, 1)]
        for trap in self.traps:
            self.grid[trap] = -1
        # Current position
        self.pos = (0, 0)
        # Possible actions: up, right, down, left
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
    def reset(self):
        self.pos = (0, 0)
        return self.pos
    
    def step(self, action_idx):
        action = self.actions[action_idx]
        # Calculate new position
        new_pos = (self.pos[0] + action[0], self.pos[1] + action[1])
        # Check if valid move (within grid)
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            self.pos = new_pos
        
        # Calculate reward
        if self.pos == self.goal:
            reward = 1
            done = True
        elif self.pos in self.traps:
            reward = -1
            done = True
        else:
            reward = -0.01  # Small penalty for each step
            done = False
            
        return self.pos, reward, done
    
    def render(self, q_table=None):
        grid_copy = self.grid.copy()
        grid_copy[self.pos] = 2  # Mark current position
        
        # Create a custom colormap
        cmap = ListedColormap(['white', 'green', 'red', 'blue'])
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grid_copy, cmap=cmap, interpolation='nearest')
        plt.grid(True, color='black', linestyle='-', linewidth=1)
        plt.xticks(np.arange(-.5, self.size, 1), [])
        plt.yticks(np.arange(-.5, self.size, 1), [])
        
        # If Q-table is provided, show best actions
        if q_table is not None:
            for i in range(self.size):
                for j in range(self.size):
                    if (i, j) != self.goal and (i, j) not in self.traps:
                        best_action = np.argmax(q_table[(i, j)])
                        if best_action == 0:  # up
                            plt.arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.1, fc='k', ec='k')
                        elif best_action == 1:  # right
                            plt.arrow(j, i, 0.3, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
                        elif best_action == 2:  # down
                            plt.arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.1, fc='k', ec='k')
                        elif best_action == 3:  # left
                            plt.arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
        
        plt.title('Grid World')
        # plt.show()  # Uncomment to display the plot

# Q-learning algorithm
def q_learning(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    # Initialize Q-table with zeros
    q_table = {}
    for i in range(env.size):
        for j in range(env.size):
            q_table[(i, j)] = np.zeros(4)  # 4 actions
    
    # Training
    rewards_per_episode = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3)  # Random action
            else:
                action = np.argmax(q_table[state])  # Best action
            
            # Take action and observe next state and reward
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Q-learning update
            best_next_action = np.argmax(q_table[next_state])
            q_table[state][action] += alpha * (reward + gamma * q_table[next_state][best_next_action] - q_table[state][action])
            
            state = next_state
        
        rewards_per_episode.append(total_reward)
        
        # Decay epsilon
        epsilon = max(0.01, epsilon * 0.995)
    
    return q_table, rewards_per_episode

# Create environment and train agent
env = GridWorld(size=5)
q_table, rewards = q_learning(env, episodes=500)

# Visualize learning progress
plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
# plt.show()  # Uncomment to display the plot

# Visualize learned policy
env.render(q_table)
```

**Resource Allocation Example**
- Used in data centers to optimize server allocation
- Balances energy consumption with performance requirements
- Adapts to changing workloads and conditions
- Can reduce energy costs by 20-30% while maintaining performance

### 1.2.4 Self-Supervised Learning

Self-supervised learning is a paradigm that bridges supervised and unsupervised learning. It creates supervised tasks from unlabeled data by using parts of the data as labels.

#### Concept

The key idea is to generate labels from the data itself, allowing models to learn useful representations without human annotation.

#### Examples

- **Masked Language Modeling**: Predicting masked words in a sentence (used in BERT)
- **Contrastive Learning**: Learning to distinguish similar from dissimilar examples (used in SimCLR, CLIP)
- **Autoregressive Modeling**: Predicting the next token in a sequence (used in GPT models)

```python
# Simple example of masked language modeling concept
text = "The quick brown fox jumps over the lazy dog"
masked_text = "The quick [MASK] fox jumps over the [MASK] dog"
# The model would be trained to predict "brown" and "lazy"
```

## 1.3 History and Evolution of AI and ML

### 1.3.1 Early AI Concepts (1950s-1960s)

- **Turing Test (1950)**: Alan Turing proposed a test for machine intelligence
- **Dartmouth Conference (1956)**: Birth of AI as a field
- **Logic Theorist (1956)**: First AI program by Allen Newell and Herbert Simon
- **General Problem Solver (1957)**: Early attempt at a general-purpose AI system

### 1.3.2 The Perceptron and Early Neural Networks (1950s-1960s)

- **Perceptron (1958)**: Frank Rosenblatt developed the first neural network model
- **ADALINE (1960)**: Bernard Widrow and Ted Hoff developed Adaptive Linear Neuron
- **XOR Problem (1969)**: Minsky and Papert showed limitations of single-layer perceptrons

### 1.3.3 AI Winters and Subsequent Resurgences

- **First AI Winter (1974-1980)**: Funding cuts after unrealistic expectations
- **Expert Systems Era (1980s)**: Rule-based systems for specific domains
- **Second AI Winter (1987-1993)**: Collapse of the expert systems market
- **Statistical ML Rise (1990s)**: Focus shifted to data-driven approaches

### 1.3.4 Rise of Statistical Learning Methods (1990s-2000s)

- **Support Vector Machines (1992)**: Vapnik developed powerful classification method
- **Random Forests (2001)**: Breiman introduced ensemble of decision trees
- **AdaBoost (1997)**: Freund and Schapire developed boosting algorithm
- **Practical applications**: Spam filters, recommendation systems, fraud detection

### 1.3.5 The Deep Learning Revolution (2010s)

- **ImageNet Competition (2012)**: AlexNet demonstrated power of deep CNNs
- **Word Embeddings (2013)**: Word2Vec showed effective word representations
- **GANs (2014)**: Generative Adversarial Networks by Ian Goodfellow
- **AlphaGo (2016)**: Defeated world champion in Go
- **Transformer Architecture (2017)**: Attention mechanism revolutionized NLP

### 1.3.6 Current Trends and the Era of Large Models (2020s)

- **Large Language Models**: GPT-3/4, PaLM, Claude, Llama
- **Diffusion Models**: DALL-E, Stable Diffusion, Midjourney
- **Multimodal Models**: CLIP, Flamingo, GPT-4V
- **Foundation Models**: Pre-trained on vast data, adapted to many tasks
- **AI Agents**: Systems that can plan, reason, and use tools

## 1.4 Setting Up Your Python Environment for ML

### 1.4.1 Installing Python

The Anaconda distribution is recommended for machine learning as it includes Python and many essential libraries.

**Installation Steps:**

1. Download Anaconda from [https://www.anaconda.com/download](https://www.anaconda.com/download)
2. Follow the installation instructions for your operating system
3. Verify installation by opening Anaconda Navigator or running `conda --version` in terminal

### 1.4.2 Using Virtual Environments

Virtual environments allow you to create isolated Python environments for different projects.

**Using conda environments:**

```bash
# Create a new environment
conda create --name ml-env python=3.10

# Activate the environment
conda activate ml-env

# Install packages
conda install numpy pandas matplotlib scikit-learn

# Deactivate when done
conda deactivate
```

**Using venv (Python's built-in tool):**

```bash
# Create a new environment
python -m venv ml-env

# Activate the environment (Windows)
ml-env\Scripts\activate

# Activate the environment (macOS/Linux)
source ml-env/bin/activate

# Install packages
pip install numpy pandas matplotlib scikit-learn

# Deactivate when done
deactivate
```

### 1.4.3 Essential Libraries Installation

```bash
# Basic data science stack
pip install numpy pandas matplotlib seaborn

# Machine learning
pip install scikit-learn

# Deep learning
pip install tensorflow  # or tensorflow-gpu for GPU support
pip install torch torchvision torchaudio  # PyTorch

# Natural language processing
pip install nltk spacy transformers

# Computer vision
pip install opencv-python pillow

# Reinforcement learning
pip install gym stable-baselines3
```

**Creating a requirements.txt file:**

```
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
scikit-learn==1.3.0
tensorflow==2.13.0
torch==2.0.1
transformers==4.31.0
```

Install with: `pip install -r requirements.txt`

### 1.4.4 IDE/Editor Choices

**Jupyter Notebooks/Lab**
- Interactive development with code, text, and visualizations
- Great for experimentation and sharing results
- Installation: `pip install jupyter jupyterlab`
- Usage: `jupyter notebook` or `jupyter lab`

**Visual Studio Code**
- Free, open-source editor with excellent Python support
- Extensions for Python, Jupyter, Git integration
- Integrated terminal and debugger
- Download from [https://code.visualstudio.com/](https://code.visualstudio.com/)

**PyCharm**
- Professional IDE specifically for Python
- Community (free) and Professional editions
- Advanced debugging and refactoring tools
- Download from [https://www.jetbrains.com/pycharm/](https://www.jetbrains.com/pycharm/)

**Google Colab**
- Free cloud-based Jupyter notebooks
- Access to GPUs and TPUs
- Easy sharing and collaboration
- Access at [https://colab.research.google.com/](https://colab.research.google.com/)

### 1.4.5 First Steps: Running Simple Python Code

**Hello World with NumPy:**

```python
# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Create some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a simple plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.show()

# Basic NumPy operations
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([6, 7, 8, 9, 10])

print(f"Array 1: {array1}")
print(f"Array 2: {array2}")
print(f"Sum: {array1 + array2}")
print(f"Product: {array1 * array2}")
print(f"Mean of Array 1: {np.mean(array1)}")
print(f"Standard deviation of Array 2: {np.std(array2)}")
```

**Checking Installed Versions:**

```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import sklearn

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Matplotlib version: {mpl.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")

# Check for GPU availability (TensorFlow)
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
except ImportError:
    print("TensorFlow not installed")

# Check for GPU availability (PyTorch)
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not installed")
```

## Summary

In this chapter, we've introduced the fundamental concepts of machine learning, including:

1. The definition and core principles of machine learning
2. The distinction between traditional programming and ML approaches
3. Key terminology used throughout the field
4. Real-world applications demonstrating ML's impact
5. The main types of machine learning: supervised, unsupervised, and reinforcement learning
6. A brief history of AI and ML development
7. Setting up a Python environment for machine learning

In the next chapter, we'll explore the mathematical foundations necessary for understanding how machine learning algorithms work, including linear algebra, calculus, probability, and statistics.

## References

1. Mitchell, T. M. (1997). Machine Learning. McGraw-Hill.
2. Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
4. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
5. Chollet, F. (2021). Deep Learning with Python. Manning Publications.
