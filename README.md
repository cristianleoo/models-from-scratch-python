<!-- <p align="center">
  <img src="your-logo-url" alt="Logo" width="80" height="80">
</p> -->

<h3 align="center">Machine Learning Models from Scratch in Python</h3>

<p align="center">
  A comprehensive repository dedicated to recreating popular machine learning models from scratch in Python, designed to facilitate deep understanding through practical examples.
  <br />
  <a href="https://github.com/cristianleoo/models-from-scratch-python"><strong>Explore the Documentation »</strong></a>
  <br />
  <br />
  <a href="https://github.com/cristianleoo/models-from-scratch-python">View Demo</a>
  ·
  <a href="https://github.com/cristianleoo/models-from-scratch-python/issues">Report Bug</a>
  ·
  <a href="https://github.com/cristianleoo/models-from-scratch-python/issues">Request Feature</a>
</p>

<p align="center">
  <img src="https://img.shields.io/github/contributors/cristianleoo/models-from-scratch-python" alt="GitHub contributors">
  <img src="https://img.shields.io/github/stars/cristianleoo/models-from-scratch-python?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/cristianleoo/models-from-scratch-python?style=social" alt="GitHub forks">
  <img src="https://img.shields.io/github/issues/cristianleoo/models-from-scratch-python" alt="GitHub issues">
  <img src="https://img.shields.io/github/license/cristianleoo/models-from-scratch-python" alt="License">
</p>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [About the Project](#about-the-project)
- [Articles](#articles)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Steps](#steps)

---

## About the Project

This repository is aimed at providing an educational resource for those seeking to understand the inner workings of machine learning algorithms. While these implementations may not match the efficiency or completeness of mature libraries like `scikit-learn`, the focus here is on simplicity and clarity, allowing users to see the mechanics behind each algorithm.

Each subfolder contains a `demo.ipynb` notebook that demonstrates a practical application of the corresponding algorithm. This includes loading a dataset, performing basic exploratory data analysis (EDA), and fitting the model to the data.

In addition to the code, a series of articles have been published that provide detailed explanations of each model. These articles cover the mathematical foundations, use cases, assumptions, advantages, and limitations of each algorithm, as well as a thorough breakdown of the corresponding Python implementation. For a deeper understanding, I highly recommend referring to these articles.

---

## Articles

Here is a collection of articles detailing the theory and implementation of each model in this repository:

### 1. Classical Supervised Learning
Foundational algorithms for classification and regression.

| Model | Description | Article |
| :--- | :--- | :--- |
| **KNN** | K-Nearest Neighbors algorithm for classification and regression. | [Read](https://medium.com/towards-data-science/the-math-behind-knn-3d34050efb71) |
| **Naive Bayes** | Probabilistic classifier based on applying Bayes' theorem. | [Read](https://medium.com/ai-in-plain-english/naive-bayes-classifier-achieving-100-accuracy-on-iris-dataset-d6df3e927096) |
| **ID3 Decision Tree** | Iterative Dichotomiser 3 algorithm for decision tree construction. | [Read](https://medium.com/@cristianleo120/master-decision-trees-and-building-them-from-scratch-in-python-af173dafb836) |
| **CART** | Classification and Regression Trees for predictive modeling. | [Read](https://medium.com/@cristianleo120/classification-and-regression-trees-cart-implementation-from-scratch-in-python-89efa31ad9a6) |
| **SVC** | Support Vector Classifier for finding optimal hyperplanes. | [Read](https://medium.com/ai-in-plain-english/support-vector-classifiers-svcs-a-comprehensive-guide-a9115a99a94f) |

### 2. Ensemble Methods
Powerful techniques combining multiple models.

| Model | Description | Article |
| :--- | :--- | :--- |
| **Random Forest** | Ensemble learning method using multiple decision trees. | [Read](https://medium.com/@cristianleo120/building-random-forest-from-scratch-in-python-16d004982788) |
| **AdaBoost** | Adaptive Boosting ensemble method focusing on difficult samples. | [Read](https://medium.com/stackademic/building-adaboost-from-scratch-in-python-18b79061fe01) |
| **XGBoost** | Optimized distributed gradient boosting library. | [Read](https://medium.com/@cristianleo120/the-math-behind-xgboost-3068c78aad9d) |

### 3. Unsupervised Learning
Discovering hidden patterns in unlabeled data.

| Model | Description | Article |
| :--- | :--- | :--- |
| **K-Means Clustering** | Unsupervised learning algorithm for partitioning data into k clusters. | [Read](https://medium.com/towards-data-science/the-math-and-code-behind-k-means-clustering-795582423666) |
| **PCA** | Dimensionality reduction using Principal Component Analysis. | [Read](https://medium.com/@cristianleo120/principal-component-analysis-pca-from-scratch-in-python-65998c681bc0) |

### 4. Optimization Algorithms
The mathematical engines behind successful model training.

| Model | Description | Article |
| :--- | :--- | :--- |
| **SGD** | Stochastic Gradient Descent optimization algorithm. | [Read](https://medium.com/@cristianleo120/stochastic-gradient-descent-math-and-python-code-35b5e66d6f79) |
| **Adam Optimizer** | Adaptive Moment Estimation optimization algorithm. | [Read](https://medium.com/towards-data-science/the-math-behind-adam-optimizer-c41407efe59b) |
| **Nadam Optimizer** | Nesterov-accelerated Adaptive Moment Estimation. | [Read](https://towardsdatascience.com/the-math-behind-nadam-optimizer-47dc1970d2cc) |

### 5. Deep Learning Foundations
The building blocks of modern AI.

| Model | Description | Article |
| :--- | :--- | :--- |
| **Neural Networks** | Foundational architecture of deep learning models. | [Read](https://medium.com/towards-data-science/the-math-behind-neural-networks-a34a51b93873) |
| **Batch Normalization** | Technique to improve training stability by normalizing layer inputs. | [Read](https://towardsdatascience.com/the-math-behind-batch-normalization-90ebbc0b1b0b) |
| **Fine-Tuning DNNs** | Strategies for adapting pre-trained deep neural networks. | [Read](https://towardsdatascience.com/the-math-behind-fine-tuning-deep-neural-networks-8138d548da69) |

### 6. Advanced Deep Learning Architectures
Specialized architectures for vision, sequence, and recursive processing.

| Model | Description | Article |
| :--- | :--- | :--- |
| **CNN** | Convolutional Neural Networks for processing grid-like data (images). | [Read](https://towardsdatascience.com/the-math-behind-convolutional-neural-networks-6aed775df076) |
| **Deep CNN (AlexNet)** | Deep Convolutional Neural Network architecture. | [Read](https://towardsdatascience.com/the-math-behind-deep-cnn-alexnet-738d858e5a2f) |
| **RNN** | Recurrent Neural Networks for sequential data processing. | [Read](https://towardsdatascience.com/the-math-behind-recurrent-neural-networks-2de4e0098ab8) |
| **LSTM** | Long Short-Term Memory networks for processing sequential data. | [Read](https://towardsdatascience.com/the-math-behind-lstm-9069b835289d) |
| **Recursive Language Models**| Novel approach to maintain infinite context via recursion. | [Read](https://medium.com/@cristianleo120/recursive-language-models-the-end-of-context-rot-649fc51885ea) |

---

## Installation

To get started with the project locally, follow these steps:

### Prerequisites

Ensure you have Python installed on your machine. You can install the necessary packages using either `pip` or `conda`.

### Steps

1. **Clone the repository:**
   ```sh
   git clone https://github.com/cristianleoo/models-from-scratch-python.git
   cd models-from-scratch-python
   ```

2. **Create and Activate a Virtual Environment:**
   It is recommended to use a virtual environment to manage dependencies.
   ```sh
   # Create virtual environment
   python3 -m venv venv
   
   # Activate it (Mac/Linux)
   source venv/bin/activate
   # Or on Windows:
   # venv\Scripts\activate
   ```

3. **Install required packages:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Register the Jupyter Kernel:**
   To ensure your notebooks use the correct environment, register it as a kernel:
   ```sh
   python -m ipykernel install --user --name=models-from-scratch --display-name "Python (Models From Scratch)"
   ```
   Now, when you open a notebook, select **Python (Models From Scratch)** as your kernel.

3. **(Optional) Run Tests:**
   Validate the setup by running the unit tests:
   ```sh
   python -m unittest discover
   ```

Now you're ready to explore the repository and learn how these machine learning models work under the hood!

---

Feel free to reach out via GitHub issues for any bugs, feature requests, or suggestions.

---

This version maintains the structure but improves the language to make it more polished and professional while emphasizing the educational nature of the project. Let me know if you need further adjustments!