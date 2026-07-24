<div align="center">

  <h1>🎭 Sentiment Analysis Project</h1>
  <p><strong>A high-performance Natural Language Processing (NLP) framework for text classification and sentiment prediction.</strong></p>

  <!-- Badges -->
  <a href="https://github.com/Abhimanyu2314/Sentiment-Analysis-Project/stargazers">
    <img src="https://img.shields.io/github/stars/Abhimanyu2314/Sentiment-Analysis-Project?style=for-the-badge&logo=github&color=gold" alt="Stars">
  </a>
  <a href="https://github.com/Abhimanyu2314/Sentiment-Analysis-Project/network/members">
    <img src="https://img.shields.io/github/forks/Abhimanyu2314/Sentiment-Analysis-Project?style=for-the-badge&logo=github&color=blue" alt="Forks">
  </a>
  <a href="https://github.com/Abhimanyu2314/Sentiment-Analysis-Project/issues">
    <img src="https://img.shields.io/github/issues/Abhimanyu2314/Sentiment-Analysis-Project?style=for-the-badge&logo=github&color=red" alt="Issues">
  </a>
  <a href="https://github.com/Abhimanyu2314/Sentiment-Analysis-Project/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
  </a>

  <br />
  <br />

  <a href="#-overview">Overview</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-project-structure">Project Structure</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-model-benchmarks">Benchmarks</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="#-contributing">Contributing</a>

</div>

---

## 📌 Overview

**Sentiment-Analysis-Project** is an end-to-end Machine Learning and Natural Language Processing (NLP) solution designed to automatically process, analyze, and classify text data into sentiment categories (**Positive**, **Negative**, or **Neutral**).

Engineered for both experimental research and lightweight deployment, this project leverages the **TinySentiment** architecture for fast inference while maintaining high accuracy across various domain texts (reviews, social media posts, user feedback logs).

---

## ✨ Key Features

- **🧹 Advanced Text Preprocessing:** Automatic noise/HTML removal, lowercasing, stop-word filtering, and NLTK lemmatization.
- **⚡ TinySentiment Core Engine:** Fast, memory-efficient feature extraction via TF-IDF vectorization paired with probabilistic classification.
- **📈 Comprehensive Metrics:** Evaluates accuracy, precision, recall, F1-scores, and outputs full evaluation matrices.
- **💾 Model Serialization:** One-click save and load capabilities for trained pipeline checkpoints (`.joblib`).
- **📄 Built-in PDF Documentation:** Integrated technical manual located directly in the repo (`tinysentiment_expanded_manual.pdf`).

---

## 📁 Project Structure

```text
Sentiment-Analysis-Project/
│
├── smarted major/                  # Core modules, notebooks, and models
│   ├── tinysentiment.py            # Main model and preprocessing class
│   ├── main.py                     # Execution, training, and inference script
│   └── tinysentiment_model.joblib  # Saved model weights
│
├── tinysentiment_expanded_manual.pdf # Full theoretical and architectural manual
├── requirements.txt               # Python package dependencies
└── README.md                       # Repository documentation
