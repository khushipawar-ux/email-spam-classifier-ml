# email-spam-classifier-ml
**Machine Learning Project on Email Spam Classification using KNN & SVM**

### 📋 **Table of Contents**  
- [📊 Project Overview](#project-overview)  
- [📂 Dataset Description](#dataset-description)  
- [⚙️ Technologies Used](#technologies-used)  
- [🚀 How to Run the Project](#how-to-run-the-project)  
- [🔍 Key Learnings](#key-learnings)

### 📊 **Project Overview**  
This project focuses on classifying emails as **spam** or **not spam** using **K-Nearest Neighbors (KNN)** and **Support Vector Machines (SVM)**. The goal is to build a robust spam detection system that can help filter unwanted emails.  

_Key Features:_  
- **Data Preprocessing:** Tokenization, stopword removal, and TF-IDF vectorization.  
- **Modeling:** Implemented **KNN** and **SVM** classifiers.  
- **Evaluation:** Compared models based on accuracy, precision, recall, and F1-score.  


📂 **Dataset Description**  
- **Source:** [UCI Spam Dataset](https://archive.ics.uci.edu/ml/datasets/Spambase) *(or Kaggle if you're using a different dataset)*  
- **Size:** ~5,500 emails with 57 features.  
- **Features:**  
  - `word_freq_make` — Frequency of the word "make".  
  - `char_freq_$` — Frequency of the character "$".  
  - `capital_run_length_average` — Average length of uninterrupted sequences of capital letters.  
  - `is_spam` — Target variable (1 for spam, 0 for not spam).  


⚙️ **Technologies Used**  
- **Languages:** Python  
- **Libraries:**  
  - Pandas, NumPy (Data Manipulation)  
  - Scikit-learn (KNN, SVM, Model Evaluation)  
  - Matplotlib, Seaborn (Visualization)  
- **Tools:**  
  - Jupyter Notebook  

### 🚀 **How to Run the Project**

1️⃣ **Clone the repository:**  
git clone https://github.com/khushipawar-ux/email-spam-classifier-ml.git  
cd email-spam-classifier  

2️⃣ **Install dependencies:**  
pip install -r requirements.txt  

3️⃣ **Run the notebook:**  
jupyter notebook notebooks/spam_classifier.ipynb  
  
🔍 **Key Learnings**  
- **KNN** is sensitive to feature scaling — applied **StandardScaler** to normalize data.  
- **SVM** with an **RBF kernel** performed better than linear for this dataset.  
- Implemented **TF-IDF vectorization** to convert text data into numerical features.  
- Explored **hyperparameter tuning** for both models to improve accuracy.  
