# 📌 Word Vectorizer Models in NLP

## 📖 Project Overview
This project explores and compares different word vectorization techniques used in Natural Language Processing (NLP) for text classification. The goal is to understand how different methods impact model performance, computational efficiency, and feature representation.

The project is implemented on the **SMS Spam Collection Dataset**, where messages are classified as **Spam** or **Ham (Not Spam)**.

---

## 🎯 Objectives
- To implement multiple word vectorization techniques
- To compare their performance on a classification task
- To analyze trade-offs between accuracy and computational complexity
- To understand the impact of feature representation in NLP models

---

## 🧠 Vectorization Techniques Used
1. **Bag of Words (BoW)**
2. **TF-IDF**
3. **N-grams (Bi-grams)**
4. **Word2Vec**

---

## ⚙️ Methodology
- Text preprocessing (cleaning, normalization, tokenization)
- Feature extraction using different vectorization techniques
- Model training using **Logistic Regression**
- Evaluation using standard metrics

---

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

---

## 📈 Results Summary

| Model     | Accuracy |
|----------|---------|
| BoW      | 97.75%  |
| TF-IDF   | 96.68%  |
| N-grams  | 98.02%  |
| Word2Vec | 87.98%  |

🔹 **N-grams achieved the highest accuracy**, showing the importance of contextual word patterns in spam detection.

---

## 🔍 Key Insights
- Spam detection relies heavily on **keyword patterns and phrases**
- **N-grams outperform** other methods by capturing context
- **Word2Vec underperforms** in this task as it focuses on semantic meaning
- Simpler models can outperform complex ones depending on the task

---

## ⚖️ Trade-offs
- N-grams → High accuracy, higher computational cost
- BoW → Fast, but no context
- TF-IDF → Balanced performance
- Word2Vec → Semantic understanding but less effective for this task

---

## 🧪 Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Gensim
- Matplotlib

---

## 📂 Project Structure

├── modeltrain.py
├── spam.csv
├── results.png
├── README.md
├── report.pdf


---

## 🚀 Future Work
- Implement advanced models like **BERT**
- Use deep learning models (LSTM, Transformers)
- Apply to larger and more complex datasets

---

## 👥 Contributors
- Person 1: [M CHAITHANYA VARSHITH]
- Person 2: [K SHREYAS REDDY]

---

## 📌 Conclusion
This project demonstrates that the effectiveness of word vectorization techniques depends on the nature of the task. For spam detection, **N-grams proved to be the most effective**, highlighting the importance of contextual patterns over semantic embeddings.

---

⭐ If you found this project useful, feel free to star the repository
