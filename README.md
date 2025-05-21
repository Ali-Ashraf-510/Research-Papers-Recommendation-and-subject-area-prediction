# Research-Papers-Recommendation-and-subject-area-prediction
# 🔍 Research Paper Recommendation & Subject Area Prediction

This project is an intelligent system that recommends similar research papers based on the title you provide. For each recommended paper, the system automatically predicts its **Subject Areas** using Natural Language Processing (NLP) and Machine Learning models.

---

## 🎥 Demo Video

Watch a short demo of the project here:  
👉 [Project Video](https://drive.google.com/file/d/1jM4GUSmWsiwGSlI6ZiLLS1Lr3zGvDEIb/view?usp=drive_link)

---

## 📚 Dataset

The project uses abstracts and metadata from arXiv papers.  
You can find the dataset here:  
📎 [Kaggle Dataset – arXiv Paper Abstracts](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts?select=arxiv_data_210930-054931.csv)

---

## 🚀 How It Works

1. The user enters the **title** of a research paper.
2. The system uses sentence embeddings to find the **top 5 most similar papers** based on their abstract.
3. It then predicts the **Subject Area(s)** of each recommended paper automatically using a multi-label classification model.

---

## ⚠️ Important Note

Due to size limitations, the pre-trained models (`.keras`, `.pkl`, `.npy`, etc.) could not be uploaded to GitHub.

To fully use the project:

- Refer to the Jupyter notebooks provided in the repository to train and save the models.
- After training, make sure to place the models in the correct folders as expected by the app.
- Alternatively, you may download the models from an external source (to be provided).

---

## ⚙️ How to Run

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
