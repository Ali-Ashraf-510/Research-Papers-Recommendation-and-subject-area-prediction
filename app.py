from flask import Flask, render_template, request
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from keras.models import load_model

app = Flask(__name__)

df = pd.read_csv(r"Dataset\arxiv_data_210930-054931.csv")
df.drop_duplicates(inplace=True)

loaded_model = load_model("Subject_area models\subject_area_model.keras")

with open(r'Subject_area models\tfidf_vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

with open('Subject_area models\label_binarizer.pkl', 'rb') as f:
    loaded_mlb = pickle.load(f)

with open("recommendation_models\embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

rec_model = SentenceTransformer("all-MiniLM-L6-v2")

# 🧠 Mapping dictionary for subject areas
subject_area_mapping = {
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.LG": "Machine Learning",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.RO": "Robotics",
    "cs.SI": "Social and Information Networks",
    "cs.DS": "Data Structures and Algorithms",
    "cs.DB": "Databases",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.FL": "Formal Languages and Automata Theory",
    "cs.GT": "Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.NI": "Networking and Internet Architecture",
    "cs.PL": "Programming Languages",
    "cs.SD": "Software Engineering",
    "cs.SE": "Software Engineering",
    "cs.SY": "Systems and Control",
    "eess.IV": "Image and Video Processing",
    "stat.ML": "Statistical Machine Learning",
    "math.ST": "Statistics",
    "math.OC": "Optimization and Control"
}


def map_subject_labels(labels):
    return [subject_area_mapping.get(label, label) for label in labels]

def recommend_papers(input_title, top_n=5):
    input_embedding = rec_model.encode([input_title])
    sim_scores = cosine_similarity(input_embedding, embeddings)[0]
    top_indices = sim_scores.argsort()[::-1][:top_n]
    # نرجع السطور كاملة لأننا محتاجين العنوان والabstract مع بعض
    results_df = df.iloc[top_indices][['titles', 'abstracts']]
    return results_df

def predict_subject_areas(abstract_text, model, vectorizer, mlb, threshold=0.5):
    abstract_vector = vectorizer.transform([abstract_text])
    abstract_vector_dense = abstract_vector.toarray()
    predictions = model.predict(abstract_vector_dense)
    binary_predictions = (predictions[0] > threshold).astype(int)
    predicted_subjects = [mlb.classes_[i] for i, val in enumerate(binary_predictions) if val == 1]
    readable_labels = map_subject_labels(predicted_subjects)
    return readable_labels

@app.route('/', methods=['GET', 'POST'])
def index():
    recommended = []
    input_title = ""

    if request.method == 'POST':
        input_title = request.form.get('title', '').strip()

        if input_title:
            rec_df = recommend_papers(input_title)
            # نضيف عمود بالموضوعات المتوقعة لكل Abstract
            rec_df = rec_df.copy()
            rec_df['predicted_subjects'] = rec_df['abstracts'].apply(
                lambda x: predict_subject_areas(x, loaded_model, loaded_vectorizer, loaded_mlb)
            )
            recommended = rec_df.to_dict(orient='records')

    return render_template(
        'index.html',
        recommended=recommended,
        input_title=input_title,
    )

if __name__ == '__main__':
    app.run(debug=True)
