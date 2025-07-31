import pandas as pd
import gradio as gr
import plotly.graph_objs as go
import mlflow
from sentence_transformers import SentenceTransformer
import faiss
import json
import math

# Load legal texts
LEGAL_TEXTS_PATH = "compliance/legal_texts.txt"
with open(LEGAL_TEXTS_PATH, "r") as f:
    legal_texts = [line.strip() for line in f if line.strip()]

# Embed legal texts
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(legal_texts)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Load simple model weights for demonstration
MODEL_PATH = "data/model.json"
try:
    with open(MODEL_PATH) as f:
        MODEL_WEIGHTS = json.load(f)
except FileNotFoundError:
    MODEL_WEIGHTS = None


def retrieve_text(query: str, top_k: int = 1):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, top_k)
    return [legal_texts[i] for i in indices[0]]


def analyze_query(query: str):
    mlflow.start_run()
    relevant = retrieve_text(query)[0]
    # Placeholder recommendation using simple string concat
    recommendation = f"Based on legal guidance: {relevant} -> ensure compliance by following this clause."
    mlflow.log_param("query", query)
    mlflow.log_metric("dummy_score", 0.9)
    mlflow.end_run()
    fig = go.Figure(go.Indicator(mode="number", value=90, title="Compliance Score"))
    return recommendation, fig


def upload_data(df: pd.DataFrame):
    return df.head()


def run_model(df: pd.DataFrame):
    """Return predictions using the simple JSON weights."""
    if MODEL_WEIGHTS is None:
        return df
    preds = []
    for _, row in df.iterrows():
        g = 1.0 if row.get("Gender") == "Male" else 0.0
        e = {"A": 0.0, "B": 1.0, "C": 2.0}.get(row.get("Ethnicity"), 0.0)
        z = (
            MODEL_WEIGHTS["age"] * row.get("Age", 0)
            + MODEL_WEIGHTS["gender"] * g
            + MODEL_WEIGHTS["ethnicity"] * e
            + MODEL_WEIGHTS["bias"]
        )
        pred = 1.0 / (1.0 + math.exp(-z))
        preds.append(pred)
    df = df.copy()
    df["Prediction"] = preds
    return df


with gr.Blocks() as demo:
    gr.Markdown("# Compliance Checker")
    with gr.Tab("Upload Data"):
        dataset = gr.Dataframe(headers=["Age", "Gender", "Ethnicity"], row_count=3)
        upload_btn = gr.Button("Upload")
        upload_output = gr.Dataframe()
        upload_btn.click(upload_data, inputs=dataset, outputs=upload_output)
    with gr.Tab("Model Prediction"):
        sample_btn = gr.Button("Load Sample Data")
        predict_btn = gr.Button("Predict")
        data_view = gr.Dataframe()
        sample_btn.click(lambda: pd.read_csv("data/sample_data.csv"), outputs=data_view)
        predict_btn.click(run_model, inputs=data_view, outputs=data_view)
    with gr.Tab("Ask Compliance Question"):
        query = gr.Textbox(label="Compliance Query")
        submit_btn = gr.Button("Submit")
        recommendation = gr.Textbox(label="Recommendation")
        chart = gr.Plot()
        submit_btn.click(analyze_query, inputs=query, outputs=[recommendation, chart])

if __name__ == "__main__":
    demo.launch()
