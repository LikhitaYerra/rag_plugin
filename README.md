# RAG Compliance Plugin

This project demonstrates a simple compliance checker using Retrieval-Augmented Generation (RAG). It allows a researcher to query legal standards such as GDPR, the EU AI Act, the FAIR Principles, and HIPAA. Results are logged with MLflow, visualized in Plotly, and displayed through a Gradio interface.

## Usage

Install the dependencies and run the application directly:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860` to upload a dataset and enter compliance questions.

### Synthetic Data and Model

The repository includes a small script to create synthetic healthcare data and
a lightweight logistic regression model implemented without external
dependencies. Run it with:

```bash
python data/generate_synthetic.py
```

This generates `data/sample_data.csv` and `data/model.json` which can be used
for testing the compliance workflow.
