# Nepal Flood Comment Classifier

A Streamlit app that classifies comments with the bundled TF-IDF + LinearSVC model and displays LIME token contributions inline. Twelve small, unlabeled sample buttons appear below the text box, with two examples for each category other than unrelated/unintelligible.

## Run locally

Use Python 3.10–3.12 to match the model's scikit-learn 1.6.1 serialization format.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

LinearSVC does not provide calibrated probabilities. The displayed normalized margins are relative scores, and LIME uses them to estimate local token contributions.
