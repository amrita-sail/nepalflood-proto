# Nepal Flood Comment Classifier

A Streamlit prototype that classifies Nepal flood comments with the bundled TF-IDF + LinearSVC model and explains each prediction with LIME in the interface.

## Run locally

Use Python 3.10–3.12 to match the model's scikit-learn 1.6.1 serialization format.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Six categories include two clickable examples each. Select one to fill the input and run classification with a local explanation. The unrelated category has no sample buttons by design.

## Notes

- The model bundle is in `models/best_classical_model.pkl`.
- The prediction uses the model's direct LinearSVC class output.
- LinearSVC does not return calibrated probabilities. The displayed class scores are softmax-normalized decision margins, used as relative scores for LIME.
- LIME attributions are local approximations for the selected text.
- Load the bundled pickle only as a trusted project artifact.
