from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from lime.lime_text import LimeTextExplainer
from scipy.special import softmax

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "best_classical_model.pkl"
CLASS_NAMES = [
    "China / Dam & Conspiracy Theories",
    "Climate Change & Natural Causes",
    "Death/Damage Toll & Casualty Speculation",
    "Government, Infrastructure & Development Criticism",
    "Prayers, Religious & Condolence Messages",
    "Reactions to Footage & Specific Moments",
    "Unrelated/Unintelligible",
]

st.set_page_config(page_title="Comment Classifier · LIME", page_icon="🔎", layout="wide")
st.title("Comment classifier")
st.caption("TF-IDF word + character features · LinearSVC · local LIME explanations")

@st.cache_resource
def load_model():
    # This pickle is bundled with the prototype and must be treated as trusted code.
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    model_classes = list(model.named_steps["clf"].classes_)
    if model_classes != list(range(len(CLASS_NAMES))):
        raise ValueError(f"Unexpected model class IDs: {model_classes}")
    explainer = LimeTextExplainer(
        class_names=CLASS_NAMES,
        bow=True,
        random_state=42,
        split_expression=r"\W+",
    )
    return model, explainer

model, explainer = load_model()


def explanation_scores(texts):
    """Convert LinearSVC decision margins to normalized scores for LIME's API.

    These softmax-normalized margins are useful as relative perturbation scores,
    but they are not calibrated class probabilities.
    """
    margins = np.asarray(model.decision_function(list(texts)))
    if margins.ndim == 1:
        margins = np.column_stack([-margins, margins])
    return softmax(margins, axis=1)


text = st.text_area(
    "Comment to classify",
    placeholder="Paste a short comment here…",
    height=130,
    max_chars=5000,
)
col1, col2 = st.columns([1, 3])
with col1:
    explain = st.button("Classify and explain", type="primary", use_container_width=True)
with col2:
    num_features = st.slider("Words or tokens to show in the explanation", 5, 20, 10)

if explain:
    if not text.strip():
        st.warning("Enter a comment first.")
    else:
        with st.spinner("Classifying and generating a local explanation…"):
            predicted_id = int(model.predict([text])[0])
            score_vector = explanation_scores([text])[0]
            exp = explainer.explain_instance(
                text,
                explanation_scores,
                labels=(predicted_id,),
                num_features=num_features,
                num_samples=1000,
            )
            explanation = exp.as_list(label=predicted_id)

        st.subheader("Prediction")
        st.markdown(f"### {CLASS_NAMES[predicted_id]}")
        st.caption(
            f"Normalized LinearSVC margin score: {score_vector[predicted_id]:.3f}. "
            "This is not a calibrated probability."
        )

        st.subheader("Local explanation")
        st.write(
            "LIME perturbs this comment and estimates which tokens most support "
            "or oppose the predicted class. Positive weights support the class; "
            "negative weights push away from it."
        )
        terms, weights = zip(*explanation) if explanation else ([], [])
        colors = ["#218c74" if w >= 0 else "#d35454" for w in weights]
        fig, ax = plt.subplots(figsize=(9, max(2.8, 0.42 * len(terms))))
        y = np.arange(len(terms))
        ax.barh(y, weights, color=colors)
        ax.set_yticks(y, labels=terms)
        ax.invert_yaxis()
        ax.axvline(0, color="#555", linewidth=0.8)
        ax.set_xlabel("LIME weight for the predicted class")
        ax.set_title("Token contribution (local approximation)")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("Scores for every class"):
            ranked = sorted(enumerate(score_vector), key=lambda pair: pair[1], reverse=True)
            for class_id, score in ranked:
                st.write(f"**{CLASS_NAMES[class_id]}** — {score:.3f}")

with st.expander("About this prototype"):
    st.write(
        "The displayed class is the model's direct LinearSVC prediction. "
        "The score list uses softmax-normalized decision margins because this "
        "classifier does not produce calibrated probabilities. LIME uses those "
        "relative scores to estimate token contributions for this input. "
        "Explanations are local approximations and can vary with perturbation settings."
    )
