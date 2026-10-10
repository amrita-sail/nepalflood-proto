from html import escape
from pathlib import Path
import pickle

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
# Two examples for each category except unrelated; categories are intentionally hidden in the UI.
EXAMPLES = [
    "China built this dam to control the river and hide the truth.",
    "This was sabotage by China; they caused the dam to collapse.",
    "Heavy monsoon rain and climate change caused the river to overflow.",
    "The extreme rainfall is a result of changing climate patterns.",
    "The death toll has risen to 120, and dozens are still missing.",
    "So many people have died and homes were destroyed in the flood.",
    "The government ignored warnings and failed to maintain the flood defenses.",
    "Authorities should build stronger bridges and improve infrastructure.",
    "Praying for everyone affected and their families. May they find strength.",
    "Heartfelt condolences to those who lost loved ones.",
    "The video shows the water rushing through the street.",
    "That moment when the bridge gives way is terrifying.",
]

st.set_page_config(page_title="Nepal Flood Comment Classifier", page_icon="🌊", layout="wide")
st.title("Nepal Flood Comment Classifier")
st.caption("Classify a comment and see which words influenced the model's prediction.")

st.markdown("""
<style>
div[data-testid="stButton"] button {
    font-size: .73rem;
    line-height: 1.25;
    padding: .35rem .55rem;
    min-height: 2.6rem;
}
.contrib { display:flex; align-items:center; gap:10px; margin:7px 0; }
.contrib-term { width:175px; font-size:.88rem; overflow-wrap:anywhere; }
.contrib-track { flex:1; height:11px; border-radius:7px; background:#edf1f0; overflow:hidden; }
.contrib-bar { height:100%; border-radius:7px; }
.contrib-value { width:65px; text-align:right; font-size:.78rem; color:#68777a; font-variant-numeric:tabular-nums; }
.legend { font-size:.8rem; color:#68777a; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # This pickle is bundled with the prototype and is a trusted project artifact.
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
    """Normalized LinearSVC margins for LIME; these are not calibrated probabilities."""
    margins = np.asarray(model.decision_function(list(texts)))
    if margins.ndim == 1:
        margins = np.column_stack([-margins, margins])
    return softmax(margins, axis=1)


def use_example(example):
    st.session_state["comment_input"] = example
    st.session_state["run_prediction"] = True

text = st.text_area(
    "Comment to classify",
    placeholder="Paste a short comment here…",
    height=130,
    max_chars=5000,
    key="comment_input",
)

st.caption("Try an example")
example_columns = st.columns(3)
for i, example in enumerate(EXAMPLES):
    with example_columns[i % 3]:
        st.button(
            example,
            key=f"example_{i}",
            use_container_width=True,
            on_click=use_example,
            args=(example,),
        )

col1, col2 = st.columns([1, 3])
with col1:
    clicked = st.button("Classify and explain", type="primary", use_container_width=True)
with col2:
    num_features = st.slider("Words to show in the explanation", 5, 20, 10)

example_requested_run = st.session_state.pop("run_prediction", False)
if clicked or example_requested_run:
    text = st.session_state.get("comment_input", "")
    if not text.strip():
        st.warning("Enter a comment first.")
    else:
        with st.spinner("Classifying and preparing the explanation…"):
            predicted_id = int(model.predict([text])[0])
            score_vector = explanation_scores([text])[0]
            exp = explainer.explain_instance(
                text,
                explanation_scores,
                labels=(predicted_id,),
                num_features=num_features,
                num_samples=1000,
            )
            explanation = [(str(term), float(weight)) for term, weight in exp.as_list(label=predicted_id)]

        st.subheader("Prediction")
        st.markdown(f"### {CLASS_NAMES[predicted_id]}")
        st.caption(
            f"Normalized LinearSVC margin score: {score_vector[predicted_id]:.3f}. "
            "This is a relative score, not a calibrated probability."
        )
        st.subheader("Why this prediction?")
        st.write("LIME estimates how each token affects this prediction. Positive values support the predicted class; negative values push away from it.")
        if explanation:
            largest = max(abs(weight) for _, weight in explanation) or 1.0
            for term, weight in explanation:
                color = "#218c74" if weight >= 0 else "#d35454"
                width = max(2.0, 100 * abs(weight) / largest)
                st.markdown(
                    f'<div class="contrib"><div class="contrib-term">{escape(term)}</div>'
                    f'<div class="contrib-track"><div class="contrib-bar" style="width:{width:.1f}%;background:{color}"></div></div>'
                    f'<div class="contrib-value">{weight:+.3f}</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown('<div class="legend">Green supports the prediction; red pushes against it. Bars are scaled within this explanation.</div>', unsafe_allow_html=True)
        else:
            st.info("This comment is too short for LIME to show distinct token contributions.")

        with st.expander("Scores for every class"):
            st.caption("Softmax-normalized LinearSVC margins are relative scores, not calibrated probabilities.")
            ranked = sorted(enumerate(score_vector), key=lambda pair: pair[1], reverse=True)
            for class_id, score in ranked:
                st.write(f"**{CLASS_NAMES[class_id]}** — {score:.3f}")

with st.expander("About the model and explanation"):
    st.write(
        "The model combines word and character TF-IDF features with a class-balanced LinearSVC. "
        "LIME perturbs this comment and fits a local approximation to show which tokens influenced "
        "the prediction. These attributions are explanatory clues, not causal effects."
    )
