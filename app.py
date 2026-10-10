from html import escape
from pathlib import Path
import pickle
import re

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

EXAMPLES = [
    ("China / Dam & Conspiracy Theories", [
        "China built this dam to control the river and hide the truth.",
        "This was sabotage by China; they caused the dam to collapse.",
    ]),
    ("Climate Change & Natural Causes", [
        "Heavy monsoon rain and climate change caused the river to overflow.",
        "The extreme rainfall is a result of changing climate patterns.",
    ]),
    ("Death/Damage Toll & Casualty Speculation", [
        "The death toll has risen to 120, and dozens are still missing.",
        "So many people have died and homes were destroyed in the flood.",
    ]),
    ("Government, Infrastructure & Development Criticism", [
        "The government ignored warnings and failed to maintain the flood defenses.",
        "Authorities should build stronger bridges and improve infrastructure.",
    ]),
    ("Prayers, Religious & Condolence Messages", [
        "Praying for everyone affected and their families. May they find strength.",
        "Heartfelt condolences to those who lost loved ones.",
    ]),
    ("Reactions to Footage & Specific Moments", [
        "The video shows the water rushing through the street.",
        "That moment when the bridge gives way is terrifying.",
    ]),
]

st.set_page_config(page_title="Nepal Flood Comment Classifier", page_icon="🌊", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#172d35; --muted:#60757c; --teal:#087e83; --aqua:#dff5f1; --line:#dbe8e6; }
.stApp { background: radial-gradient(ellipse at 8% 0%, #e6f5f2 0, transparent 34%), linear-gradient(180deg,#f8fbfa 0%,#f3f7f6 100%); color:var(--ink); }
.block-container { max-width: 1160px; padding-top: 2.2rem; padding-bottom: 4rem; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; color:var(--ink); }
.hero { padding: 1rem 0 1.4rem; }
.eyebrow { color:var(--teal); font-size:.76rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
.hero-title { font-family:'Manrope',sans-serif; font-size:clamp(2.4rem,5vw,4rem); line-height:1.04; font-weight:800; color:#173840; margin:.45rem 0 .65rem; }
.hero-copy { font-size:1.05rem; color:#526b72; max-width:760px; line-height:1.65; }
.st-key-example_card [data-testid="stVerticalBlockBorderWrapper"] { background:rgba(255,255,255,.9); border:1px solid var(--line); border-radius:20px; padding:.8rem .9rem; box-shadow:0 10px 32px rgba(27,72,73,.055); }
.section-kicker { font-size:.74rem; color:#648087; font-weight:800; letter-spacing:.13em; text-transform:uppercase; margin-bottom:.45rem; }
.example-title { color:#173840; font-size:1rem; font-weight:800; line-height:1.35; min-height:2.7rem; padding:.1rem 0 .5rem; }
.stTextArea textarea { border-radius:14px; border:1px solid #cbdedb; background:#fcfefd; font-size:1rem; }
.stButton button[kind="primary"] { background:linear-gradient(100deg,#087e83,#0c9a8c); border:0; border-radius:12px; min-height:3rem; font-weight:800; box-shadow:0 8px 18px rgba(8,126,131,.18); }
.stButton button { border-radius:12px; border-color:#d3e3e0; color:#24454c; background:#fff; transition:all .15s ease; }
.stButton button:hover { border-color:#0c9691; color:#087e83; transform:translateY(-1px); }
.result-card { background:linear-gradient(120deg,#e3f5f1,#f5fbf9); border:1px solid #c7e8e1; border-radius:18px; padding:1.2rem 1.4rem; margin:.4rem 0 1rem; }
.result-label { color:#087e83; text-transform:uppercase; font-size:.72rem; font-weight:800; letter-spacing:.13em; }
.result-name { color:#173840; font-family:'Manrope',sans-serif; font-size:1.55rem; font-weight:800; margin-top:.25rem; line-height:1.25; }
.score-note { color:#5d757a; font-size:.87rem; margin-top:.55rem; }
.contrib { display:flex; align-items:center; gap:12px; margin:9px 0; }
.contrib-term { width:180px; color:#29464c; font-size:.9rem; font-weight:600; overflow-wrap:anywhere; }
.contrib-track { flex:1; height:12px; border-radius:8px; background:#edf2f1; overflow:hidden; }
.contrib-bar { height:100%; border-radius:8px; }
.contrib-value { width:68px; text-align:right; font-variant-numeric:tabular-nums; color:#60757c; font-size:.8rem; }
.legend { color:#60757c; font-size:.82rem; }
div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:14px; background:#fff; }
@media(max-width:700px) { .contrib-term{width:120px}.block-container{padding-left:1rem;padding-right:1rem} }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Nepal flood · community comments</div>
  <div class="hero-title">Nepal Flood Comment Classifier</div>
  <div class="hero-copy">Explore how comments are grouped and which words influenced each prediction. Choose an example to try it, or paste in a comment of your own.</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # This bundled pickle is a trusted project artifact.
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
    """Return normalized LinearSVC margins for LIME's classifier API.

    The normalized margins are relative scores, not calibrated probabilities.
    """
    margins = np.asarray(model.decision_function(list(texts)))
    if margins.ndim == 1:
        margins = np.column_stack([-margins, margins])
    return softmax(margins, axis=1)


def use_example(example):
    st.session_state["comment_input"] = example
    st.session_state["run_prediction"] = True


st.markdown('<div class="section-kicker">Try a sample comment</div>', unsafe_allow_html=True)
columns = st.columns(3, gap="medium")
for index, (label, examples) in enumerate(EXAMPLES):
    with columns[index % 3]:
        with st.container(border=True, key=f"example_card_{index}"):
            st.markdown(f'<div class="example-title">{escape(label)}</div>', unsafe_allow_html=True)
            for example_index, example in enumerate(examples):
                st.button(
                    example,
                    key=f"example_{index}_{example_index}",
                    use_container_width=True,
                    on_click=use_example,
                    args=(example,),
                )

st.markdown("<br>", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown('<div class="section-kicker">Your comment</div>', unsafe_allow_html=True)
    text = st.text_area(
        "Paste a comment to classify",
        placeholder="For example: The government should have warned people sooner…",
        height=120,
        max_chars=5000,
        label_visibility="collapsed",
        key="comment_input",
    )
    controls = st.columns([1, 2, 3])
    with controls[0]:
        clicked = st.button("Classify comment", type="primary", use_container_width=True)
    with controls[1]:
        num_features = st.slider("Words to explain", 5, 15, 9)

example_requested_run = st.session_state.pop("run_prediction", False)
run_now = clicked or example_requested_run
if run_now:
    if not text.strip():
        st.warning("Enter a comment first.")
    else:
        with st.spinner("Reading the comment and preparing its explanation…"):
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

        st.markdown('<div class="section-kicker">Model result</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-card"><div class="result-label">Predicted category</div>'
            f'<div class="result-name">{escape(CLASS_NAMES[predicted_id])}</div>'
            f'<div class="score-note">Relative LinearSVC margin score: {score_vector[predicted_id]:.3f} · not a calibrated probability</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("### What shaped this prediction?")
        st.caption("LIME changes words in the comment and observes how the model responds. Green bars support the predicted category; coral bars push away from it. This is a local approximation for this comment.")

        if explanation:
            largest = max(abs(weight) for _, weight in explanation) or 1.0
            for term, weight in explanation:
                color = "#168477" if weight >= 0 else "#d16d62"
                width = max(2.0, 100 * abs(weight) / largest)
                st.markdown(
                    f'<div class="contrib"><div class="contrib-term">{escape(term)}</div>'
                    f'<div class="contrib-track"><div class="contrib-bar" style="width:{width:.1f}%;background:{color}"></div></div>'
                    f'<div class="contrib-value">{weight:+.3f}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("LIME did not find distinct token contributions for this very short comment.")
        st.markdown('<div class="legend">Bars are sized relative to the strongest token in this explanation.</div>', unsafe_allow_html=True)

        with st.expander("See relative scores for all categories"):
            st.caption("These normalized margins help LIME compare perturbations; they are not probabilities.")
            ranked = sorted(enumerate(score_vector), key=lambda pair: pair[1], reverse=True)
            for class_id, score in ranked:
                st.write(f"**{CLASS_NAMES[class_id]}** · {score:.3f}")

with st.expander("About the model and explanations"):
    st.write(
        "This prototype uses a word and character TF-IDF feature pipeline with a class-balanced LinearSVC. "
        "The model's class prediction is direct. LinearSVC does not produce calibrated probabilities, so the "
        "score list uses softmax-normalized decision margins. LIME uses those relative scores to estimate "
        "token contributions. Explanations describe a local approximation, not a causal explanation."
    )
