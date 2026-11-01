"""LegalLoRA dashboard: try the tuned model live and browse the evaluation results.

Run from the repo root:  .venv/bin/streamlit run app/dashboard.py
"""

import html
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import COARSE, messages, parse_prediction  # noqa: E402

MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
ADAPTER = ROOT / "adapters"
COLORS = {"PER": "#c084fc", "ORG": "#22d3ee", "LOC": "#4ade80", "NRM": "#facc15",
          "REG": "#60a5fa", "RS": "#f87171", "LIT": "#f9a8d4"}
RUNS = {"Zero-shot": "base_zeroshot", "5-shot prompt": "base_fewshot5",
        "LoRA (step 250)": "tuned_step250"}

MODEL_SCALE = alt.Scale(domain=list(RUNS), range=["#94a3b8", "#60a5fa", "#4ade80"])

st.set_page_config(page_title="LegalLoRA", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Loading Qwen2.5-3B + LoRA adapter ...")
def load_model():
    from mlx_lm import load
    return load(MODEL, adapter_path=str(ADAPTER))


@st.cache_data
def load_results():
    return {name: json.loads((ROOT / "results" / f"{f}.json").read_text(encoding="utf-8"))
            for name, f in RUNS.items()}


@st.cache_data
def load_examples():
    rows = []
    for line in (ROOT / "data" / "test.jsonl").open(encoding="utf-8"):
        m = json.loads(line)["messages"]
        rows.append({"sentence": m[1]["content"], "entities": json.loads(m[2]["content"])})
    return rows


def predict(sentence: str):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = load_model()
    prompt = tok.apply_chat_template(messages(sentence), add_generation_prompt=True)
    raw = generate(model, tok, prompt=prompt, max_tokens=300, sampler=make_sampler(temp=0.0))
    return parse_prediction(raw), raw


def highlight(sentence: str, entities: list[dict]) -> str:
    spans, taken = [], []
    for e in entities:
        start = sentence.find(e["text"])
        while start != -1 and any(start < b and a < start + len(e["text"]) for a, b in taken):
            start = sentence.find(e["text"], start + 1)
        if start != -1:
            taken.append((start, start + len(e["text"])))
            spans.append((start, start + len(e["text"]), e["type"]))
    out, pos = [], 0
    for a, b, t in sorted(spans):
        out.append(html.escape(sentence[pos:a]))
        out.append(f'<span style="background:{COLORS[t]};color:#111;padding:2px 6px;'
                   f'border-radius:6px;font-weight:600">{html.escape(sentence[a:b])} '
                   f'<sub>{t}</sub></span>')
        pos = b
    out.append(html.escape(sentence[pos:]))
    return f'<div style="font-size:1.1rem;line-height:2.2">{"".join(out)}</div>'


def legend():
    chips = "".join(f'<span style="background:{COLORS[t]};color:#111;padding:2px 8px;'
                    f'border-radius:6px;margin-right:6px;font-size:.8rem">{t}: {d}</span>'
                    for t, d in COARSE.items())
    st.markdown(chips, unsafe_allow_html=True)


st.title("⚖️ LegalLoRA")
st.caption("A 3B model, LoRA-tuned on a MacBook, extracting German legal entities as JSON.")
tab_try, tab_results = st.tabs(["Try it", "Results"])

with tab_try:
    examples = load_examples()
    with_ents = [r for r in examples if r["entities"] and len(r["sentence"]) <= 220]
    choice = st.selectbox(
        "Pick a held-out test sentence, or type your own below",
        ["(write my own)"] + [f"{i + 1}. {r['sentence'][:90]}…" for i, r in enumerate(with_ents)],
    )
    gold = None
    default = ""
    if choice != "(write my own)":
        row = with_ents[int(choice.split(".")[0]) - 1]
        default, gold = row["sentence"], row["entities"]
    sentence = st.text_area("German sentence (tokens separated by spaces, as in the dataset)",
                            value=default, height=100, key=f"s_{choice}")
    legend()
    if st.button("Extract entities", type="primary") and sentence.strip():
        with st.spinner("Running the tuned model ..."):
            pred, raw = predict(sentence.strip())
        if pred is None:
            st.error("The model output was not valid JSON.")
            st.code(raw)
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Model")
                st.markdown(highlight(sentence, pred), unsafe_allow_html=True)
                st.json(pred)
            with c2:
                st.subheader("Expert labels")
                if gold is not None:
                    st.markdown(highlight(sentence, gold), unsafe_allow_html=True)
                    st.json(gold)
                    got = {(e["text"], e["type"]) for e in pred}
                    want = {(e["text"], e["type"]) for e in gold}
                    st.success(f"{len(got & want)}/{len(want)} expert entities matched exactly, "
                               f"{len(got - want)} extra predicted")
                else:
                    st.info("Only available for the held-out test sentences.")

with tab_results:
    res = load_results()
    st.subheader("Entity F1 on 500 held-out sentences")
    summary = pd.DataFrame([{
        "Model": n, "F1": r["micro"]["f1"], "Precision": r["micro"]["precision"],
        "Recall": r["micro"]["recall"], "Valid JSON": r["json_valid_rate"],
        "Latency (s)": r["mean_latency_s"]} for n, r in res.items()])
    left, right = st.columns([3, 2])
    with left:
        chart = alt.Chart(summary).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("F1:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Model:N", sort=None, title=None),
            color=alt.Color("Model:N", legend=None, scale=MODEL_SCALE),
            tooltip=["Model", alt.Tooltip("F1:Q", format=".3f")]).properties(height=180)
        st.altair_chart(chart, use_container_width=True)
    with right:
        st.dataframe(summary.style.format({"F1": "{:.3f}", "Precision": "{:.3f}",
                                           "Recall": "{:.3f}", "Valid JSON": "{:.1%}",
                                           "Latency (s)": "{:.1f}"}),
                     hide_index=True, use_container_width=True)
    st.subheader("F1 by entity type")
    rows = [{"Type": t, "Model": n, "F1": r["per_type"][t]["f1"],
             "n": r["per_type"][t]["support"]} for n, r in res.items() for t in COARSE]
    per = pd.DataFrame(rows)
    st.altair_chart(alt.Chart(per).mark_bar().encode(
        x=alt.X("Type:N", sort=list(COARSE), title=None),
        y=alt.Y("F1:Q", scale=alt.Scale(domain=[0, 1])),
        xOffset="Model:N",
        color=alt.Color("Model:N", scale=MODEL_SCALE),
        tooltip=["Type", "Model", alt.Tooltip("F1:Q", format=".2f"), "n"]).properties(height=320),
        use_container_width=True)
    st.caption("Exact match on text and type. Single run, 500 sentences. The LoRA model is the "
               "step-250 checkpoint of a planned 750-step epoch. The dataset splits by sentence, "
               "so train/test leakage is possible. See the README for limitations.")
