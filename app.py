"""
Streamlit Web Application for SemEval 2027 MMCultureQA.
Interactive playground, dataset explorer, benchmark dashboard, and CodaBench submission generator.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st

from mmcultureqa.asr import SpeechTranscriber
from mmcultureqa.config import Config, SUPPORTED_LANGUAGES
from mmcultureqa.cultural_knowledge import CulturalKnowledgeBase, CulturalRetriever
from mmcultureqa.dataset import MMCultureQADataset
from mmcultureqa.evaluation import CodaBenchPackager, MMCultureQAEvaluator, compute_all_metrics
from mmcultureqa.models import CulturalVLMSolver, LocalCulturalBaseline, MMCultureQAPipeline

# Page setup
st.set_page_config(
    page_title="MMCultureQA 2027 Prototype",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .cultural-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4F46E5;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_components():
    """Cache core pipeline components across sessions."""
    cfg = Config()
    cfg.ensure_directories()
    kb = CulturalKnowledgeBase()
    retriever = CulturalRetriever(kb)
    transcriber = SpeechTranscriber()
    baseline_solver = LocalCulturalBaseline(kb)
    return cfg, kb, retriever, transcriber, baseline_solver


cfg, kb, retriever, transcriber, baseline_solver = load_components()

# Sidebar
st.sidebar.title("MMCultureQA 2027")
st.sidebar.markdown(
    "**Shared Task**: Multilingual Multimodal Cultural Question Answering ([SemEval 2027](https://mmcultureqa-semeval27.github.io/))"
)

st.sidebar.subheader("Model Configuration")
solver_choice = st.sidebar.radio(
    "Reasoning Solver",
    ["Local Cultural Baseline (Fast, Offline)", "Cultural VLM (RAG with Vision-LLM)"],
    index=0,
)

api_key_input = None
vlm_model_name = "gpt-4o-mini"
if "Cultural VLM" in solver_choice:
    api_key_input = st.sidebar.text_input(
        "OpenAI API Key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        help="Required for remote Vision-Language Model inference.",
    )
    vlm_model_name = st.sidebar.selectbox("VLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], index=0)

st.sidebar.subheader("Language Track")
lang_options = list(SUPPORTED_LANGUAGES.keys())
selected_lang = st.sidebar.selectbox(
    "Active Language",
    lang_options,
    format_func=lambda x: f"{SUPPORTED_LANGUAGES[x]['name']} ({x.upper()})",
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Official Ranking Metric**: BERTScore F1. Auxiliary metrics include SacreBLEU & ROUGE-L."
)

# Header
st.markdown('<div class="main-title">MMCultureQA 2027 🖼️🗣️</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Culturally Grounded Spoken & Textual Visual Question Answering Prototype (SemEval 2027)</div>',
    unsafe_allow_html=True,
)

# Main Navigation Tabs
tab_playground, tab_explorer, tab_eval, tab_submit = st.tabs([
    "🎯 Interactive Playground",
    "📚 Dataset Explorer",
    "📊 Benchmark & Evaluation",
    "📦 CodaBench Submission",
])


# ==========================================
# TAB 1: INTERACTIVE PLAYGROUND
# ==========================================
with tab_playground:
    st.subheader("Test Multimodal Cultural Reasoning")
    st.markdown("Select an image and ask a question either via text (Task 2) or audio (Task 1).")

    # Mode selection
    task_mode = st.radio("Select Task", ["Task 2: Textual Visual QA (Text + Image)", "Task 1: Spoken Visual QA (Audio + Image)"], horizontal=True)
    is_spoken = "Task 1" in task_mode

    # Load sample split for convenience
    try:
        sample_ds = MMCultureQADataset.load_split(split="dev", task="sqa" if is_spoken else "qa", lang=selected_lang)
    except Exception:
        sample_ds = MMCultureQADataset.load_split(split="dev", task="sqa" if is_spoken else "qa", lang="en")

    col_media, col_qa = st.columns([1, 1])

    with col_media:
        st.markdown("#### 1. Visual Input")
        use_sample = st.checkbox("Use a dataset sample", value=True)

        selected_record = None
        current_image_path = None

        if use_sample and len(sample_ds) > 0:
            sample_labels = [f"Item {i+1}: {r.country or 'Culture'} - {r.category or 'General'}" for i, r in enumerate(sample_ds)]
            selected_idx = st.selectbox("Pick Dataset Sample", range(len(sample_labels)), format_func=lambda i: sample_labels[i])
            selected_record = sample_ds[selected_idx]
            current_image_path = selected_record.resolve_image_path()
            if current_image_path.exists():
                st.image(str(current_image_path), caption=f"ID: {selected_record.id[:12]}... ({selected_record.country})", use_container_width=True)
        else:
            uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            if uploaded_img:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                tfile.write(uploaded_img.read())
                current_image_path = Path(tfile.name)
                st.image(str(current_image_path), caption="Uploaded Image", use_container_width=True)

    with col_qa:
        st.markdown("#### 2. Question & Cultural Reasoning")
        current_audio_path = None
        question_text = ""

        if is_spoken:
            # Task 1: Spoken QA
            st.markdown("##### Spoken Audio Question (Task 1)")
            if use_sample and selected_record and selected_record.audio:
                current_audio_path = selected_record.resolve_audio_path()
                if current_audio_path and current_audio_path.exists():
                    st.audio(str(current_audio_path), format="audio/wav")
            else:
                uploaded_audio = st.file_uploader("Upload Audio Question (WAV)", type=["wav"])
                if uploaded_audio:
                    t_aud = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    t_aud.write(uploaded_audio.read())
                    current_audio_path = Path(t_aud.name)
                    st.audio(str(current_audio_path), format="audio/wav")

        else:
            # Task 2: Text QA
            st.markdown("##### Written Question (Task 2)")
            default_q = selected_record.question if (use_sample and selected_record and selected_record.question) else ""
            question_text = st.text_area("Question", value=default_q, height=90)

        # Cultural Context Grounding Inspector
        query_for_retrieval = question_text or (selected_record.question if selected_record else "")
        if query_for_retrieval:
            with st.expander("🔍 Inspect Retrieved Cultural Knowledge (Cultural RAG)", expanded=False):
                matched = retriever.retrieve(
                    query_for_retrieval,
                    country=selected_record.country if selected_record else None,
                    category=selected_record.category if selected_record else None,
                    top_k=2,
                )
                for ent, score in matched:
                    st.markdown(f"- **{ent.name}** ({ent.country_or_region}) — Relevance: `{score:.1f}`")
                    st.caption(ent.description)

        st.markdown("---")
        btn_generate = st.button("🚀 Generate Cultural Answer", type="primary", use_container_width=True)

        if btn_generate:
            if not current_image_path or not current_image_path.exists():
                st.error("Please provide a valid image.")
            elif is_spoken and (not current_audio_path or not current_audio_path.exists()):
                st.error("Please provide a valid audio WAV file for Task 1.")
            elif not is_spoken and not question_text.strip():
                st.error("Please enter a question for Task 2.")
            else:
                with st.spinner("Analyzing multimodal and cultural context..."):
                    # Initialize chosen solver
                    active_solver = baseline_solver
                    if "Cultural VLM" in solver_choice:
                        try:
                            active_solver = CulturalVLMSolver(model_name=vlm_model_name, api_key=api_key_input)
                        except Exception as e:
                            st.warning(f"VLM Initialization failed: {e}. Defaulting to Local Cultural Baseline.")
                            active_solver = baseline_solver

                    pipeline = MMCultureQAPipeline(solver=active_solver, transcriber=transcriber)
                    start_time = time.time()

                    if is_spoken:
                        res = pipeline.run_task1_sqa(
                            image_path=current_image_path,
                            audio_path=current_audio_path,
                            lang=selected_lang,
                            country=selected_record.country if selected_record else None,
                            category=selected_record.category if selected_record else None,
                        )
                        st.info(f"🎙️ **ASR Transcription:** {res['transcription']}")
                        pred_answer = res["prediction"]
                    else:
                        res = pipeline.run_task2_qa(
                            image_path=current_image_path,
                            question=question_text,
                            lang=selected_lang,
                            country=selected_record.country if selected_record else None,
                            category=selected_record.category if selected_record else None,
                        )
                        pred_answer = res["prediction"]

                    latency = time.time() - start_time

                st.success("### Prediction Generated:")
                st.markdown(f"> **{pred_answer}**")
                st.caption(f"Inference latency: {latency:.2f}s | Solver: {solver_choice.split('(')[0].strip()}")

                # If reference answer is available, compare with metrics
                if selected_record and selected_record.answer:
                    st.markdown("#### Reference Comparison")
                    st.markdown(f"**Gold Reference:** {selected_record.answer}")

                    scores = compute_all_metrics([pred_answer], [selected_record.answer], lang=selected_lang)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("BERTScore F1", f"{scores['bert_score_f1']:.4f}")
                    c2.metric("BLEU-4", f"{scores['bleu_4']:.1f}")
                    c3.metric("ROUGE-L", f"{scores['rouge_l']:.1f}")
                    c4.metric("Token F1", f"{scores['token_f1']:.1f}%")


# ==========================================
# TAB 2: DATASET EXPLORER
# ==========================================
with tab_explorer:
    st.subheader("Explore OASIS & MMCultureQA Data")
    c_split, c_task, c_lang = st.columns(3)
    exp_split = c_split.selectbox("Split", ["dev", "train"])
    exp_task = c_task.selectbox("Task Type", ["qa", "sqa"], format_func=lambda x: "Task 2: Text QA (qa)" if x == "qa" else "Task 1: Spoken QA (sqa)")
    exp_lang = c_lang.selectbox("Language", ["en", "msa"])

    try:
        explore_ds = MMCultureQADataset.load_split(split=exp_split, task=exp_task, lang=exp_lang)
        stats = explore_ds.summary_statistics()

        st.markdown(f"Loaded **{len(explore_ds)}** records from `{explore_ds.source_file.name if explore_ds.source_file else 'split'}`")

        # Filters
        f1, f2 = st.columns(2)
        all_countries = ["All"] + [c[0] for c in stats["countries"]]
        filter_country = f1.selectbox("Filter by Country", all_countries)

        all_categories = ["All"] + [cat[0] for cat in stats["categories"]]
        filter_cat = f2.selectbox("Filter by Category", all_categories)

        filtered_records = explore_ds.records
        if filter_country != "All":
            filtered_records = [r for r in filtered_records if r.country == filter_country]
        if filter_cat != "All":
            filtered_records = [r for r in filtered_records if r.category == filter_cat]

        st.markdown(f"Displaying **{len(filtered_records)}** records:")

        for r in filtered_records:
            with st.container():
                st.markdown("---")
                col_img, col_info = st.columns([1, 2])
                with col_img:
                    img_p = r.resolve_image_path()
                    if img_p.exists():
                        st.image(str(img_p), use_container_width=True)
                with col_info:
                    st.markdown(f"**ID:** `{r.id}`")
                    st.markdown(
                        f"<span class='cultural-badge'>🌍 {r.country or 'General'}</span>"
                        f"<span class='cultural-badge'>🏷️ {r.category or 'General'}</span>",
                        unsafe_allow_html=True,
                    )
                    if r.subcategory:
                        st.caption(f"Subcategory: {r.subcategory}")

                    if r.question:
                        st.markdown(f"**Question:** {r.question}")
                    if r.audio:
                        aud_p = r.resolve_audio_path()
                        if aud_p and aud_p.exists():
                            st.audio(str(aud_p), format="audio/wav")

                    if r.answer:
                        st.markdown(f"**Answer:** {r.answer}")

    except Exception as e:
        st.error(f"Failed to load split: {e}")


# ==========================================
# TAB 3: BENCHMARK & EVALUATION
# ==========================================
with tab_eval:
    st.subheader("Official Benchmark Evaluation Harness")
    st.markdown("Benchmark the model against the development split using official SemEval 2027 metrics.")

    b_col1, b_col2, b_col3 = st.columns(3)
    bench_task = b_col1.selectbox("Benchmark Task", ["qa", "sqa"], key="bench_task")
    bench_lang = b_col2.selectbox("Track", ["en", "msa"], key="bench_lang")
    bench_limit = b_col3.slider("Evaluation Limit", min_value=1, max_value=15, value=6)

    btn_eval = st.button("📊 Run Benchmark Evaluation", type="primary")

    if btn_eval:
        with st.spinner(f"Evaluating {bench_task.upper()} ({bench_lang.upper()})..."):
            dev_ds = MMCultureQADataset.load_split(split="dev", task=bench_task, lang=bench_lang)
            pipeline = MMCultureQAPipeline(solver=baseline_solver, transcriber=transcriber)

            preds = pipeline.predict_dataset(dev_ds, limit=bench_limit)
            evaluator = MMCultureQAEvaluator(preds, dev_ds, lang=bench_lang)
            results = evaluator.evaluate()

            overall = results["overall"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("BERTScore F1 (Official)", f"{overall['bert_score_f1']:.4f}")
            m2.metric("BLEU-4", f"{overall['bleu_4']:.2f}")
            m3.metric("ROUGE-L", f"{overall['rouge_l']:.2f}")
            m4.metric("Exact Match", f"{overall['exact_match']:.1f}%")

            st.markdown("#### Performance Breakdown by Cultural Category")
            cat_data = []
            for cat, scores in results["by_category"].items():
                cat_data.append({
                    "Category": cat,
                    "Count": scores["count"],
                    "BERTScore F1": scores["bert_score_f1"],
                    "BLEU-4": scores["bleu_4"],
                    "ROUGE-L": scores["rouge_l"],
                })
            df_cat = pd.DataFrame(cat_data)
            st.dataframe(df_cat, use_container_width=True)

            st.markdown("#### Performance Breakdown by Country")
            country_data = []
            for c, scores in results["by_country"].items():
                country_data.append({
                    "Country": c,
                    "Count": scores["count"],
                    "BERTScore F1": scores["bert_score_f1"],
                    "BLEU-4": scores["bleu_4"],
                    "ROUGE-L": scores["rouge_l"],
                })
            df_country = pd.DataFrame(country_data)
            st.dataframe(df_country, use_container_width=True)


# ==========================================
# TAB 4: CODABENCH SUBMISSION BUILDER
# ==========================================
with tab_submit:
    st.subheader("CodaBench Submission Archive Builder")
    st.markdown("Generate and package validated predictions into `submission.zip` for CodaBench upload.")

    sub_task = st.selectbox("Target Task", ["qa", "sqa"], key="sub_task")
    sub_lang = st.selectbox("Target Language Track", ["en", "msa"], key="sub_lang")

    btn_create_sub = st.button("📦 Generate & Package Submission", type="primary")

    if btn_create_sub:
        with st.spinner("Generating predictions and assembling submission archive..."):
            dev_ds = MMCultureQADataset.load_split(split="dev", task=sub_task, lang=sub_lang)
            pipeline = MMCultureQAPipeline(solver=baseline_solver, transcriber=transcriber)
            preds = pipeline.predict_dataset(dev_ds)

            out_zip = cfg.submissions_dir / f"submission_{sub_task}_{sub_lang}.zip"
            meta = CodaBenchPackager.create_submission_zip(preds, out_zip)

            st.success("✅ Submission package successfully created and validated!")
            c_s1, c_s2, c_s3 = st.columns(3)
            c_s1.metric("Archive Name", Path(meta["submission_zip"]).name)
            c_s2.metric("Records Included", meta["records_count"])
            c_s3.metric("Size", f"{meta['size_bytes']} bytes")

            # Download button
            with open(out_zip, "rb") as fp:
                st.download_button(
                    label="⬇️ Download submission.zip",
                    data=fp,
                    file_name=f"submission_{sub_task}_{sub_lang}.zip",
                    mime="application/zip",
                )
