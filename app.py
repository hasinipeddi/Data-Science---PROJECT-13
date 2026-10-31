"""
app.py
======
Modern AI-Powered Product Dashboard for Handwritten Digit Recognition
Dual-Engine System: K-Nearest Neighbors (KNN) & Support Vector Machine (SVM) on MNIST.

Features
--------
* Interactive drawing canvas (streamlit-drawable-canvas) & image file upload (PNG, JPG, BMP, TIFF)
* MNIST-calibrated preprocessing: background inversion, noise floor suppression,
  bounding box cropping, 20×20 aspect-preserving resize, stroke calibration, and center-of-mass centering
* AI Model Comparison Dashboard:
  - KNN predicted digit & neighbour vote share
  - SVM predicted digit & decision margin / calibrated softmax score
  - Nearest neighbour inspection (labels & Euclidean distances)
  - Model consensus / disagreement detection with high-visibility alerts
  - Side-by-side 10-class probability and decision score chart
* Benchmark metrics viewer & IEEE report summary
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from PIL import Image
import joblib

# ── Ensure project root is in sys.path ────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from src.preprocess import preprocess_canvas_image
from src.predict import predict_single, predict_with_details, compare_models, get_confidence_label

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroDigit AI • KNN vs SVM Digit Recognition",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modern Dark AI Theme CSS ────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    /* Global Styles */
    .stApp {
        background: radial-gradient(circle at 15% 20%, #0d1527 0%, #080c14 100%);
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* AI Branding Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(8px);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
    }
    .hero-desc {
        color: #94a3b8;
        font-size: 15px;
        margin: 0;
        line-height: 1.5;
    }

    /* Cards */
    .model-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s, border-color 0.2s;
    }
    .model-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-2px);
    }
    .digit-display {
        font-size: 64px;
        font-weight: 900;
        text-align: center;
        line-height: 1;
        margin: 10px 0;
    }
    .digit-knn {
        color: #38bdf8;
        text-shadow: 0 0 25px rgba(56, 189, 248, 0.4);
    }
    .digit-svm {
        color: #c084fc;
        text-shadow: 0 0 25px rgba(192, 132, 252, 0.4);
    }

    /* Consensus / Disagreement Badges */
    .consensus-card {
        background: linear-gradient(90deg, rgba(34, 197, 94, 0.12) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 18px 0;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .disagreement-card {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.15) 0%, rgba(245, 158, 11, 0.08) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 18px 0;
        display: flex;
        align-items: center;
        gap: 15px;
    }

    /* Metric tags */
    .metric-chip {
        display: inline-block;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 13px;
        margin: 4px 4px 4px 0;
    }

    /* Hide raw streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Paths & Resource Loader ──────────────────────────────────────────────────
MODEL_DIR = os.path.join(APP_DIR, "models")
OUTPUT_DIR = os.path.join(APP_DIR, "outputs")
KNN_PATH  = os.path.join(MODEL_DIR, "knn_mnist.joblib")
SVM_PATH  = os.path.join(MODEL_DIR, "svm_mnist.joblib")

@st.cache_resource
def load_all_models():
    """Load both KNN and SVM models into memory."""
    knn = joblib.load(KNN_PATH) if os.path.exists(KNN_PATH) else None
    svm = joblib.load(SVM_PATH) if os.path.exists(SVM_PATH) else None
    return knn, svm

knn_model, svm_model = load_all_models()

# ── Sidebar Navigation & AI System Info ──────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ NeuroDigit AI")
    st.caption("College Data Science Capstone Project")
    st.markdown("---")

    nav_mode = st.radio(
        "Navigation",
        options=["🎯 Live Recognition & Comparison", "📊 Benchmark & IEEE Metrics", "🔬 Model Architecture & Diagnostics"],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Inference Configuration")
    view_mode = st.selectbox(
        "Evaluation Mode",
        options=["Dual-Model Comparison (KNN + SVM)", "KNN Focus", "SVM Focus"],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### 📐 Preprocessing Pipeline")
    with st.expander("Inspection Pipeline Details", expanded=False):
        st.markdown(
            """
            - **Grayscale & Auto-Inversion:** Outer border sampling detects white paper vs. dark canvas.
            - **Noise Floor Suppression:** Baseline clipping clears gray camera artifacts.
            - **Dynamic Bounding Box:** Tight crop on foreground stroke geometry.
            - **Lanczos 20×20 Resampling:** Aspect-ratio preserved scaling.
            - **Stroke Calibration:** Adaptive dilation for thin ballpoint pen strokes.
            - **Center of Mass Centering:** Canonical translation to (14.0, 14.0).
            - **Full Contrast Normalization:** Intensity mapped to $[0.0, 1.0]$.
            """
        )

    st.markdown("---")
    st.markdown("#### 📊 Benchmark Specs")
    st.markdown(
        """
        <div style="font-size:13px; line-height:1.7; color:#94a3b8;">
        • <b>Dataset:</b> MNIST (60,000 train / 10,000 test)<br>
        • <b>KNN (k=3):</b> 97.05% Accuracy<br>
        • <b>SVM (RBF, C=10):</b> 97.67% Accuracy<br>
        • <b>SVM + PCA (50):</b> 98.68% Accuracy<br>
        • <b>Evaluation:</b> 10,000 Unseen Test Samples
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── TAB 1: Live Recognition & AI Model Comparison Dashboard ──────────────────
if nav_mode.startswith("🎯 Live Recognition"):

    # Hero Header Banner
    st.markdown(
        """
        <div class="hero-container">
            <span class="hero-badge">AI Pattern Recognition System</span>
            <h1 class="hero-title">Handwritten Digit Recognition Dashboard</h1>
            <p class="hero-desc">
                High-performance comparative pattern recognition on the MNIST benchmark.
                Draw on the digital canvas or upload an image to inspect real-time predictions,
                neighbour vote distribution, decision margins, and dual-model consensus.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Missing model check
    if knn_model is None or svm_model is None:
        st.error(
            f"❌ Models not found in `{MODEL_DIR}`.\n"
            "Please ensure `models/knn_mnist.joblib` and `models/svm_mnist.joblib` exist."
        )
        st.stop()

    # Input Section: Draw vs Upload
    st.markdown("### 📥 Input Digit Sample")
    tab_draw, tab_upload = st.tabs(["🖊️ Interactive Drawing Canvas", "📁 Upload Handwritten Image"])

    raw_image_array = None

    with tab_draw:
        if "canvas_key" not in st.session_state:
            st.session_state["canvas_key"] = 0

        col_draw_canvas, col_draw_help = st.columns([1.1, 1])

        with col_draw_canvas:
            st.markdown("**Draw a digit (0–9) below:**")
            try:
                from streamlit_drawable_canvas import st_canvas  # type: ignore
                canvas_result = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=18,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    return_image_data=True,
                    key=f"canvas_draw_{st.session_state['canvas_key']}",
                )

                col_btn_clear, col_btn_pred = st.columns([1, 1])
                with col_btn_clear:
                    if st.button("🗑️ Clear Canvas", use_container_width=True):
                        st.session_state["canvas_key"] += 1
                        st.rerun()

                if canvas_result is not None:
                    try:
                        img_data = canvas_result.image_data
                        if img_data is not None and np.any(img_data[:, :, :3] > 0):
                            raw_image_array = img_data
                    except (RuntimeError, AttributeError):
                        raw_image_array = None

            except ImportError:
                st.warning("`streamlit-drawable-canvas` is not installed. Please use the Upload tab.")

        with col_draw_help:
            st.markdown(
                """
                <div style="background: rgba(30,41,59,0.4); border-radius:12px; padding:18px; border:1px solid rgba(255,255,255,0.06); height:280px;">
                    <h4 style="margin-top:0; color:#38bdf8;">💡 Guidelines for Optimal Accuracy</h4>
                    <ul style="font-size:14px; color:#cbd5e1; line-height:1.8; padding-left:20px; margin-bottom:0;">
                        <li>Draw a <b>large, centered digit</b> inside the 280×280 box.</li>
                        <li>Use bold, fluid strokes.</li>
                        <li>Write naturally: <i>3, 5, 7, 8, 9</i> are calibrated with center-of-mass alignment.</li>
                        <li>Hit <b>Clear Canvas</b> to reset for the next digit.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_upload:
        col_up_box, col_up_prev = st.columns([1.1, 1])
        with col_up_box:
            st.markdown("**Upload a scanned or photographed handwritten digit:**")
            uploaded_file = st.file_uploader(
                "Accepts PNG, JPG, JPEG, BMP, TIFF",
                type=["png", "jpg", "jpeg", "bmp", "tiff"],
                key="file_uploader",
            )
            if uploaded_file is not None:
                pil_uploaded = Image.open(uploaded_file).convert("RGB")
                raw_image_array = np.array(pil_uploaded)

        with col_up_prev:
            if uploaded_file is not None:
                st.markdown("**Uploaded Image Source:**")
                st.image(uploaded_file, caption=f"Original ({pil_uploaded.size[0]}×{pil_uploaded.size[1]} px)", width=180)
            else:
                st.info("Upload any phone picture, scanned notebook page, or digital sketch containing a single handwritten numeral.")

    # Inference Button
    st.markdown("---")
    col_trigger, col_spacer = st.columns([1, 3])
    with col_trigger:
        run_prediction = st.button("🚀 Analyze & Predict Digit", type="primary", use_container_width=True)

    if run_prediction:
        if raw_image_array is None:
            st.warning("⚠️ No drawing or uploaded image detected. Please draw a digit or upload an image file first.")
        else:
            with st.spinner("Executing MNIST-Calibrated Preprocessing & Dual Inference..."):
                # Preprocess input
                image_vec = preprocess_canvas_image(raw_image_array)  # shape (1, 784)
                preprocessed_28 = image_vec.reshape(28, 28)

                # Run comparative inference
                comparison = compare_models(knn_model, svm_model, image_vec)
                knn_res = comparison["knn"]
                svm_res = comparison["svm"]
                agreement = comparison["agreement"]

            st.markdown("## 🔍 Recognition Results")

            # Consensus or Disagreement Alert Banner
            if agreement:
                st.markdown(
                    f"""
                    <div class="consensus-card">
                        <span style="font-size:32px;">🎯</span>
                        <div>
                            <div style="font-size:18px; font-weight:800; color:#4ade80;">
                                Dual-Model Consensus: Both Models Agreed on Digit {comparison['consensus_digit']}
                            </div>
                            <div style="font-size:14px; color:#cbd5e1;">
                                Both K-Nearest Neighbors (k=3) and Support Vector Machine (RBF, C=10) arrived at the same classification decision.
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="disagreement-card">
                        <span style="font-size:32px;">⚠️</span>
                        <div>
                            <div style="font-size:18px; font-weight:800; color:#f87171;">
                                Model Disagreement Detected: KNN Predicted {knn_res['digit']} vs. SVM Predicted {svm_res['digit']}
                            </div>
                            <div style="font-size:14px; color:#cbd5e1;">
                                KNN relied on nearest neighbour Euclidean distances, while SVM evaluated its maximal-margin hyperplane. Inspect the diagnostic metrics below.
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Three Column Metric Cards: Image Preview | KNN Card | SVM Card
            col_img, col_knn, col_svm = st.columns([1, 1.2, 1.2])

            with col_img:
                st.markdown(
                    """
                    <div class="model-card">
                        <div style="font-weight:700; color:#94a3b8; font-size:14px; margin-bottom:8px;">PREPROCESSED INPUT</div>
                    """,
                    unsafe_allow_html=True,
                )
                fig_p28, ax_p28 = plt.subplots(figsize=(3, 3))
                fig_p28.patch.set_facecolor('#0f172a')
                ax_p28.set_facecolor('#0f172a')
                ax_p28.imshow(preprocessed_28, cmap="magma", vmin=0, vmax=1)
                ax_p28.axis("off")
                plt.tight_layout()
                st.pyplot(fig_p28)
                plt.close(fig_p28)

                st.markdown(
                    f"""
                    <div style="font-size:12px; color:#94a3b8; text-align:center; margin-top:4px;">
                        Dimension: <b>28×28</b> (784 features)<br>
                        Max Intensity: <b>{preprocessed_28.max():.2f}</b> • Active Px: <b>{np.count_nonzero(preprocessed_28 > 0.15)}</b>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_knn:
                st.markdown(
                    f"""
                    <div class="model-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:700; color:#38bdf8; font-size:14px;">K-NEAREST NEIGHBORS (k=3)</span>
                            <span style="background:rgba(56,189,248,0.15); color:#38bdf8; font-size:11px; padding:2px 8px; border-radius:10px; font-weight:700;">LAZY LEARNER</span>
                        </div>
                        <div class="digit-display digit-knn">{knn_res['digit']}</div>
                        <div style="text-align:center; font-size:14px; color:#e2e8f0; margin-bottom:12px;">
                            <b>Vote Share:</b> {knn_res['vote_summary']}
                        </div>
                        <hr style="border:none; border-top:1px solid rgba(255,255,255,0.08); margin:10px 0;">
                        <div style="font-size:12px; color:#94a3b8; line-height:1.6;">
                            <b>Nearest Neighbours:</b> <code>{knn_res['neigh_labels']}</code><br>
                            <b>Euclidean Distances:</b> <code>{knn_res['neigh_dists']}</code><br>
                            <b>Classification Strategy:</b> Majority vote across $k=3$ nearest training vectors.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_svm:
                st.markdown(
                    f"""
                    <div class="model-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:700; color:#c084fc; font-size:14px;">SUPPORT VECTOR MACHINE</span>
                            <span style="background:rgba(192,132,252,0.15); color:#c084fc; font-size:11px; padding:2px 8px; border-radius:10px; font-weight:700;">RBF KERNEL (C=10)</span>
                        </div>
                        <div class="digit-display digit-svm">{svm_res['digit']}</div>
                        <div style="text-align:center; font-size:14px; color:#e2e8f0; margin-bottom:12px;">
                            <b>Decision Margin:</b> {svm_res['winning_margin']:+.2f}
                        </div>
                        <hr style="border:none; border-top:1px solid rgba(255,255,255,0.08); margin:10px 0;">
                        <div style="font-size:12px; color:#94a3b8; line-height:1.6;">
                            <b>Softmax Calibrated Score:</b> <code>{svm_res['confidence']*100:.1f}%</code><br>
                            <b>Margin Metric:</b> Uncalibrated distance to separating hyperplane.<br>
                            <b>Classification Strategy:</b> Maximum-margin decision in RBF feature space.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── NEW COMPONENT: AI MODEL COMPARISON DASHBOARD ──────────────────
            st.markdown("---")
            st.markdown("### 📊 AI Model Comparison Dashboard")
            st.caption("Side-by-side analysis of class-wise scores, confidence metrics, and prediction stability.")

            col_chart, col_table = st.columns([1.6, 1])

            with col_chart:
                st.markdown("**Dual-Model Score Distribution across Digits 0–9**")

                # Generate grouped bar chart
                fig_bar, ax_bar = plt.subplots(figsize=(8, 4.2))
                fig_bar.patch.set_facecolor('#0b1120')
                ax_bar.set_facecolor('#0b1120')

                digits = np.arange(10)
                width = 0.38

                knn_scores = knn_res["scores"]
                svm_scores = svm_res["scores"]

                bars_knn = ax_bar.bar(digits - width/2, knn_scores, width, label="KNN Vote Share (k=3)",
                                      color="#38bdf8", alpha=0.85, edgecolor="#0284c7")
                bars_svm = ax_bar.bar(digits + width/2, svm_scores, width, label="SVM Softmax Score (RBF)",
                                      color="#c084fc", alpha=0.85, edgecolor="#9333ea")

                ax_bar.set_xticks(digits)
                ax_bar.set_xticklabels([f"Digit {i}" for i in digits], fontsize=9, color="#cbd5e1", rotation=25)
                ax_bar.tick_params(colors="#cbd5e1")
                ax_bar.set_ylabel("Normalized Score", fontsize=11, color="#cbd5e1")
                ax_bar.set_ylim(0, 1.15)
                ax_bar.grid(True, linestyle="--", alpha=0.15, color="#ffffff")
                ax_bar.spines['top'].set_visible(False)
                ax_bar.spines['right'].set_visible(False)
                ax_bar.spines['left'].set_color('#334155')
                ax_bar.spines['bottom'].set_color('#334155')

                # Highlight predictions
                for bar in bars_knn:
                    h = bar.get_height()
                    if h > 0.05:
                        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                                    ha="center", va="bottom", fontsize=7.5, color="#38bdf8", fontweight="bold")
                for bar in bars_svm:
                    h = bar.get_height()
                    if h > 0.05:
                        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                                    ha="center", va="bottom", fontsize=7.5, color="#c084fc", fontweight="bold")

                ax_bar.legend(loc="upper right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#f1f5f9", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close(fig_bar)

            with col_table:
                st.markdown("**Detailed Diagnostic Comparison**")

                comp_table_data = [
                    {"Metric": "Predicted Digit", "KNN (k=3)": str(knn_res['digit']), "SVM (RBF)": str(svm_res['digit'])},
                    {"Metric": "Primary Score", "KNN (k=3)": knn_res['vote_summary'], "SVM (RBF)": f"{svm_res['winning_margin']:+.2f} (margin)"},
                    {"Metric": "Calibrated Score", "KNN (k=3)": f"{knn_res['confidence']*100:.1f}%", "SVM (RBF)": f"{svm_res['confidence']*100:.1f}%"},
                    {"Metric": "Runner-Up Digit", "KNN (k=3)": str(np.argsort(knn_scores)[-2]), "SVM (RBF)": str(np.argsort(svm_scores)[-2])},
                    {"Metric": "Model Agreement", "KNN (k=3)": "Yes" if agreement else "No", "SVM (RBF)": "Yes" if agreement else "No"},
                    {"Metric": "Test Benchmark Acc", "KNN (k=3)": "97.05%", "SVM (RBF)": "97.67%"},
                ]
                st.dataframe(pd.DataFrame(comp_table_data), use_container_width=True, hide_index=True)

                st.markdown(
                    """
                    <div style="font-size:12px; color:#94a3b8; background:rgba(15,23,42,0.6); padding:10px; border-radius:8px; border:1px solid rgba(255,255,255,0.06); margin-top:8px;">
                        <b>Note on Confidence Calibration:</b><br>
                        • <b>KNN confidence</b> represents the direct empirical vote fraction among nearest training instances.<br>
                        • <b>SVM confidence</b> represents softmax-normalized decision margins. Raw SVM scores measure geometric margin distance ($w \\cdot x + b$) rather than absolute class probability.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ── TAB 2: Benchmark & IEEE Metrics ──────────────────────────────────────────
elif nav_mode.startswith("📊 Benchmark"):
    st.markdown(
        """
        <div class="hero-container">
            <span class="hero-badge">Academic Project Deliverables</span>
            <h1 class="hero-title">Model Performance & IEEE Paper Metrics</h1>
            <p class="hero-desc">
                Comprehensive evaluation results obtained by executing the pipeline across all 10,000 unseen MNIST test samples.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load results.csv
    results_csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    robust_csv_path = os.path.join(OUTPUT_DIR, "robustness_results.csv")

    if os.path.exists(results_csv_path):
        df_results = pd.read_csv(results_csv_path)
        st.markdown("### 📈 Test Set Summary Metrics (10,000 Samples)")
        st.dataframe(df_results, use_container_width=True, hide_index=True)

    col_plot1, col_plot2 = st.columns(2)
    comp_bar_path = os.path.join(OUTPUT_DIR, "comparison_bar.png")
    robust_bar_path = os.path.join(OUTPUT_DIR, "robustness_comparison.png")

    with col_plot1:
        if os.path.exists(comp_bar_path):
            st.markdown("**Metric Comparison (300 DPI)**")
            st.image(comp_bar_path, use_container_width=True)

    with col_plot2:
        if os.path.exists(robust_bar_path):
            st.markdown("**Robustness Degradation (Noise / Rotation / Shift)**")
            st.image(robust_bar_path, use_container_width=True)

    st.markdown("### 🗂️ Confusion Matrices")
    col_cm1, col_cm2 = st.columns(2)
    cm_knn_path = os.path.join(OUTPUT_DIR, "confusion_knn.png")
    cm_svm_path = os.path.join(OUTPUT_DIR, "confusion_svm.png")

    with col_cm1:
        if os.path.exists(cm_knn_path):
            st.image(cm_knn_path, caption="KNN (k=3) Confusion Matrix Heatmap", use_container_width=True)

    with col_cm2:
        if os.path.exists(cm_svm_path):
            st.image(cm_svm_path, caption="SVM (RBF, C=10) Confusion Matrix Heatmap", use_container_width=True)


# ── TAB 3: Model Architecture & Diagnostics ──────────────────────────────────
elif nav_mode.startswith("🔬 Model Architecture"):
    st.markdown(
        """
        <div class="hero-container">
            <span class="hero-badge">Algorithmic Architecture</span>
            <h1 class="hero-title">Mathematical Formulation & Diagnostic Analysis</h1>
            <p class="hero-desc">
                Technical foundation and structural error analysis comparing distance-based and kernel-margin classifiers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_arch1, col_arch2 = st.columns(2)

    with col_arch1:
        st.markdown(
            """
            ### 📌 K-Nearest Neighbors (KNN)
            - **Hyperparameters:** $k = 3$, Distance Metric: Euclidean ($L_2$)
            - **Mathematical Formulation:**
              $$d(x, y) = \\sqrt{\\sum_{i=1}^{784} (x_i - y_i)^2}$$
            - **Inference Principle:** Stores all 60,000 training images. Computes pairwise distances to the query sample and assigns majority vote among the 3 nearest vectors.
            - **Sensitivity:** Sensitive to vector magnitude and stroke density. Thin or faint strokes have low $L_2$ norms and collapse towards Digit 1 (which has the smallest average mass in MNIST).
            - **Mitigation:** Contrast stretching and adaptive morphological dilation restore standard MNIST stroke width (~3px).
            """
        )

    with col_arch2:
        st.markdown(
            """
            ### ⚡ Support Vector Machine (SVM)
            - **Hyperparameters:** $C = 10$, Kernel: Radial Basis Function (RBF), $\\gamma = \\text{'scale'}$
            - **Mathematical Formulation:**
              $$K(x, y) = \\exp\\left(-\\gamma \\|x - y\\|^2\\right)$$
            - **Inference Principle:** Maps 784-dimensional pixel vectors to an infinite-dimensional Hilbert space, finding optimal separating hyperplanes maximizing the functional margin:
              $$\\min_{w, b, \\xi} \\frac{1}{2} \\|w\\|^2 + C \\sum_{i=1}^n \\xi_i$$
            - **Robustness:** Decision boundaries isolate structural contours rather than raw energy, making SVM less susceptible to minor stroke thickness variations.
            """
        )

    st.markdown("---")
    st.markdown("### 🧩 Error Analysis & Known Ambiguities")
    st.markdown(
        """
        Analysis of the 10,000 test images reveals that misclassifications primarily occur in structurally ambiguous pairs:
        - **3 ↔ 5:** Shared bottom loop curvature and open top stroke.
        - **4 ↔ 9:** Closed upper loop vs. vertical stroke intersection.
        - **7 ↔ 1:** European serif variations and vertical slant angles.
        - **8 ↔ 3:** Open top loop in handwritten 8s mimicking 3.
        """
    )

    col_err1, col_err2 = st.columns(2)
    err_knn_path = os.path.join(OUTPUT_DIR, "errors_knn.png")
    err_svm_path = os.path.join(OUTPUT_DIR, "errors_svm.png")
    with col_err1:
        if os.path.exists(err_knn_path):
            st.image(err_knn_path, caption="KNN Misclassified Samples (True vs Predicted)", use_container_width=True)
    with col_err2:
        if os.path.exists(err_svm_path):
            st.image(err_svm_path, caption="SVM Misclassified Samples (True vs Predicted)", use_container_width=True)
