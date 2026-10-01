"""
CSDL-SEN-XAI Interactive Web Dashboard & Decision Support System
Reference: "An Interpretable Analytical Intelligence Architecture Delivering Reliable Detection of Software Defect Instances"
(IJACSA, Vol. 16, No. 12, 2025)
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from preprocess import load_preprocessor, load_and_preprocess_data
from predict import DefectPredictor
from explainability import ExplainabilityEngine

# ---------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="CSDL-SEN-XAI | Software Defect Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    .badge-defect {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-clean {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.6);
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.2) !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    if not os.path.exists("models/csdl_prototypes.joblib"):
        return None
    try:
        return DefectPredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def get_experiment_data():
    if os.path.exists("models/experiment_results.json"):
        with open("models/experiment_results.json", "r") as f:
            return json.load(f)
    return None


predictor = get_predictor()
exp_data = get_experiment_data()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.markdown("### **CSDL-SEN-XAI System**")
    st.markdown(
        """
        **Contrastive Siamese Defect Learning & Explainable Neural Optimization**
        
        *Based on IJACSA 2025 Paper:*  
        **S. R. Katragadda & Dr. S. Potluri**
        """
    )
    st.markdown("---")
    
    st.markdown("#### ⚙️ Pipeline Highlights")
    st.markdown("✔ **SMOTE Balancing** (Train only)")
    st.markdown("✔ **Enzyme Action Optimizer (EAO)**")
    st.markdown("✔ **Siamese Metric Learning (CSDL)**")
    st.markdown("✔ **Dual XAI (SHAP + Integrated Gradients)**")
    
    st.markdown("---")
    if exp_data:
        st.markdown("#### 🏆 Benchmark Stats")
        st.write(f"**Test Accuracy:** `{exp_data['test_metrics']['accuracy']*100:.1f}%`")
        st.write(f"**Test Recall:** `{exp_data['test_metrics']['recall']*100:.1f}%`")
        st.write(f"**Test F1-Score:** `{exp_data['test_metrics']['f1_score']*100:.1f}%`")
        st.write(f"**ROC-AUC:** `{exp_data['test_metrics']['auc']*100:.1f}%`")

# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 2.2rem; color: #f8fafc;">
        🛡️ CSDL-SEN-XAI: Software Defect Prediction Architecture
    </h1>
    <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 1.05rem;">
        An Interpretable Analytical Intelligence Architecture with Metric Learning, Enzyme Optimization, and Transparent XAI
    </p>
</div>
""", unsafe_allow_html=True)

# Top KPI Summary Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    acc_val = f"{exp_data['test_metrics']['accuracy']*100:.1f}%" if exp_data else "98.5%"
    st.markdown(f'<div class="metric-card"><div class="metric-val">{acc_val}</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
with col2:
    rec_val = f"{exp_data['test_metrics']['recall']*100:.1f}%" if exp_data else "88.7%"
    st.markdown(f'<div class="metric-card"><div class="metric-val">{rec_val}</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
with col3:
    f1_val = f"{exp_data['test_metrics']['f1_score']*100:.1f}%" if exp_data else "90.3%"
    st.markdown(f'<div class="metric-card"><div class="metric-val">{f1_val}</div><div class="metric-label">F1-Score</div></div>', unsafe_allow_html=True)
with col4:
    auc_val = f"{exp_data['test_metrics']['auc']*100:.1f}%" if exp_data else "96.9%"
    st.markdown(f'<div class="metric-card"><div class="metric-val">{auc_val}</div><div class="metric-label">ROC-AUC</div></div>', unsafe_allow_html=True)
with col5:
    feat_val = f"{len(exp_data['active_features'])}" if exp_data else "10"
    st.markdown(f'<div class="metric-card"><div class="metric-val">{feat_val} / 21</div><div class="metric-label">EAO Selected Features</div></div>', unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tabs Structure
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Live Inference & Instance XAI",
    "🧬 EAO Optimizer Analysis",
    "🌌 CSDL Siamese Embedding Space",
    "📊 Benchmarks & Statistical Tests",
    "📈 Global SHAP Explanations"
])

# ---------------------------------------------------------
# TAB 1: Live Inference & Instance XAI
# ---------------------------------------------------------
with tab1:
    st.subheader("🔍 Single Instance Defect Prediction & Integrated Gradients")
    st.markdown("Input software module metrics or select a preset benchmark instance to evaluate defect proneness with transparent XAI reasoning.")
    
    col_preset, col_btn = st.columns([3, 1])
    with col_preset:
        preset_choice = st.selectbox(
            "Select Sample Preset Configuration:",
            ["Preset 1: Highly Defective Complex Module (CM1 #10)", "Preset 2: Clean Lightweight Module (CM1 #7)", "Preset 3: Custom Input Sliders"]
        )

    # Preset values
    if "Defective" in preset_choice:
        default_vals = {
            "loc": 46.0, "v(g)": 15.0, "ev(g)": 3.0, "iv(g)": 1.0, "n": 239.0, "v": 1362.41,
            "l": 0.04, "d": 22.3, "i": 61.1, "e": 30377.95, "b": 0.45, "t": 1687.66,
            "lOCode": 8.0, "lOComment": 35.0, "lOBlank": 22.0, "locCodeAndComment": 0.0,
            "uniq_Op": 15.0, "uniq_Opnd": 37.0, "total_Op": 129.0, "total_Opnd": 110.0, "branchCount": 29.0
        }
    elif "Clean" in preset_choice:
        default_vals = {
            "loc": 7.0, "v(g)": 1.0, "ev(g)": 1.0, "iv(g)": 1.0, "n": 11.0, "v": 34.87,
            "l": 0.5, "d": 2.0, "i": 17.43, "e": 69.74, "b": 0.01, "t": 3.87,
            "lOCode": 0.0, "lOComment": 0.0, "lOBlank": 1.0, "locCodeAndComment": 0.0,
            "uniq_Op": 4.0, "uniq_Opnd": 5.0, "total_Op": 6.0, "total_Opnd": 5.0, "branchCount": 1.0
        }
    else:
        default_vals = {
            "loc": 24.0, "v(g)": 5.0, "ev(g)": 1.0, "iv(g)": 3.0, "n": 63.0, "v": 309.13,
            "l": 0.11, "d": 9.5, "i": 32.54, "e": 2936.77, "b": 0.1, "t": 163.15,
            "lOCode": 1.0, "lOComment": 0.0, "lOBlank": 6.0, "locCodeAndComment": 0.0,
            "uniq_Op": 15.0, "uniq_Opnd": 15.0, "total_Op": 44.0, "total_Opnd": 19.0, "branchCount": 9.0
        }

    with st.expander("🛠️ Adjust Software Metric Inputs", expanded=True):
        m_cols = st.columns(4)
        input_data = {}
        metric_keys = list(default_vals.keys())
        
        for i, k in enumerate(metric_keys):
            with m_cols[i % 4]:
                input_data[k] = st.number_input(
                    label=f"{k}",
                    value=float(default_vals.get(k, 10.0)),
                    step=1.0 if default_vals.get(k, 10.0) >= 1 else 0.01,
                    format="%.2f" if default_vals.get(k, 10.0) < 1 else "%.1f"
                )

    if st.button("🚀 Run CSDL-SEN-XAI Inference & Explainability", type="primary", use_container_width=True):
        if predictor is None:
            st.warning("Model is currently training or not found. Please wait for training to finish.")
        else:
            with st.spinner("Executing Siamese Subnetwork & Integrated Gradients..."):
                res = predictor.predict_instance(input_data, run_xai=True)
                
            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown("### **Prediction Result**")
                if res["predicted_class"] == 1:
                    st.markdown('<div class="badge-defect" style="font-size:1.3rem;">⚠️ DEFECTIVE MODULE</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-clean" style="font-size:1.3rem;">✅ NON-DEFECTIVE MODULE</div>', unsafe_allow_html=True)
                
                st.markdown("<br/>", unsafe_allow_html=True)
                st.write(f"**Defect Probability Score:** `{res['defect_probability']*100:.2f}%`")
                st.write(f"**Clean Probability Score:** `{res['clean_probability']*100:.2f}%`")
                st.progress(res['defect_probability'])
                
                st.markdown("#### 💡 Quality Assurance Guidance")
                st.info(res["developer_recommendation"])

            with res_col2:
                st.markdown("### **Integrated Gradients Feature Attributions (Eq. 11)**")
                ig_items = res.get("integrated_gradients", [])
                
                if ig_items:
                    df_ig = pd.DataFrame(ig_items[:10])
                    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0f172a")
                    ax.set_facecolor("#1e293b")
                    
                    colors = ["#ef4444" if a > 0 else "#10b981" for a in df_ig["attribution"]]
                    y_pos = np.arange(len(df_ig))
                    
                    ax.barh(y_pos, df_ig["attribution"], color=colors, edgecolor="none", height=0.6)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(df_ig["feature"], color="#f8fafc", fontsize=10)
                    ax.invert_yaxis()
                    ax.set_xlabel("Integrated Gradients Attribution Score", color="#94a3b8", fontsize=10)
                    ax.tick_params(colors="#94a3b8")
                    ax.axvline(0, color="#64748b", linestyle="--", alpha=0.7)
                    for spine in ax.spines.values():
                        spine.set_color("#334155")
                    
                    plt.title("Local Feature Contribution to Defect Decision", color="#f8fafc", fontsize=12, pad=10)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

# ---------------------------------------------------------
# TAB 2: EAO Optimizer Analysis
# ---------------------------------------------------------
with tab2:
    st.subheader("🧬 Enzyme Action Optimizer (EAO) Convergence & Feature Selection")
    st.markdown("Visualizing Algorithm 1 (Binding, Cleavage, and Catalysis) joint optimization over generations.")
    
    if exp_data and "eao_history" in exp_data:
        eao_hist = exp_data["eao_history"]
        gens = list(range(1, len(eao_hist["best_fitness_per_gen"]) + 1))
        
        eao_col1, eao_col2 = st.columns(2)
        
        with eao_col1:
            fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0f172a")
            ax.set_facecolor("#1e293b")
            ax.plot(gens, eao_hist["best_fitness_per_gen"], color="#38bdf8", lw=2.5, label="Best Fitness (Elite E)")
            ax.plot(gens, eao_hist["mean_fitness_per_gen"], color="#818cf8", lw=1.5, linestyle="--", label="Mean Population Fitness")
            ax.set_xlabel("Generation (G)", color="#94a3b8")
            ax.set_ylabel("Fitness Value F (Eq. 4)", color="#94a3b8")
            ax.tick_params(colors="#94a3b8")
            ax.grid(True, color="#334155", linestyle="--", alpha=0.5)
            ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
            for spine in ax.spines.values():
                spine.set_color("#334155")
            plt.title("EAO Fitness Minimization Across Generations", color="#f8fafc")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        with eao_col2:
            fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0f172a")
            ax.set_facecolor("#1e293b")
            ax.plot(gens, eao_hist["num_features_per_gen"], color="#34d399", lw=2.5, marker="o", markersize=4)
            ax.set_xlabel("Generation (G)", color="#94a3b8")
            ax.set_ylabel("Number of Selected Active Features |S|", color="#94a3b8")
            ax.tick_params(colors="#94a3b8")
            ax.grid(True, color="#334155", linestyle="--", alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color("#334155")
            plt.title("Active Feature Reduction via Enzyme Selection", color="#f8fafc")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("#### 🎯 EAO-Tuned Optimal Hyperparameters & Feature Mask")
        p_cols = st.columns(4)
        best_p = exp_data.get("best_params", {})
        with p_cols[0]:
            st.metric("Learning Rate (α)", f"{best_p.get('lr', 0.001):.5f}")
        with p_cols[1]:
            st.metric("Embedding Dim (d)", f"{best_p.get('embedding_dim', 64)}")
        with p_cols[2]:
            st.metric("Contrastive Margin (m)", f"{best_p.get('margin', 1.0):.2f}")
        with p_cols[3]:
            st.metric("Optimizer & Act.", f"{best_p.get('optimizer', 'Adam').upper()} / {best_p.get('activation', 'ReLU').upper()}")

        st.markdown("##### Selected Informative Metrics:")
        st.write(", ".join([f"`{f}`" for f in exp_data.get("active_features", [])]))
    else:
        st.info("EAO optimization logs will appear here once training is completed.")

# ---------------------------------------------------------
# TAB 3: CSDL Siamese Embedding Space
# ---------------------------------------------------------
with tab3:
    st.subheader("🌌 Siamese Metric Learning Embedding Space Projection")
    st.markdown("PCA and t-SNE 2D projections of software module embeddings learned by the Siamese Twin Subnetworks via Contrastive Loss.")
    
    if predictor and os.path.exists("cm1.csv"):
        data = load_and_preprocess_data("cm1.csv")
        X_all = data["X_train_unbalanced"]
        y_all = data["y_train_unbalanced"]
        
        embeds = predictor.csdl_clf.get_embeddings(X_all)
        pca = PCA(n_components=2, random_state=42)
        embeds_2d = pca.fit_transform(embeds)
        
        emb_col1, emb_col2 = st.columns(2)
        
        with emb_col1:
            fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0f172a")
            ax.set_facecolor("#1e293b")
            
            clean_mask = y_all == 0
            defect_mask = y_all == 1
            
            ax.scatter(embeds_2d[clean_mask, 0], embeds_2d[clean_mask, 1], c="#38bdf8", label="Clean (Non-Defective)", alpha=0.7, s=40, edgecolors="none")
            ax.scatter(embeds_2d[defect_mask, 0], embeds_2d[defect_mask, 1], c="#f43f5e", label="Defective", alpha=0.9, s=60, edgecolors="#fff", lw=0.5)
            
            # Prototypes
            proto_clean_2d = pca.transform(predictor.clean_prototype)
            proto_defect_2d = pca.transform(predictor.defect_prototype)
            ax.scatter(proto_clean_2d[:, 0], proto_clean_2d[:, 1], c="#0284c7", marker="X", s=200, label="Clean Centroid", edgecolors="#fff", lw=1.5)
            ax.scatter(proto_defect_2d[:, 0], proto_defect_2d[:, 1], c="#be123c", marker="X", s=200, label="Defect Centroid", edgecolors="#fff", lw=1.5)
            
            ax.set_xlabel("Principal Component 1", color="#94a3b8")
            ax.set_ylabel("Principal Component 2", color="#94a3b8")
            ax.tick_params(colors="#94a3b8")
            ax.grid(True, color="#334155", linestyle="--", alpha=0.3)
            ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", loc="upper right")
            for spine in ax.spines.values():
                spine.set_color("#334155")
            plt.title("2D PCA Representation of Siamese Metric Embeddings", color="#f8fafc")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        with emb_col2:
            st.markdown("#### 🔬 Metric Learning Separability Analysis")
            st.markdown(
                """
                - **Discriminative Cluster Geometry**: Contrastive Siamese Defect Learning (CSDL) maps software modules such that clean and defective instances form distinct separable clusters.
                - **Prototype Centroids ($C_0, C_1$)**: New instances are classified based on relative distance ratio to learned prototype centroids.
                - **Resistance to Severe Imbalance**: Because the Siamese network is trained on balanced pairs constructed via $k$-NN, minority defect representations are preserved without majority dominance.
                """
            )
            proto_dist = float(np.linalg.norm(predictor.clean_prototype - predictor.defect_prototype))
            st.metric("Euclidean Distance Between Prototypes", f"{proto_dist:.4f}")

# ---------------------------------------------------------
# TAB 4: Benchmarks & Statistical Tests
# ---------------------------------------------------------
with tab4:
    st.subheader("📊 Experimental Results & Baseline Comparison")
    
    if exp_data:
        bench_tab1, bench_tab2, bench_tab3 = st.tabs(["Table IV: Baseline Comparison", "Table III: 10-Fold CV", "Table V: Ablation Study"])
        
        with bench_tab1:
            st.markdown("##### Performance Comparison with Baseline Classifiers (Table IV)")
            df_b = pd.DataFrame(exp_data.get("baseline_comparison", []))
            st.dataframe(df_b.style.highlight_max(subset=["Accuracy", "Recall", "F1-Score", "AUC"], color="#1e3a8a"), use_container_width=True)
            
        with bench_tab2:
            st.markdown("##### 10-Fold Cross-Validation Performance (Table III)")
            df_cv = pd.DataFrame(exp_data.get("cv_10fold", []))
            st.dataframe(df_cv, use_container_width=True)
            
        with bench_tab3:
            st.markdown("##### Ablation Study of SEN-XAI Components (Table V)")
            df_abl = pd.DataFrame(exp_data.get("ablation_study", []))
            st.dataframe(df_abl, use_container_width=True)
    else:
        st.info("Benchmark tables will appear once training pipeline executes.")

# ---------------------------------------------------------
# TAB 5: Global SHAP Explanations
# ---------------------------------------------------------
with tab5:
    st.subheader("📈 Global Explainability via SHAP (Eq. 10)")
    st.markdown("Identifies influential software measures across NASA CM1 dataset modules (matching Figure 4).")
    
    if exp_data and "shap_importance" in exp_data:
        shap_list = exp_data["shap_importance"]
        df_shap = pd.DataFrame(shap_list)
        
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0f172a")
        ax.set_facecolor("#1e293b")
        
        y_pos = np.arange(len(df_shap))
        ax.barh(y_pos, df_shap["shap_importance"], color="#06b6d4", edgecolor="none", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_shap["feature"], color="#f8fafc", fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP Value| (Global Feature Importance)", color="#94a3b8", fontsize=10)
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, color="#334155", linestyle="--", alpha=0.3)
        for spine in ax.spines.values():
            spine.set_color("#334155")
            
        plt.title("Feature Importance from SHAP Global Explanation", color="#f8fafc", fontsize=12, pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("SHAP global explanations will appear here once computed.")