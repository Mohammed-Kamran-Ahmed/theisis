"""
Stage 7 — Automated Visual Findings & Correlation Dashboard Generator (Fixed)
"""

import os
import webbrowser
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE

EMBEDDINGS_CSV = "./patch_embeddings_stage5.csv"
METADATA_CSV = "./patch_metadata_clustered.csv"
REPORT_TXT = "./stage6_evaluation_results.txt"
OUTPUT_HTML = "./stage7_visual_report.html"

def generate_dashboard():
    print("🚀 Generating Visual Correlation Dashboard...")
    
    if not os.path.exists(EMBEDDINGS_CSV):
        raise FileNotFoundError(f"❌ Missing {EMBEDDINGS_CSV}")

    df_emb = pd.read_csv(EMBEDDINGS_CSV)
    
    # Load physical band metadata and merge
    if os.path.exists(METADATA_CSV):
        df_meta = pd.read_csv(METADATA_CSV)
        if "valid" in df_meta.columns:
            df_meta = df_meta[df_meta["valid"] == True].reset_index(drop=True)
        
        # Merge physical bands if they exist in metadata
        band_cols = [c for c in ["VV", "VH", "NDVI", "NDWI", "EVI", "RVI"] if c in df_meta.columns]
        for col in band_cols:
            if col not in df_emb.columns:
                df_emb[col] = df_meta[col].values

    label_col = "gmm_cluster" if "gmm_cluster" in df_emb.columns else "cluster"
    df_emb[label_col] = df_emb[label_col].astype(str)

    # 1. t-SNE
    feature_cols = [c for c in df_emb.columns if c.startswith("z_") or c.startswith("emb_") or c.startswith("dim_")]
    if not feature_cols:
        exclude_cols = ["patch_id", "file_path", "filename", "tile_id", "valid", "gmm_cluster", "cluster", "patch_name", label_col, "VV", "VH", "NDVI", "NDWI", "EVI", "RVI"]
        potential_cols = [c for c in df_emb.columns if c not in exclude_cols]
        feature_cols = df_emb[potential_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = df_emb[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X)
    
    df_emb["tSNE_1"] = X_2d[:, 0]
    df_emb["tSNE_2"] = X_2d[:, 1]

    hover_cols = [c for c in ["patch_id", "tile_id", "filename", label_col] if c in df_emb.columns]

    fig_tsne = px.scatter(
        df_emb,
        x="tSNE_1",
        y="tSNE_2",
        color=label_col,
        symbol=label_col,
        hover_data=hover_cols,
        title="<b>Interactive 2D t-SNE Latent Space Topology</b>",
        labels={label_col: "GMM Cluster"},
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_tsne.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color='DarkSlateGrey')))

    # 2. FEATURE CORRELATION MATRIX
    physical_cols = [c for c in ["VV", "VH", "NDVI", "NDWI", "EVI", "RVI"] if c in df_emb.columns]
    
    if physical_cols:
        corr_data = df_emb[physical_cols + [label_col]].copy()
        corr_data[label_col] = corr_data[label_col].astype(int)
        corr_matrix = corr_data.corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="<b>Physical Feature & Cluster Correlation Matrix</b>",
            template="plotly_white"
        )
    else:
        # Fallback to correlating top 10 embedding dimensions against cluster ID
        emb_subset = feature_cols[:10]
        corr_data = df_emb[emb_subset + [label_col]].copy()
        corr_data[label_col] = corr_data[label_col].astype(int)
        corr_matrix = corr_data.corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="<b>Latent Embedding Projections vs. Cluster Correlation Matrix</b>",
            template="plotly_white"
        )

    # 3. METRICS REPORT
    summary_text = "Stage 6 Report file not found."
    if os.path.exists(REPORT_TXT):
        with open(REPORT_TXT, "r", encoding="utf-8") as f:
            summary_text = f.read()

    tsne_html = fig_tsne.to_html(full_html=False, include_plotlyjs='cdn')
    corr_html = fig_corr.to_html(full_html=False, include_plotlyjs=False)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Thesis Multi-Modal Representation Learning Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background-color: #f8f9fa; color: #212529; }}
            .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
            h1 {{ color: #1a365d; border-bottom: 3px solid #2b6cb0; padding-bottom: 10px; }}
            h2 {{ color: #2c5282; margin-top: 30px; }}
            .card {{ background: #edf2f7; padding: 20px; border-radius: 8px; margin-bottom: 20px; font-family: monospace; white-space: pre-wrap; font-size: 14px; overflow-x: auto; }}
            .graph-box {{ margin-bottom: 40px; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; background: #ffffff; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Thesis Dual-Encoder Multi-Modal Representation Findings</h1>
            <p><b>Dataset:</b> 172 Sentinel-1/Sentinel-2 Patches | <b>Latent Projection:</b> 64-dim L2-Normalized Vectors</p>
            
            <h2>1. Latent Manifold Topology (Interactive t-SNE)</h2>
            <div class="graph-box">{tsne_html}</div>
            
            <h2>2. Physical Band & Cluster Correlation</h2>
            <div class="graph-box">{corr_html}</div>
            
            <h2>3. Stage 6 Evaluation & Retrieval Benchmark Output</h2>
            <div class="card">{summary_text}</div>
        </div>
    </body>
    </html>
    """

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Visual report generated successfully: {OUTPUT_HTML}")
    webbrowser.open("file://" + os.path.abspath(OUTPUT_HTML))

if __name__ == "__main__":
    generate_dashboard()