#!/usr/bin/env python3
"""
Fraud dashboard (Dash + Gradio) — CSS fixed

This file is a drop-in replacement for your script with improved layout and
separate CSS written into assets/style.css at startup. The script will create
an `assets` folder and write the stylesheet if it doesn't exist, so you don't
need to manage extra files.

Main visual improvements:
- Uses semantic classNames and a responsive flex/grid layout for the upload
  area and control buttons so they don't overlap.
- Tables are placed inside a scrollable container to avoid overflow.
- Images are responsive and constrained to container width.
- Small visual polish on buttons and upload box.

To run: python fraud_dashboard_with_fixed_css.py
"""

import os
import io
import base64
import threading
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optionals
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

import joblib
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from flask import send_file

import gradio as gr

# ---------- Paths & config ----------
MODEL_PATH = "models/member2_model.pkl"
STATIC_DIR = "static_outputs"
ASSETS_DIR = "assets"
Path("models").mkdir(exist_ok=True)
Path(STATIC_DIR).mkdir(exist_ok=True)
Path(ASSETS_DIR).mkdir(exist_ok=True)

# Write a friendly stylesheet into assets/style.css (idempotent)
CSS_PATH = os.path.join(ASSETS_DIR, "style.css")
CSS_CONTENT = """
/* Responsive layout for the fraud dashboard */
:root{
  --gap: 12px;
  --accent: #2b7cff;
  --muted: #666;
  --card-bg: #ffffff;
  --page-bg: #f5f7fb;
}
html,body{height:100%;margin:0;font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial; background:var(--page-bg);}

.app-container{max-width:1200px;margin:24px auto;padding:18px;box-sizing:border-box;}
.header{display:flex;align-items:center;gap:12px;flex-wrap:wrap;}
.header h1{margin:0;font-size:20px}
.top-row{display:flex;gap:var(--gap);align-items:center;flex-wrap:wrap;margin-top:12px}
.upload-area{flex:1 1 320px;min-width:260px;background:var(--card-bg);padding:10px;border-radius:8px;border:1px dashed #cfd8e3;text-align:center}
.controls{display:flex;gap:8px;align-items:center;flex:0 0 auto}
.controls .btn{padding:8px 12px;border-radius:6px;border:none;background:var(--accent);color:white;cursor:pointer}
.controls .btn.secondary{background:#556;opacity:0.95}

.status{margin-top:8px;color:var(--muted)}

.section{background:var(--card-bg);padding:12px;border-radius:8px;margin-top:14px}
.table-container {
  max-height: 360px;
  max-width: 800px;     /* ✅ limit width */
  overflow: auto;
  border-radius: 6px;
  padding: 6px;
  border: 1px solid #e6e9ef;
  background: #fff;
  margin: 0 auto;       /* ✅ center on page */
  box-shadow: 0 2px 6px rgba(0,0,0,0.05); /* optional subtle shadow */
}


img.responsive{max-width:100%;height:auto;display:block;margin:8px 0}

.input-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.input-row .input{flex:0 0 120px}

/* Small screens tweaks */
@media (max-width:640px){
  .controls{width:100%;justify-content:space-between}
  .header h1{font-size:18px}
}
"""

if not os.path.exists(CSS_PATH):
    with open(CSS_PATH, "w") as f:
        f.write(CSS_CONTENT)

# ---------- Model loading / training ----------
def load_or_train_model(model_path: str = MODEL_PATH) -> Tuple[object, list]:
    # Try to load a model and .meta for feature names
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        meta_path = model_path + ".meta"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                cols = [line.strip() for line in f if line.strip()]
        else:
            cols = None
        return model, cols

    # If no model provided, train a dummy RF on synthetic data (for local testing)
    rng = np.random.default_rng(42)
    n = 2000
    X = pd.DataFrame({
        "amount": rng.lognormal(mean=3, sigma=1.0, size=n),
        "age": rng.integers(18, 80, size=n),
        "num_txn": rng.integers(1, 20, size=n),
        "is_foreign": rng.integers(0, 2, size=n),
    })
    y = ((X["amount"] > X["amount"].quantile(0.9)) & (X["is_foreign"] == 1)).astype(int)

    model = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
    model.fit(X, y)

    sample_path = "models/sample_model.pkl"
    joblib.dump(model, sample_path)
    with open(sample_path + ".meta", "w") as f:
        for c in X.columns:
            f.write(c + "\n")
    return model, list(X.columns)

MODEL, MODEL_FEATURES = load_or_train_model()
if MODEL_FEATURES is None:
    # fallback to numeric auto-detect at runtime
    MODEL_FEATURES = None

# ---------- SHAP helper (safe for numpy/shap mismatch) ----------
def safe_shap_explainer(model, X_sample):
    # older shap expects np.bool; add attribute if missing
    if not hasattr(np, "bool"):
        setattr(np, "bool", bool)
    import shap as _shap
    return _shap.Explainer(model, X_sample)

# ---------- Predict function ----------
def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Determine features
    features = MODEL_FEATURES
    if features and all(isinstance(c, str) for c in features):
        features = [c for c in features if c in df.columns]
    else:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    if not features:
        raise ValueError("No valid numeric feature columns found for prediction.")

    X = df[features].fillna(0)
    try:
        probs = MODEL.predict_proba(X)[:, 1]
    except Exception:
        try:
            probs = MODEL.predict(X).astype(float)
        except Exception:
            probs = np.zeros(len(X))

    out = df.copy()
    out["fraud_probability"] = probs
    out["predicted_label"] = (probs >= 0.5).astype(int)
    return out

# ---------- SHAP / importance plotting ----------
def shap_global_plot(df: pd.DataFrame, save_path: str) -> str:
    features = MODEL_FEATURES
    if all(isinstance(c, str) for c in (features or [])):
        features = [c for c in features if c in df.columns]
    else:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(features) == 0:
        # nothing to plot
        fig, ax = plt.subplots(figsize=(6,3))
        ax.text(0.5, 0.5, 'No numeric features to plot', ha='center', va='center')
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    X = df[features].fillna(0)
    # attempt SHAP
    if SHAP_AVAILABLE and len(features) > 0:
        try:
            expl = safe_shap_explainer(MODEL, X.sample(min(100, len(X)), random_state=1))
            sv = expl(X)
            plt.figure(figsize=(8,6))
            import shap as _shap
            _shap.summary_plot(sv, X, show=False)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            return save_path
        except Exception:
            pass

    # fallback to sklearn feature importances or correlation
    plt.figure(figsize=(8,6))
    if hasattr(MODEL, "feature_importances_"):
        imp = pd.Series(MODEL.feature_importances_, index=features).sort_values()
        imp.plot(kind="barh")
        plt.title("Model Feature Importances (fallback)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        corr = np.abs(df[features].corrwith(df.get("target", df.iloc[:,0])).fillna(0)).sort_values()
        corr.plot(kind="barh")
        plt.title("Feature correlation with target (fallback)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    return save_path


def shap_local_plot(single_row: pd.DataFrame, save_path: str) -> str:
    # single_row is a 1-row DataFrame
    features = MODEL_FEATURES
    if all(isinstance(c, str) for c in (features or [])):
        features = [c for c in features if c in single_row.columns]
    else:
        features = single_row.select_dtypes(include=[np.number]).columns.tolist()

    if len(features) == 0:
        fig, ax = plt.subplots(figsize=(4,3))
        ax.text(0.5,0.5,'No numeric features to plot',ha='center',va='center')
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    X = single_row[features].fillna(0)
    if SHAP_AVAILABLE and len(features) > 0:
        try:
            if not hasattr(np, "bool"):
                setattr(np, "bool", bool)
            import shap as _shap
            expl = _shap.Explainer(MODEL, X)
            sv = expl(X)
            vals = sv.values[0]
            if vals.ndim == 2:
                if vals.shape[1] > 1:
                    vals = vals[:,1]
                else:
                    vals = vals[:,0]
            vals = pd.Series(np.abs(vals), index=X.columns).sort_values(ascending=False)[:10]
            plt.figure(figsize=(6,4))
            vals.plot(kind='barh')
            plt.title("Top SHAP Features (local)")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            return save_path
        except Exception:
            pass

    # fallback to model importances or values
    if hasattr(MODEL, "feature_importances_"):
        imp = pd.Series(MODEL.feature_importances_, index=features).sort_values(ascending=False)[:10]
        plt.figure(figsize=(6,4))
        imp.plot(kind='barh')
        plt.title("Top feature importances (fallback)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path
    else:
        vals = pd.Series(np.abs(X.iloc[0]).values, index=X.columns).sort_values(ascending=False)[:10]
        plt.figure(figsize=(6,4))
        vals.plot(kind='barh')
        plt.title("Top features (fallback)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path

# ---------- Dash app layout & callbacks ----------
app = dash.Dash(__name__, assets_folder=ASSETS_DIR)
server = app.server

app.layout = html.Div(className='app-container', children=[
    html.Div(className='header', children=[
        html.H1("Fraud Detection Dashboard (Dash + Gradio)"),
        html.Div(className='status', id='upload-status')
    ]),

    html.Div(className='top-row', children=[
        html.Div(className='upload-area', children=[
            dcc.Upload(id='upload-data',
                       children=html.Button('Upload CSV', className='btn'),
                       multiple=False),
            html.Div("Only CSV files supported", style={'fontSize':'12px','color':'#777','marginTop':'6px'})
        ]),

        html.Div(className='controls', children=[
            html.Button('Run Prediction', id='run-pred', n_clicks=0, className='btn'),
            html.Button('Download Processed CSV', id='download-btn', n_clicks=0, className='btn secondary'),
        ])
    ]),

    html.Div(className='section', children=[
        html.H3('Results'),
        html.Div(className='table-container', children=[
            dash_table.DataTable(
    id='results-table',
    page_size=10,
    style_table={
        "overflowX": "auto",   # ✅ horizontal scroll if needed
        "overflowY": "auto",   # ✅ vertical scroll if needed
        "maxHeight": "400px",  # ✅ respect box height
        "maxWidth": "1000px",   # ✅ respect box width
    },
    style_cell={
        'textAlign':'center',
        'padding':'4px',
        'fontSize':'16px'
    },
)


        ])
    ]),

    html.Div(className='section', children=[
        html.H3('Global SHAP / Importances'),
        html.Div(id='global-shap-container')
    ]),

    html.Div(className='section', children=[
        html.H3('Local SHAP (select row index)'),
        html.Div(className='input-row', children=[
            dcc.Input(id='local-row-index', type='number', placeholder='Row index (0-based)', className='input'),
            html.Button('Show Local SHAP', id='show-local', n_clicks=0, className='btn')
        ]),
        html.Div(id='local-shap-container')
    ]),

    dcc.Store(id='stored-df')
])

# Helper to parse CSV from dcc.Upload
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if filename.lower().endswith('.csv'):
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    else:
        raise ValueError("Only CSV supported.")

@app.callback(Output('upload-status','children'), Output('stored-df','data'),
              Input('upload-data','contents'), State('upload-data','filename'))
def handle_upload(contents, filename):
    if contents is None:
        return 'No file uploaded yet.', None
    try:
        df = parse_contents(contents, filename)
        tmp = os.path.join(STATIC_DIR, 'last_uploaded.csv')
        df.to_csv(tmp, index=False)
        return f'Uploaded {filename} with {len(df)} rows.', df.to_json(date_format='iso', orient='split')
    except Exception as e:
        return f'Upload failed: {e}', None

@app.callback(Output('results-table','data'), Output('results-table','columns'),
              Input('run-pred','n_clicks'), State('stored-df','data'))
def run_prediction(n_clicks, stored_json):
    if n_clicks == 0 or stored_json is None:
        return [], []
    df = pd.read_json(stored_json, orient='split')
    res = predict_dataframe(df)
    outp = os.path.join(STATIC_DIR, 'processed.csv')
    res.to_csv(outp, index=False)
    cols = [{'name':c,'id':c} for c in res.columns]
    return res.to_dict('records'), cols

@app.callback(Output('global-shap-container','children'),
              Input('run-pred','n_clicks'), State('stored-df','data'))
def update_global_shap(n_clicks, stored_json):
    if n_clicks == 0 or stored_json is None:
        return html.Div('No SHAP yet. Run Prediction.')
    df = pd.read_json(stored_json, orient='split')
    out_img = os.path.join(STATIC_DIR, 'global_shap.png')
    shap_global_plot(df, out_img)
    encoded = base64.b64encode(open(out_img,'rb').read()).decode()
    src = f'data:image/png;base64,{encoded}'
    return html.Img(src=src, className='responsive')

@app.callback(Output('local-shap-container','children'),
              Input('show-local','n_clicks'),
              State('stored-df','data'), State('local-row-index','value'))
def update_local_shap(n_clicks, stored_json, idx):
    if n_clicks == 0 or stored_json is None or idx is None:
        return html.Div('No local SHAP yet.')
    df = pd.read_json(stored_json, orient='split')
    if idx < 0 or idx >= len(df):
        return html.Div('Index out of range.')
    row = df.iloc[[idx]]
    out_img = os.path.join(STATIC_DIR, f'local_shap_{idx}.png')
    shap_local_plot(row, out_img)
    encoded = base64.b64encode(open(out_img,'rb').read()).decode()
    return html.Img(src=f'data:image/png;base64,{encoded}', className='responsive')

@server.route('/download/processed')
def download_processed():
    path = os.path.join(STATIC_DIR, 'processed.csv')
    if os.path.exists(path):
        return send_file(path, mimetype='text/csv', as_attachment=True, download_name='processed.csv')
    return 'No processed file. Run prediction first.', 404

@app.callback(Output('download-btn','children'), Input('download-btn','n_clicks'))
def prepare_download(n):
    if n > 0:
        return html.A('Click here to download processed CSV', href='/download/processed')
    return 'Download Processed CSV'

# ---------- Gradio mini-interface ----------
def gradio_predict_and_shap(text_input: str):
    values = [v.strip() for v in text_input.split(',') if v.strip()]
    if len(values) == 0:
        return "No input", None
    if '=' in values[0]:
        d = {}
        for kv in values:
            if '=' in kv:
                k,v = kv.split('=',1)
                try:
                    d[k.strip()] = float(v)
                except Exception:
                    d[k.strip()] = v
        row = pd.DataFrame([d])
    else:
        if MODEL_FEATURES and all(isinstance(c,str) for c in MODEL_FEATURES):
            cols = MODEL_FEATURES
        else:
            cols = [f'col{i}' for i in range(len(values))]
        d = {}
        for i,v in enumerate(values):
            if i < len(cols):
                try:
                    d[cols[i]] = float(v)
                except Exception:
                    d[cols[i]] = v
        row = pd.DataFrame([d])
    try:
        pred_df = predict_dataframe(row)
        prob = pred_df['fraud_probability'].iloc[0]
        label = pred_df['predicted_label'].iloc[0]
        pred_text = f"Probability={prob:.4f}, Label={label}"
        tmp = os.path.join(STATIC_DIR, 'gr_local_shap.png')
        shap_local_plot(row, tmp)
        return pred_text, tmp
    except Exception as e:
        return f"Prediction failed: {e}", None


def run_gradio_app():
    demo = gr.Interface(fn=gradio_predict_and_shap,
                        inputs=gr.Textbox(lines=2, placeholder='e.g. amount=1234,is_foreign=1,num_txn=3'),
                        outputs=["text", gr.Image(type="filepath")],
                        title="Gradio Mini Demo: Single Input -> Prediction + SHAP")
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)

# ---------- Unit tests ----------
def _test_predict_dataframe_local():
    df_test = pd.DataFrame({c: [0,1] for c in (MODEL_FEATURES or ['a','b'])})
    res = predict_dataframe(df_test)
    assert 'fraud_probability' in res.columns and 'predicted_label' in res.columns


def _test_shap_plot_functions():
    df_for_test = pd.DataFrame({c: [0,1] for c in (MODEL_FEATURES or ['a','b'])})
    outg = os.path.join(STATIC_DIR, 'test_global.png')
    outl = os.path.join(STATIC_DIR, 'test_local.png')
    shap_global_plot(df_for_test, outg)
    shap_local_plot(df_for_test.iloc[[0]], outl)
    assert os.path.exists(outg) and os.path.exists(outl)


def run_unit_tests():
    print("Running unit tests...")
    _test_predict_dataframe_local()
    _test_shap_plot_functions()
    print("Unit tests passed.")

# ---------- Main (start servers) ----------
if __name__ == "__main__":
    # Run unit tests first
    try:
        run_unit_tests()
    except AssertionError as e:
        print("Unit tests failed:", e)

    # Start Dash and Gradio concurrently
    t_dash = threading.Thread(target=lambda: app.run(debug=False, use_reloader=False))
    t_gr = threading.Thread(target=run_gradio_app)

    t_dash.start()
    t_gr.start()

    t_dash.join()
    t_gr.join()
