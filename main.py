import os, webbrowser, threading
from pathlib import Path


def open_browser():
    webbrowser.open("http://127.0.0.1:8501")

if __name__ == "__main__":
    # Streamlit launches the dashboard; opening the browser here makes main.py the single entry point.
    import sys
    if os.environ.get("ANOMALY_IDS_STREAMLIT") != "1":
        import subprocess
        env = os.environ.copy()
        env["ANOMALY_IDS_STREAMLIT"] = "1"
        threading.Timer(2.0, open_browser).start()
        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), "--server.headless", "true", "--browser.gatherUsageStats", "false"], env=env))

import io
import time
import json
import tempfile
import traceback
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.units import inch

st.set_page_config(page_title="Anomaly-Based Cyber Attack Detection System", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

APP = "Anomaly-Based Cyber Attack Detection System"

# ----------------------------- Utilities -----------------------------
NSL_KDD_COLUMNS_41 = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]
NSL_KDD_COLUMNS_42 = NSL_KDD_COLUMNS_41 + ["label"]
NSL_KDD_COLUMNS_43 = NSL_KDD_COLUMNS_41 + ["label", "difficulty"]

def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df

def _looks_like_headerless_nsl_kdd(df):
    """Detect the standard headerless NSL-KDD/KDDTrain+/KDDTest+ layout."""
    if df.shape[1] not in (42, 43):
        return False
    if len(df) == 0:
        return False
    first = df.iloc[0].astype(str).str.strip().str.lower().tolist()
    if len(first) < 4:
        return False
    # Standard NSL-KDD starts with numeric duration, protocol, service, flag.
    protocols = {"tcp", "udp", "icmp"}
    flags = {"sf", "s0", "rej", "rstr", "rstos0", "s1", "s2", "s3", "sh", "oth"}
    try:
        float(first[0])
    except Exception:
        return False
    return first[1] in protocols and first[3] in flags

def load_uploaded(file):
    name = file.name.lower()
    raw = file.getvalue()
    if not name.endswith((".csv", ".txt")):
        raise ValueError("Unable to read this file. Please provide a supported CSV/TXT dataset.")

    # NSL-KDD files are commonly supplied as headerless comma-separated TXT files.
    # Reading the first row as a header is the reason those files were previously
    # classified as Generic CSV and could lose the test label column.
    try:
        probe = pd.read_csv(io.BytesIO(raw), header=None, nrows=3, sep=",", engine="python")
        if _looks_like_headerless_nsl_kdd(probe):
            if probe.shape[1] == 43:
                cols = NSL_KDD_COLUMNS_43
            else:
                cols = NSL_KDD_COLUMNS_42
            df = pd.read_csv(io.BytesIO(raw), header=None, names=cols, sep=",", engine="python")
            # Difficulty is a dataset metadata field, not a predictive feature.
            if "difficulty" in df.columns:
                df = df.drop(columns=["difficulty"])
            return clean_columns(df)
    except Exception:
        pass

    # Normal CSV/TXT files with a header.
    for kwargs in [dict(sep=None, engine="python"), dict()]:
        try:
            return clean_columns(pd.read_csv(io.BytesIO(raw), **kwargs))
        except Exception:
            pass
    raise ValueError("Unable to read this file as CSV/TXT. Please provide a supported tabular dataset.")

def detect_label(df):
    candidates = ["label", "attack", "attack_type", "class", "target", "category", "y", "Label", "Class"]
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    # NSL-KDD often has the label in the last column.
    if len(df.columns) > 1:
        last = df.columns[-1]
        if df[last].nunique(dropna=True) <= max(50, int(len(df) * 0.05)):
            return last
    return None

def infer_dataset(df):
    cols = {str(c).lower() for c in df.columns}
    if {"protocol_type", "service", "flag"}.issubset(cols):
        return "NSL-KDD / KDD-style"
    # CICIDS commonly contains flow-oriented fields.
    cic_tokens = {"flow_duration", "tot_fwd_pkts", "tot_bwd_pkts", "fwd_pkt_len_max", "bwd_pkt_len_max"}
    if len(cols.intersection(cic_tokens)) >= 2:
        return "CICIDS-style"
    return "Generic CSV"

def profile(df):
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric": int(df.select_dtypes(include=np.number).shape[1]),
        "categorical": int(df.select_dtypes(exclude=np.number).shape[1]),
    }

def normalize_label(x):
    s = str(x).strip()
    # NSL-KDD has labels such as normal, neptune, smurf, etc.; map to attack families.
    nsl = {
        "normal": "Normal",
        "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
        "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS", "mailbomb": "DoS", "worm": "DoS",
        "ipsweep": "Probing", "nmap": "Probing", "portsweep": "Probing", "satan": "Probing", "mscan": "Probing", "saint": "Probing",
        "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
        "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
    }
    key = s.lower().rstrip('.')
    if key in nsl:
        return nsl[key]
    # CICIDS and generic datasets frequently contain explicit attack family words.
    if "normal" in key or key in {"benign", "0", "false"}:
        return "Normal"
    for token, family in [("dos", "DoS"), ("ddos", "DoS"), ("probe", "Probing"), ("scan", "Probing"), ("brute", "R2L"), ("infil", "R2L"), ("bot", "R2L"), ("web", "R2L"), ("password", "R2L"), ("u2r", "U2R"), ("r2l", "R2L")]:
        if token in key:
            return family
    return s

def make_preprocessor(X):
    num_cols = list(X.select_dtypes(include=np.number).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", MinMaxScaler())]), num_cols))
    if cat_cols:
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", enc)]), cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

def preprocess(train_df, test_df, label_col, use_pca=True, variance=0.95):
    train = train_df.copy(); test = test_df.copy()
    train_label_col = label_col if label_col in train.columns else detect_label(train)
    test_label_col = label_col if label_col in test.columns else detect_label(test)
    if not train_label_col:
        raise ValueError("Training dataset label column could not be identified.")
    if not test_label_col:
        raise ValueError("Testing dataset label column could not be identified. For NSL-KDD, upload the standard headerless KDDTest+.txt file or a CSV containing the label column.")
    y_train_raw = train.pop(train_label_col)
    y_test_raw = test.pop(test_label_col)
    # align feature columns; discard target only from features.
    common = [c for c in train.columns if c in test.columns]
    if not common:
        raise ValueError("Training and testing datasets have no common feature columns after label removal.")
    train = train[common]; test = test[common]
    prep = make_preprocessor(train)
    Xtr = prep.fit_transform(train)
    Xte = prep.transform(test)
    pca = None
    before = Xtr.shape[1]
    if use_pca and before > 2:
        pca = PCA(n_components=variance, random_state=42)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)
    return Xtr, Xte, y_train_raw, y_test_raw, prep, pca, before

def majority_mapping(cluster_ids, y):
    mapping = {}
    tmp = pd.DataFrame({"cluster": cluster_ids, "label": [normalize_label(v) for v in y]})
    for cluster, grp in tmp.groupby("cluster"):
        mapping[int(cluster)] = grp["label"].value_counts().idxmax()
    return mapping

def map_clusters_to_labels(train_clusters, y_train, test_clusters):
    mapping = majority_mapping(train_clusters, y_train)
    fallback = pd.Series([normalize_label(v) for v in y_train]).value_counts().idxmax()
    return np.array([mapping.get(int(c), fallback) for c in test_clusters]), mapping

def binary_metrics(y_true, y_pred):
    yt = np.array([0 if normalize_label(v) == "Normal" else 1 for v in y_true])
    yp = np.array([0 if str(v) == "Normal" else 1 for v in y_pred])
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
    return {
        "Accuracy": accuracy_score(yt, yp)*100,
        "Detection Rate": recall_score(yt, yp, zero_division=0)*100,
        "Precision": precision_score(yt, yp, zero_division=0)*100,
        "F1 Score": f1_score(yt, yp, zero_division=0)*100,
        "False Alarm Rate": (fp/(fp+tn)*100) if (fp+tn) else 0.0,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }

def multiclass_metrics(y_true, y_pred):
    yt = np.array([normalize_label(v) for v in y_true]); yp = np.array(y_pred)
    return {
        "Accuracy": accuracy_score(yt, yp)*100,
        "Detection Rate": recall_score(yt, yp, average="weighted", zero_division=0)*100,
        "Precision": precision_score(yt, yp, average="weighted", zero_division=0)*100,
        "F1 Score": f1_score(yt, yp, average="weighted", zero_division=0)*100,
        "False Alarm Rate": np.nan,
    }

def run_kmeans(Xtr, Xte, ytr, k, max_iter):
    t0 = time.perf_counter()
    model = KMeans(n_clusters=int(k), init="k-means++", n_init=10, max_iter=int(max_iter), random_state=42)
    ctr = model.fit_predict(Xtr)
    cte = model.predict(Xte)
    pred, mapping = map_clusters_to_labels(ctr, ytr, cte)
    elapsed = time.perf_counter()-t0
    return dict(model=model, train_clusters=ctr, test_clusters=cte, pred=pred, mapping=mapping, runtime=elapsed, inertia=float(model.inertia_), iterations=int(model.n_iter_))

def recommend_dbscan_params(Xtr, min_samples=None, sample_size=4000):
    """Estimate conservative DBSCAN parameters from a bounded sample.

    The recommendation is intended as a resource-safe starting point, not as a
    claim that the parameters are globally optimal for detection performance.
    """
    n = int(Xtr.shape[0])
    d = int(Xtr.shape[1])
    # A modest MinPts reduces sensitivity to isolated points while keeping the
    # neighborhood query manageable. For PCA-reduced data this stays bounded.
    rec_min = max(5, min(15, 2 * max(1, d)))
    if min_samples is not None:
        rec_min = int(min_samples)
    rng = np.random.RandomState(42)
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        sample = Xtr[idx]
    else:
        sample = Xtr
    k = min(max(2, rec_min), max(1, len(sample) - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=1)
    nn.fit(sample)
    dist, _ = nn.kneighbors(sample)
    # First neighbour is the point itself. Use the kth non-self distance.
    kth = dist[:, k]
    p80 = float(np.percentile(kth, 80))
    p90 = float(np.percentile(kth, 90))
    p95 = float(np.percentile(kth, 95))
    # Keep a non-zero, rounded value suitable for the UI. p90 is a conservative
    # starting point; p95 is the warning boundary rather than a hard optimum.
    safe_eps = max(0.001, p90)
    warning_eps = max(safe_eps, p95)
    return {
        "recommended_eps": safe_eps,
        "warning_eps": warning_eps,
        "recommended_min_samples": rec_min,
        "sample_size": int(len(sample)),
        "features": d,
        "k_distance_p80": p80,
        "k_distance_p90": p90,
        "k_distance_p95": p95,
    }

def assess_dbscan_safety(Xtr, eps, min_samples, recommendation, max_estimated_neighbors=2_000_000):
    """Preflight DBSCAN to prevent configurations likely to exhaust RAM.

    A bounded sample is used to estimate radius-neighbour density. The full
    experiment is blocked when the estimated number of neighbour relations is
    excessive. This is intentionally conservative because DBSCAN may materialize
    large neighbourhood structures internally.
    """
    n = int(Xtr.shape[0])
    if n < 2:
        return {"safe": False, "reason": "The training dataset must contain at least two records."}
    eps = float(eps)
    min_samples = int(min_samples)
    sample_n = min(1200, n)
    rng = np.random.RandomState(123)
    if n > sample_n:
        sample = Xtr[rng.choice(n, size=sample_n, replace=False)]
    else:
        sample = Xtr
    nn = NearestNeighbors(radius=eps, n_jobs=1)
    nn.fit(sample)
    neigh = nn.radius_neighbors(sample, return_distance=False)
    counts = np.fromiter((len(x) for x in neigh), dtype=np.int64, count=len(neigh))
    mean_neighbors = float(counts.mean()) if len(counts) else 0.0
    p95_neighbors = float(np.percentile(counts, 95)) if len(counts) else 0.0
    estimated_total = mean_neighbors * n
    ratio = eps / max(float(recommendation["recommended_eps"]), 1e-12)
    reasons = []
    if estimated_total > max_estimated_neighbors:
        reasons.append(f"estimated neighbour relations ({estimated_total:,.0f}) exceed the safety limit ({max_estimated_neighbors:,.0f})")
    if ratio > 3.0:
        reasons.append(f"epsilon is {ratio:.1f}× the data-derived safe starting value")
    safe = not reasons
    return {
        "safe": safe,
        "mean_neighbors": mean_neighbors,
        "p95_neighbors": p95_neighbors,
        "estimated_total_neighbors": estimated_total,
        "sample_size": sample_n,
        "ratio_to_recommended_eps": ratio,
        "reason": "; ".join(reasons),
    }

def run_dbscan(Xtr, Xte, ytr, eps, min_samples):
    t0 = time.perf_counter()
    # Single-threaded execution is deliberate: it reduces peak CPU/RAM pressure
    # and makes the application much less likely to freeze a user's computer.
    model = DBSCAN(eps=float(eps), min_samples=int(min_samples), n_jobs=1)
    ctr = model.fit_predict(Xtr)
    # DBSCAN has no native predict; assign each test point to its nearest core point within eps.
    core_idx = model.core_sample_indices_
    if len(core_idx):
        core = Xtr[core_idx]
        nbrs = NearestNeighbors(n_neighbors=1).fit(core)
        dist, idx = nbrs.kneighbors(Xte)
        cte = np.full(len(Xte), -1, dtype=int)
        mask = dist[:,0] <= float(eps)
        core_labels = model.labels_[core_idx]
        cte[mask] = core_labels[idx[mask,0]]
    else:
        cte = np.full(len(Xte), -1, dtype=int)
    # Noise is treated as an anomaly prediction, while clustered points inherit the majority training label.
    pred_cluster, mapping = map_clusters_to_labels(ctr, ytr, cte)
    pred_cluster = np.array(pred_cluster, dtype=object)
    pred_cluster[cte == -1] = "Attack"
    noise = int(np.sum(ctr == -1))
    clusters = int(len(set(ctr)) - (1 if -1 in ctr else 0))
    elapsed = time.perf_counter()-t0
    return dict(model=model, train_clusters=ctr, test_clusters=cte, pred=pred_cluster, mapping=mapping, runtime=elapsed, noise=noise, clusters=clusters)

def plot_confusion(y_true, y_pred, title):
    labels = sorted(set([normalize_label(v) for v in y_true]) | set(map(str, y_pred)))
    cm = confusion_matrix([normalize_label(v) for v in y_true], y_pred, labels=labels)
    fig = px.imshow(cm, text_auto=True, x=labels, y=labels, aspect="auto", title=title)
    fig.update_xaxes(title="Predicted"); fig.update_yaxes(title="Actual")
    return fig

def save_fig_png(fig):
    try:
        return fig.to_image(format="png", width=1100, height=650, scale=1)
    except Exception:
        return None

def build_pdf(results, dataset_info):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet(); story=[]
    story.append(Paragraph(APP, styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Dataset: {dataset_info['type']} | Training rows: {dataset_info['train_rows']} | Testing rows: {dataset_info['test_rows']}", styles["BodyText"]))
    story.append(Spacer(1, 12))
    for name, res in results.items():
        story.append(Paragraph(name, styles["Heading2"]))
        m = res["metrics"]
        data=[["Metric","Value"],["Accuracy",f"{m['Accuracy']:.2f}%"],["Detection Rate",f"{m['Detection Rate']:.2f}%"],["Precision",f"{m['Precision']:.2f}%"],["F1 Score",f"{m['F1 Score']:.2f}%"],["False Alarm Rate", "N/A" if np.isnan(m['False Alarm Rate']) else f"{m['False Alarm Rate']:.2f}%"],["Runtime",f"{res['runtime']:.3f} s"]]
        tbl=Table(data, colWidths=[180,120]); tbl.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.5,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1f2937')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('ALIGN',(1,1),(-1,-1),'RIGHT')]))
        story.append(tbl); story.append(Spacer(1,10))
    doc.build(story)
    return buf.getvalue()

# ----------------------------- Styling -----------------------------
st.markdown("""
<style>
:root { --bg:#0b1220; --panel:#111827; --border:#25324a; --accent:#3b82f6; }
.stApp { background: #0b1220; color: #e5e7eb; }
[data-testid="stSidebar"] { background:#0a1020; border-right:1px solid #1f2a40; }
.block-container { max-width: 1450px; padding-top: 1.2rem; }
.hero { padding: 1.25rem 1.5rem; border:1px solid #24324b; border-radius:16px; background:linear-gradient(135deg,#111827,#0f172a); margin-bottom:1rem; }
.hero h1 { margin:0; font-size:2rem; color:#f8fafc; }
.hero p { margin:.35rem 0 0; color:#94a3b8; }
.card { background:#111827; border:1px solid #25324a; border-radius:14px; padding:1rem; min-height:100px; }
.card .label { color:#94a3b8; font-size:.85rem; }
.card .value { color:#f8fafc; font-size:1.7rem; font-weight:700; margin-top:.3rem; }
.small-muted { color:#94a3b8; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="hero"><h1>🛡️ {APP}</h1><p>Interactive anomaly detection and clustering analytics for network security datasets.</p></div>', unsafe_allow_html=True)

if "results" not in st.session_state: st.session_state.results = {}
if "run_info" not in st.session_state: st.session_state.run_info = None

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("", ["Dashboard", "Dataset", "Preprocessing", "Models", "Results", "Visualizations", "Export"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Real-time experiment engine")
    if st.session_state.run_info:
        st.success("Experiment results available")
    else:
        st.info("Upload data to begin")

# ----------------------------- Dataset -----------------------------
if page in ["Dashboard", "Dataset"]:
    st.subheader("Dataset")
    c1,c2=st.columns(2)
    with c1:
        train_file=st.file_uploader("Training dataset", type=["csv","txt"], key="train_file")
    with c2:
        test_file=st.file_uploader("Testing dataset", type=["csv","txt"], key="test_file")
    if train_file and test_file:
        try:
            train_df=load_uploaded(train_file); test_df=load_uploaded(test_file)
            st.session_state.train_df=train_df; st.session_state.test_df=test_df
            st.session_state.label_col=detect_label(train_df)
            p=profile(train_df); q=profile(test_df)
            st.session_state.dataset_type=infer_dataset(train_df)
            st.success("Training and testing datasets loaded successfully.")
            cols=st.columns(5)
            for col,label,val in [(cols[0],"Dataset",st.session_state.dataset_type),(cols[1],"Training rows",p['rows']),(cols[2],"Testing rows",q['rows']),(cols[3],"Training features",p['columns']),(cols[4],"Missing values",p['missing']+q['missing'])]:
                col.markdown(f'<div class="card"><div class="label">{label}</div><div class="value" style="font-size:1.1rem">{val}</div></div>', unsafe_allow_html=True)
            st.write("")
            st.write(f"**Detected label column:** `{st.session_state.label_col}`" if st.session_state.label_col else "**Detected label column:** Not found")
            st.write("**Training preview**")
            st.dataframe(train_df.head(10), use_container_width=True)
        except Exception as e:
            st.error(str(e))
    elif page == "Dataset":
        st.info("Upload both a training dataset and a testing dataset to continue.")

# ----------------------------- Configuration -----------------------------
if page in ["Dashboard", "Preprocessing", "Models"]:
    if "train_df" not in st.session_state:
        st.warning("Upload training and testing datasets first.")
    else:
        st.subheader("Experiment Configuration")
        left,right=st.columns(2)
        with left:
            st.markdown("### Preprocessing")
            use_encoding=st.checkbox("Categorical encoding", True, disabled=True)
            use_scaling=st.checkbox("Feature scaling", True, disabled=True)
            use_pca=st.checkbox("Principal Component Analysis (PCA)", True, key="use_pca")
            pca_var=st.slider("PCA variance retained", 0.80, 0.99, 0.95, 0.01, disabled=not use_pca)
        with right:
            st.markdown("### Model selection")
            model_choice=st.radio("Select model", ["K-Means", "DBSCAN", "Both"], horizontal=True, key="model_choice")
            if model_choice in ["K-Means","Both"]:
                st.markdown("**K-Means parameters**")
                k=st.number_input("Number of clusters (K)", min_value=2, max_value=30, value=4, step=1, key="k")
                max_iter=st.number_input("Maximum iterations", min_value=50, max_value=2000, value=300, step=50, key="k_iter")
            if model_choice in ["DBSCAN","Both"]:
                st.markdown("**DBSCAN parameters**")
                st.caption("The system recommends a conservative ε/MinPts pair from a bounded k-distance sample. The recommendation is a safe starting point, not a guaranteed accuracy optimum.")
                # Recommendation is calculated after preprocessing because ε depends on the actual model feature space.
                eps_default = 0.5
                min_default = 6
                if "dbscan_recommendation" in st.session_state:
                    rec = st.session_state.dbscan_recommendation
                    eps_default = float(max(0.001, min(100.0, rec["recommended_eps"])))
                    min_default = int(rec["recommended_min_samples"])
                eps=st.number_input("Epsilon (ε)", min_value=0.001, max_value=100.0, value=float(eps_default), step=0.001, format="%.3f", key="eps")
                min_samples=st.number_input("MinPts / min samples", min_value=2, max_value=100, value=int(min_default), step=1, key="min_samples")
                if "dbscan_recommendation" in st.session_state:
                    rec = st.session_state.dbscan_recommendation
                    st.info(f"Recommended safe starting point: ε ≈ {rec['recommended_eps']:.3f}, MinPts = {rec['recommended_min_samples']} | warning boundary ≈ ε {rec['warning_eps']:.3f}")
        if st.button("🚀 RUN EXPERIMENT", type="primary", use_container_width=True):
            try:
                train_df=st.session_state.train_df; test_df=st.session_state.test_df; label_col=st.session_state.label_col
                if not label_col:
                    raise ValueError("The system could not automatically identify the attack-label column. Please provide a dataset with a recognizable label/class/attack column.")
                with st.status("Running experiment...", expanded=True) as status:
                    st.write("Validating datasets...")
                    Xtr,Xte,ytr,yte,prep,pca,before=preprocess(train_df,test_df,label_col,use_pca,pca_var)
                    st.write(f"Preprocessing complete: {before} encoded features → {Xtr.shape[1]} model features.")
                    db_rec = recommend_dbscan_params(Xtr)
                    st.session_state.dbscan_recommendation = db_rec
                    if model_choice in ["DBSCAN", "Both"]:
                        st.write(f"DBSCAN safe starting point: ε ≈ {db_rec['recommended_eps']:.3f}, MinPts = {db_rec['recommended_min_samples']} (warning boundary ≈ {db_rec['warning_eps']:.3f}).")
                    results={}
                    if model_choice in ["K-Means","Both"]:
                        st.write("Running K-Means...")
                        r=run_kmeans(Xtr,Xte,ytr,k,max_iter)
                        m=multiclass_metrics(yte,r['pred'])
                        # Binary anomaly metrics are more directly aligned with anomaly detection.
                        bm=binary_metrics(yte,r['pred'])
                        r['metrics']=bm; r['multiclass_metrics']=m
                        results['K-Means']=r
                    if model_choice in ["DBSCAN","Both"]:
                        st.write("Checking DBSCAN memory safety...")
                        safety = assess_dbscan_safety(Xtr, eps, min_samples, db_rec)
                        if not safety["safe"]:
                            raise RuntimeError(
                                "DBSCAN configuration was stopped before execution to protect system resources. "
                                + safety["reason"]
                                + f". Recommended starting point: ε ≈ {db_rec['recommended_eps']:.3f}, MinPts = {db_rec['recommended_min_samples']}."
                            )
                        st.write(f"DBSCAN safety check passed: estimated mean neighbours ≈ {safety['mean_neighbors']:.1f}; estimated total neighbour relations ≈ {safety['estimated_total_neighbors']:,.0f}.")
                        st.write("Running DBSCAN (resource-safe single-threaded mode)...")
                        r=run_dbscan(Xtr,Xte,ytr,eps,min_samples)
                        m=multiclass_metrics(yte,r['pred'])
                        bm=binary_metrics(yte,r['pred'])
                        r['metrics']=bm; r['multiclass_metrics']=m
                        r['safety']=safety
                        r['recommendation']=db_rec
                        results['DBSCAN']=r
                    st.session_state.results=results
                    st.session_state.run_info={"dataset_type":st.session_state.dataset_type,"train_rows":len(train_df),"test_rows":len(test_df),"label":label_col,"features_before":before,"features_after":Xtr.shape[1],"pca":use_pca,"pca_variance":float(pca.explained_variance_ratio_.sum()) if pca is not None else None,"y_test":yte,"train_df":train_df,"test_df":test_df,"Xtr":Xtr,"Xte":Xte,"prep":prep,"pca_obj":pca}
                    status.update(label="Experiment completed successfully", state="complete")
                st.success("Experiment completed. Open Results or Visualizations from the sidebar.")
            except Exception as e:
                st.error(f"Experiment failed: {e}")
                st.code(traceback.format_exc())

# ----------------------------- Results -----------------------------
if page == "Results":
    st.subheader("Experiment Results")
    if not st.session_state.results:
        st.info("Run an experiment first.")
    else:
        for name,res in st.session_state.results.items():
            st.markdown(f"### {name}")
            m=res['metrics']; cards=[("Accuracy",m['Accuracy']), ("Detection Rate",m['Detection Rate']), ("Precision",m['Precision']), ("F1 Score",m['F1 Score']), ("False Alarm Rate",m['False Alarm Rate'])]
            cc=st.columns(len(cards))
            for c,(lab,val) in zip(cc,cards):
                display="N/A" if isinstance(val,float) and np.isnan(val) else f"{val:.2f}%"
                c.markdown(f'<div class="card"><div class="label">{lab}</div><div class="value">{display}</div></div>', unsafe_allow_html=True)
            st.write(f"Runtime: **{res['runtime']:.3f} seconds**")
            if name == "DBSCAN":
                rec = res.get("recommendation") or st.session_state.get("dbscan_recommendation")
                safety = res.get("safety")
                if rec:
                    st.markdown("#### DBSCAN parameter recommendation")
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("Safe starting ε", f"{rec['recommended_eps']:.3f}")
                    rc2.metric("Recommended MinPts", str(rec['recommended_min_samples']))
                    rc3.metric("Warning ε", f"{rec['warning_eps']:.3f}")
                    st.caption(f"Derived from a bounded k-distance sample of {rec['sample_size']:,} training records and {rec['features']} model features. This is a resource-safe starting point; validate accuracy with your experiment.")
                if safety:
                    if safety.get("safe"):
                        st.success(f"DBSCAN safety check passed. Estimated mean neighbours: {safety['mean_neighbors']:.1f}; estimated total neighbour relations: {safety['estimated_total_neighbors']:,.0f}.")
                    else:
                        st.warning("This configuration was not executed because it was estimated to be unsafe for system resources.")
            st.dataframe(pd.DataFrame({"Metric":["Accuracy","Detection Rate","Precision","F1 Score","False Alarm Rate"],"Value (%)":[m['Accuracy'],m['Detection Rate'],m['Precision'],m['F1 Score'],m['False Alarm Rate']]}), use_container_width=True, hide_index=True)
        if len(st.session_state.results)==2:
            comp=[]
            for metric in ["Accuracy","Detection Rate","Precision","F1 Score","False Alarm Rate"]:
                for name,res in st.session_state.results.items(): comp.append({"Metric":metric,"Model":name,"Value":res['metrics'][metric]})
            st.markdown("### Model comparison")
            st.plotly_chart(px.bar(pd.DataFrame(comp),x="Metric",y="Value",color="Model",barmode="group",range_y=[0,100],title="K-Means vs DBSCAN"),use_container_width=True)

# ----------------------------- Visualizations -----------------------------
if page == "Visualizations":
    st.subheader("Interactive Visualizations")
    if not st.session_state.results or not st.session_state.run_info:
        st.info("Run an experiment first.")
    else:
        info=st.session_state.run_info; yte=info['y_test']; Xte=info['Xte']
        # PCA / feature-space visualization
        if Xte.shape[1] >= 2:
            if Xte.shape[1] > 2:
                plot_xy=Xte[:,:2]
            else: plot_xy=Xte
            base=pd.DataFrame({"Component 1":plot_xy[:,0],"Component 2":plot_xy[:,1],"Label":[normalize_label(v) for v in yte]})
            st.plotly_chart(px.scatter(base,x="Component 1",y="Component 2",color="Label",title="Testing Data — Feature Space"),use_container_width=True)
        for name,res in st.session_state.results.items():
            st.markdown(f"### {name} visualizations")
            if name=="K-Means":
                st.plotly_chart(px.scatter(pd.DataFrame({"Component 1":Xte[:,0],"Component 2":Xte[:,1],"Cluster":res['test_clusters'].astype(str)}),x="Component 1",y="Component 2",color="Cluster",title="K-Means Cluster Visualization"),use_container_width=True)
                st.plotly_chart(px.bar(pd.DataFrame({"Cluster":pd.Series(res['train_clusters']).value_counts().index.astype(str),"Count":pd.Series(res['train_clusters']).value_counts().values}),x="Cluster",y="Count",title="K-Means Cluster Distribution"),use_container_width=True)
                st.plotly_chart(plot_confusion(yte,res['pred'],"K-Means Confusion Matrix"),use_container_width=True)
            if name=="DBSCAN":
                cl=res['test_clusters']; labels=np.array(["Noise" if x==-1 else str(x) for x in cl])
                st.plotly_chart(px.scatter(pd.DataFrame({"Component 1":Xte[:,0],"Component 2":Xte[:,1],"Cluster":labels}),x="Component 1",y="Component 2",color="Cluster",title="DBSCAN Cluster / Noise Visualization"),use_container_width=True)
                train_labels=res['train_clusters']; counts=pd.Series(train_labels).value_counts().sort_index(); st.plotly_chart(px.bar(pd.DataFrame({"Cluster":counts.index.astype(str),"Count":counts.values}),x="Cluster",y="Count",title="DBSCAN Cluster Distribution"),use_container_width=True)
                st.plotly_chart(plot_confusion(yte,res['pred'],"DBSCAN Confusion Matrix"),use_container_width=True)
        if info['pca_obj'] is not None:
            ratio=info['pca_obj'].explained_variance_ratio_; cum=np.cumsum(ratio)
            st.plotly_chart(go.Figure([go.Scatter(x=np.arange(1,len(ratio)+1),y=cum*100,mode='lines+markers')]).update_layout(title="PCA Cumulative Explained Variance",xaxis_title="Components",yaxis_title="Cumulative variance (%)"),use_container_width=True)

# ----------------------------- Export -----------------------------
if page == "Export":
    st.subheader("Export Experiment")
    if not st.session_state.results:
        st.info("Run an experiment first.")
    else:
        info=st.session_state.run_info
        rows=[]
        for name,r in st.session_state.results.items():
            row={"Model":name,"Runtime_seconds":r['runtime']}; row.update({k:v for k,v in r['metrics'].items() if k not in ['TN','FP','FN','TP']}); rows.append(row)
        csv=pd.DataFrame(rows).to_csv(index=False).encode()
        st.download_button("⬇ Download Results CSV", csv, "experiment_results.csv", "text/csv", use_container_width=True)
        pdf=build_pdf(st.session_state.results, {"type":info['dataset_type'],"train_rows":info['train_rows'],"test_rows":info['test_rows']})
        st.download_button("⬇ Download PDF Report", pdf, "anomaly_detection_report.pdf", "application/pdf", use_container_width=True)
        st.download_button("⬇ Download Experiment JSON", json.dumps({k:{"metrics":v['metrics'],"runtime":v['runtime']} for k,v in st.session_state.results.items()}, indent=2, default=str).encode(), "experiment_results.json", "application/json", use_container_width=True)

# Dashboard landing message
if page == "Dashboard" and "train_df" not in st.session_state:
    st.markdown("### Welcome")
    st.write("Upload training and testing network datasets to begin an experiment.")
    a,b,c=st.columns(3)
    a.markdown('<div class="card"><div class="label">Supported</div><div class="value" style="font-size:1.15rem">NSL-KDD</div><div class="small-muted">KDD-style network data</div></div>',unsafe_allow_html=True)
    b.markdown('<div class="card"><div class="label">Supported</div><div class="value" style="font-size:1.15rem">CICIDS2017</div><div class="small-muted">Flow-based intrusion data</div></div>',unsafe_allow_html=True)
    c.markdown('<div class="card"><div class="label">Supported</div><div class="value" style="font-size:1.15rem">Generic CSV</div><div class="small-muted">Compatible labelled network data</div></div>',unsafe_allow_html=True)
