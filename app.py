app_code = '''
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json, time, os, io, warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
warnings.filterwarnings("ignore")

FEATURE_COLS = ["platform_id", "time_delta_hours", "contact_freq_7d",
                "new_account_flag", "graph_distance"]

NORM = {
    "platform_id":      5.0,
    "time_delta_hours": 200.0,
    "contact_freq_7d":  39.0,
    "new_account_flag": 1.0,
    "graph_distance":   2.0,
}

PHASE_NAMES = {
    1: "Direct Contact",
    2: "Indirect Contact",
    3: "Platform Migration",
    4: "Proxy Harassment",
}

PLATFORM_NAMES = {0: "Instagram", 1: "Twitter", 2: "WhatsApp",
                   3: "Facebook", 4: "Telegram", 5: "Reddit"}

DIST_NAMES = {0: "Mutual connection", 1: "Friend-of-friend", 2: "Stranger"}

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"
DATASET_PATH   = os.path.join("data", "cyberstalking_dataset.csv")

BASELINE_BT = -0.135

class GuardianMLP(nn.Module):
    def __init__(self, input_dim=5, hidden1=32, hidden2=16, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2),   nn.ReLU(),
            nn.Linear(hidden2, output_dim), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    m = GuardianMLP().to(device)
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "model_phase4.pt"), map_location=device,weights_only=False)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m, device

@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)

@st.cache_data
def load_metrics():
    with open(os.path.join(RESULTS_DIR, "metrics_phase4.json"), "r") as f:
        metrics = json.load(f)
    return {int(k): v for k, v in metrics["phase_accuracies"].items()}

@st.cache_resource
def load_shap_explainer(_model, _device):
    df = load_dataset()
    bg = _normalise(df[df["phase"] == 1].sample(100, random_state=42))
    def predict_fn(X):
        X = np.atleast_2d(X).astype(np.float32)
        t = torch.tensor(X, dtype=torch.float32).to(_device)
        with torch.no_grad():
            preds = _model(t).cpu().numpy()
        return np.atleast_1d(preds.squeeze())
    return shap.KernelExplainer(predict_fn, bg)

def _normalise(df_slice):
    arr = df_slice[FEATURE_COLS].values.astype(np.float32)
    for i, col in enumerate(FEATURE_COLS):
        arr[:, i] /= NORM[col]
    return arr

def normalise_single(row):
    return np.array([row[col] / NORM[col] for col in FEATURE_COLS], dtype=np.float32)

def infer(model, row, device):
    t = torch.tensor(normalise_single(row), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        stalking_prob = model(t).item()
    pred = 1 if stalking_prob >= 0.5 else 0
    confidence = stalking_prob if pred == 1 else (1.0 - stalking_prob)
    return pred, round(confidence, 4)

def bt_score(phase_accuracies, T=4):
    bt_sum, count = 0.0, 0
    for i in range(1, T):
        if len(phase_accuracies[i]) >= 2:
            bt_sum += phase_accuracies[i][-1] - phase_accuracies[i][0]
            count  += 1
    return bt_sum / count if count > 0 else 0.0

def phase_accuracy_from_events(events, phase):
    phase_evts = [e for e in events if e["phase"] == phase]
    if not phase_evts:
        return None
    return sum(1 for e in phase_evts if e["correct"]) / len(phase_evts)

def shap_explanation_lines(features_raw, shap_vals):
    contributions = sorted(zip(FEATURE_COLS, shap_vals),
                            key=lambda x: abs(x[1]), reverse=True)
    lines = []
    for col, val in contributions[:3]:
        raw = features_raw[col]
        direction = "up" if val > 0 else "down"
        if col == "time_delta_hours":
            lines.append((direction, f"Contact gap of {raw:.1f}h {'raised' if direction=='up' else 'lowered'} threat score"))
        elif col == "contact_freq_7d":
            lines.append((direction, f"Contact frequency {int(raw)}/wk {'raised' if direction=='up' else 'lowered'} threat score"))
        elif col == "new_account_flag":
            lines.append((direction, f"{'New account' if raw==1 else 'Established account'} {'raised' if direction=='up' else 'lowered'} threat score"))
        elif col == "graph_distance":
            lines.append((direction, f"Sender is {DIST_NAMES.get(int(raw), str(int(raw)))} — {'raised' if direction=='up' else 'lowered'} threat score"))
        elif col == "platform_id":
            lines.append((direction, f"Platform {PLATFORM_NAMES.get(int(raw), raw)} {'raised' if direction=='up' else 'lowered'} threat score"))
    return lines

def generate_pdf_report(events, victim_id, mode, phases_used, bt, baseline_bt):
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=letter,
                             leftMargin=0.75*inch, rightMargin=0.75*inch,
                             topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18, spaceAfter=6)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13, spaceAfter=4, textColor=colors.HexColor("#1a1a2e"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceAfter=3, textColor=colors.HexColor("#333366"))
    normal  = styles["Normal"]
    caption = ParagraphStyle("Caption", parent=normal, fontSize=8, textColor=colors.grey)
    threat_style = ParagraphStyle("Threat", parent=normal, backColor=colors.HexColor("#ffeeee"), textColor=colors.HexColor("#990000"), fontSize=9, leading=14)
    safe_style   = ParagraphStyle("Safe",   parent=normal, backColor=colors.HexColor("#eeffee"), textColor=colors.HexColor("#006600"), fontSize=9, leading=14)

    story = []
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    story.append(Paragraph("GUARDIAN AI", title_style))
    story.append(Paragraph("Legal Evidence Report — Cyberstalking Detection", h1))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#333366")))
    story.append(Spacer(1, 8))

    meta = [
        ["Victim ID",        f"Victim {victim_id:02d}"],
        ["Report Mode",      "Multi-Phase Stalking Flow" if mode=="flow" else "Single Phase Evaluation"],
        ["Phases Covered",   ", ".join(f"Phase {p} ({PHASE_NAMES[p]})" for p in sorted(phases_used))],
        ["Total Events",     str(len(events))],
        ["Threats Flagged",  str(sum(1 for e in events if e["predicted_label"]==1))],
        ["Overall Accuracy", f"{sum(1 for e in events if e['correct'])/len(events)*100:.1f}%"],
        ["BT Score",         f"{bt*100:.1f}% (baseline: {baseline_bt*100:.1f}%)"],
        ["Generated",        now],
    ]
    meta_table = Table(meta, colWidths=[2*inch, 4.5*inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#e8e8f0")),
        ("FONTNAME",   (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f7f7fb")]),
        ("PADDING",    (0,0), (-1,-1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Phase-Wise Accuracy", h1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 6))
    acc_data = [["Phase", "Description", "Events", "Correct", "Accuracy"]]
    for p in sorted(phases_used):
        p_evts    = [e for e in events if e["phase"] == p]
        p_correct = sum(1 for e in p_evts if e["correct"])
        p_acc     = p_correct / len(p_evts) * 100 if p_evts else 0
        acc_data.append([f"Phase {p}", PHASE_NAMES[p], str(len(p_evts)), str(p_correct), f"{p_acc:.1f}%"])
    acc_table = Table(acc_data, colWidths=[0.7*inch, 2.2*inch, 0.8*inch, 0.8*inch, 0.9*inch])
    acc_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f7f7fb")]),
        ("ALIGN",       (2,0), (-1,-1), "CENTER"),
        ("PADDING",     (0,0), (-1,-1), 5),
    ]))
    story.append(acc_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Incident Timeline", h1))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 6))
    current_phase = None
    for i, evt in enumerate(events):
        if evt["phase"] != current_phase:
            current_phase = evt["phase"]
            story.append(Spacer(1, 8))
            story.append(Paragraph(f"Phase {current_phase} — {PHASE_NAMES[current_phase]}", h2))
        fr        = evt["features_raw"]
        is_threat = evt["predicted_label"] == 1
        verdict   = "THREAT DETECTED" if is_threat else "Safe — Normal Event"
        correct   = "" if evt["correct"] else "  [MISCLASSIFIED]"
        platform  = PLATFORM_NAMES.get(int(fr["platform_id"]), "?")
        dist      = DIST_NAMES.get(int(fr["graph_distance"]), "?")
        acct      = "New account" if fr["new_account_flag"]==1 else "Known account"
        line = (f"Event {i+1}: {verdict}{correct}  |  Confidence: {evt['confidence']:.1%}  |  "
                f"Platform: {platform}  |  Freq: {int(fr['contact_freq_7d'])}/wk  |  "
                f"Gap: {fr['time_delta_hours']:.1f}h  |  {acct}  |  {dist}")
        story.append(Paragraph(line, threat_style if is_threat else safe_style))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#333366")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated by Guardian AI  |  Brain-Inspired Continual Learning System  |  {now}", caption))
    story.append(Paragraph("This report is generated by an AI system and should be reviewed by a qualified professional before use as legal evidence.", caption))
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

@st.cache_data
def build_stream_single(victim_id, phase, label_filter, n_events):
    df   = load_dataset()
    mask = (df["victim_id"] == victim_id) & (df["phase"] == phase)
    if label_filter is not None:
        mask = mask & (df["label"] == label_filter)
    subset = df[mask].reset_index(drop=True)
    if n_events < len(subset):
        subset = subset.iloc[:n_events]
    return subset.to_dict(orient="records")

@st.cache_data
def build_stream_flow(victim_id, phases, n_events_per_phase, label_filter):
    df   = load_dataset()
    rows = []
    for p in phases:
        mask = (df["victim_id"] == victim_id) & (df["phase"] == p)
        if label_filter is not None:
            mask = mask & (df["label"] == label_filter)
        subset = df[mask].reset_index(drop=True)
        if n_events_per_phase < len(subset):
            subset = subset.iloc[:n_events_per_phase]
        rows.extend(subset.to_dict(orient="records"))
    return rows

# ══════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════
st.set_page_config(page_title="Guardian AI", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
/* ── Base ── */
.stApp                          { background-color: #F5F6FA; color: #111827; }
.block-container                { padding-top: 1.5rem; }

/* ── Sidebar ── */
[data-testid="stSidebarContent"] {
    background-color: #1E2240;
    color: #FFFFFF;
}
[data-testid="stSidebarContent"] label,
[data-testid="stSidebarContent"] .stRadio label,
[data-testid="stSidebarContent"] .stSelectbox label,
[data-testid="stSidebarContent"] .stSlider label,
[data-testid="stSidebarContent"] p,
[data-testid="stSidebarContent"] span,
[data-testid="stSidebarContent"] div  { color: #FFFFFF !important; }
[data-testid="stSidebarContent"] [data-testid="stCaptionContainer"] p { color: #A5B4FC !important; }
[data-testid="stSidebarContent"] h1,
[data-testid="stSidebarContent"] h2,
[data-testid="stSidebarContent"] h3   { color: #FFFFFF !important; }
[data-testid="stSidebarContent"] hr   { border-color: #3A3F6E; }

/* ── Sidebar selectbox & radio text ── */
[data-testid="stSidebarContent"] .stSelectbox div[data-baseweb="select"] div,
[data-testid="stSidebarContent"] .stSelectbox div[data-baseweb="select"] span {
    color: #111827 !important;
}
[data-testid="stSidebarContent"] .stRadio div[role="radiogroup"] label span {
    color: #FFFFFF !important;
}

/* ── Sidebar slider labels and values ── */
[data-testid="stSidebarContent"] [data-testid="stSlider"] div,
[data-testid="stSidebarContent"] [data-testid="stSlider"] span,
[data-testid="stSidebarContent"] [data-testid="stSlider"] p {
    color: #FFFFFF !important;
}

/* ── Sidebar buttons ── */
[data-testid="stSidebarContent"] .stButton > button {
    background-color: #3A4080 !important;
    color: #FFFFFF !important;
    border: 1px solid #6B7DB3 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebarContent"] .stButton > button:hover {
    background-color: #4F6EF7 !important;
    border-color: #4F6EF7 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]     { background: #E8EAFF; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"]          { color: #1E2240 !important; font-weight: 700; border-radius: 6px; }
.stTabs [aria-selected="true"]        { background: #FFFFFF !important; color: #1E2240 !important; }

/* ── Headings & text ── */
h1, h2, h3, h4                        { color: #1E2240 !important; }
p, span, label, div                   { color: #111827; }

/* ── Metrics ── */
[data-testid="stMetric"]              { background: #FFFFFF; border-radius: 10px;
                                        padding: 12px 16px; border: 1px solid #DDE1F0; }
[data-testid="stMetricLabel"]         { color: #1E2240 !important; font-weight: 700; }
[data-testid="stMetricValue"]         { color: #1E2240 !important; font-weight: 800; }
[data-testid="stMetricDelta"]         { font-weight: 600; color: #374151 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"]           { background: #FFFFFF; border-radius: 8px; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div   { background-color: #4F6EF7; }

/* ── Buttons ── */
.stButton > button[kind="primary"]    { background: #4F6EF7; color: #fff; border: none;
                                        font-weight: 700; border-radius: 8px; }
.stButton > button                    { border-radius: 8px; font-weight: 600;
                                        color: #1E2240 !important; }

/* ── Event cards ── */
.threat-card {
    background: #FFF0F0;
    border-left: 5px solid #DC2626;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(220,38,38,0.12);
}
.safe-card {
    background: #F0FFF4;
    border-left: 5px solid #16A34A;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(22,163,74,0.10);
}
.event-label {
    font-size: 1.0em;
    font-weight: 700;
    color: #111827 !important;
}
.event-meta {
    font-size: 0.85em;
    color: #1F2937 !important;
    margin-top: 6px;
    line-height: 1.8;
    word-wrap: break-word;
    white-space: normal;
}
.event-meta strong {
    color: #111827 !important;
}
.phase-divider {
    margin: 16px 0 8px 0;
    padding: 8px 14px;
    background: #E8EAFF;
    border-left: 4px solid #4F6EF7;
    border-radius: 6px;
    color: #1E2240 !important;
    font-size: 0.92em;
    font-weight: 700;
    word-wrap: break-word;
    white-space: normal;
}

/* ── Caption text ── */
[data-testid="stCaptionContainer"] p  { color: #374151 !important; }

/* ── Info / warning boxes ── */
[data-testid="stAlert"] p             { color: #1E2240 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

model, device    = load_model()
df_full          = load_dataset()
phase_accuracies = load_metrics()
bt               = bt_score(phase_accuracies)
bt_improvement   = bt - BASELINE_BT

# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ Guardian AI")
    st.caption("Brain-Inspired Continual Learning · Cyberstalking Detection")
    st.markdown("---")

    bt_col   = "#4ADE80" if bt > BASELINE_BT else "#F87171"
    imp_sign = "+" if bt_improvement >= 0 else ""
    st.markdown(f"""
    <div style="background:#2A3060;border:1px solid #4F6EF7;border-radius:10px;
                padding:14px 16px;margin-bottom:14px;">
        <div style="color:#A5B4FC;font-size:0.72em;letter-spacing:0.08em;font-weight:700;">
            BACKWARD TRANSFER (BT)
        </div>
        <div style="color:#FFFFFF;font-size:2.2em;font-weight:800;line-height:1.1;margin-top:4px;">
            {bt*100:.1f}%
        </div>
        <div style="color:{bt_col};font-size:0.82em;margin-top:4px;font-weight:600;">
            {imp_sign}{bt_improvement*100:.1f}% vs baseline ({BASELINE_BT*100:.1f}%)
        </div>
        <div style="color:#A5B4FC;font-size:0.72em;margin-top:6px;">
            Target: &gt;80% &nbsp;·&nbsp; Baseline: {BASELINE_BT*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    selected_victim = st.selectbox("👤 Victim",
                                   options=list(range(10)),
                                   format_func=lambda v: f"Victim {v:02d}")

    demo_mode = st.radio("🎬 Demo Mode",
                         ["Single Phase Evaluation", "Multi-Phase Stalking Flow"],
                         index=0)

    if demo_mode == "Single Phase Evaluation":
        selected_phases = [st.selectbox(
            "📅 Phase", options=[1,2,3,4],
            format_func=lambda p: f"Phase {p} — {PHASE_NAMES[p]}"
        )]
        event_type = st.radio("🎯 Event Type",
                              ["Both", "Normal only (label=0)", "Stalking only (label=1)"],
                              index=0)
        n_events     = st.slider("📊 Events to stream", 5, 50, 20)
        label_filter = {"Both": None, "Normal only (label=0)": 0, "Stalking only (label=1)": 1}[event_type]
        mode_key     = "single"
    else:
        phase_options   = st.multiselect("📅 Phases (streamed in order)",
                                         options=[1,2,3,4], default=[1,2,3,4],
                                         format_func=lambda p: f"Phase {p} — {PHASE_NAMES[p]}")
        selected_phases = sorted(phase_options) if phase_options else [1]
        event_type      = st.radio("🎯 Event Type",
                                   ["Both", "Normal only (label=0)", "Stalking only (label=1)"],
                                   index=0)
        n_events        = st.slider("📊 Events per phase", 5, 50, 20)
        label_filter    = {"Both": None, "Normal only (label=0)": 0, "Stalking only (label=1)": 1}[event_type]
        mode_key        = "flow"

    stream_speed = st.select_slider("⚡ Speed",
                                    options=["Slow (1.0s)", "Normal (0.4s)", "Fast (0.1s)"],
                                    value="Normal (0.4s)")
    delay = {"Slow (1.0s)": 1.0, "Normal (0.4s)": 0.4, "Fast (0.1s)": 0.1}[stream_speed]

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    start_btn  = c1.button("▶ Run",  use_container_width=True, type="primary")
    pause_btn  = c2.button("⏸ Pause", use_container_width=True)
    reset_btn  = c3.button("↺ Reset", use_container_width=True)

# ══════════════════════════════════════════════════════
#  EVENT STREAM — Fix 4: append new events, never wipe
# ══════════════════════════════════════════════════════
if mode_key == "single":
    event_stream = build_stream_single(selected_victim, selected_phases[0], label_filter, n_events)
else:
    event_stream = build_stream_flow(selected_victim, selected_phases, n_events, label_filter)

total_events = len(event_stream)
stream_key   = f"{selected_victim}_{mode_key}_{'_'.join(map(str,selected_phases))}_{label_filter}_{n_events}"

# Initialise session state on very first load
if "stream_key" not in st.session_state:
    st.session_state.stream_key    = stream_key
    st.session_state.events        = []
    st.session_state.cursor        = 0
    st.session_state.running       = False
    st.session_state.all_streams   = []   # cumulative list across all stream configs

# Reset button: wipe everything
if reset_btn:
    st.session_state.stream_key  = stream_key
    st.session_state.events      = []
    st.session_state.cursor      = 0
    st.session_state.running     = False
    st.session_state.all_streams = []

# Sidebar config changed (but NOT reset): keep old events, start fresh cursor for new segment
elif st.session_state.stream_key != stream_key:
    st.session_state.stream_key = stream_key
    st.session_state.cursor     = 0
    st.session_state.running    = False
    # events are intentionally NOT cleared — new segment appends to existing history

if start_btn: st.session_state.running = True
if pause_btn: st.session_state.running = False

tab1, tab2, tab3 = st.tabs(["📋 Incident Report", "📊 Model Behaviour", "🔍 Event Inspector"])
evts = st.session_state.events

# ══════════════════════════════════════════════════════
#  TAB 1 — INCIDENT REPORT
# ══════════════════════════════════════════════════════
with tab1:
    mode_label = ("Multi-Phase Stalking Flow" if mode_key=="flow"
                  else f"Phase {selected_phases[0]} — {PHASE_NAMES[selected_phases[0]]}")
    st.markdown(f"### Live Incident Timeline — Victim {selected_victim:02d} · {mode_label}")

    prog_val    = st.session_state.cursor / total_events if total_events > 0 else 0
    progress_ph = st.progress(prog_val)
    status_ph   = st.empty()

    if total_events == 0:
        st.warning("No events match this combination. Adjust filters.")
    else:
        status_ph.caption(
            f"{'🔴 Streaming' if st.session_state.running else '⏸ Paused'}  —  "
            f"{st.session_state.cursor}/{total_events} events in current segment  —  "
            f"{len(evts)} total events in timeline"
        )

    # Phase-wise accuracy summary across ALL accumulated events
    all_phases_seen = sorted({e["phase"] for e in evts}) if evts else []
    if len(all_phases_seen) > 0:
        st.markdown("#### Phase-Wise Accuracy")
        acc_cols = st.columns(max(len(all_phases_seen), 1))
        for idx, p in enumerate(all_phases_seen):
            acc     = phase_accuracy_from_events(evts, p)
            p_evts  = [e for e in evts if e["phase"] == p]
            flagged = sum(1 for e in p_evts if e["predicted_label"]==1)
            if acc is not None:
                acc_cols[idx].metric(
                    f"P{p}: {PHASE_NAMES[p]}",
                    f"{acc*100:.1f}%",
                    delta=f"{flagged} threats / {len(p_evts)} events"
                )
        st.markdown("---")

    rendered_phase = None
    for evt in evts:
        fr         = evt["features_raw"]
        is_threat  = evt["predicted_label"] == 1
        card_class = "threat-card" if is_threat else "safe-card"
        verdict    = "⚠️ THREAT DETECTED" if is_threat else "✅ Normal Event"
        platform   = PLATFORM_NAMES.get(int(fr["platform_id"]), "?")
        dist       = DIST_NAMES.get(int(fr["graph_distance"]), "?")
        acct       = "🆕 New acct" if fr["new_account_flag"]==1 else "👤 Known acct"
        misc       = "" if evt["correct"] else "&nbsp;&nbsp;<span style='color:#DC2626;font-weight:700;'>⚠ misclassified</span>"

        if evt["phase"] != rendered_phase:
            rendered_phase = evt["phase"]
            st.markdown(
                f"<div class='phase-divider'>📍 Phase {rendered_phase} — {PHASE_NAMES[rendered_phase]}</div>",
                unsafe_allow_html=True
            )

        st.markdown(f"""
        <div class="{card_class}">
            <div class="event-label">{verdict}{misc}</div>
            <div class="event-meta">
                Confidence: <strong>{evt["confidence"]:.1%}</strong>
                &nbsp;·&nbsp; 📱 {platform}
                &nbsp;·&nbsp; 📈 {int(fr["contact_freq_7d"])}/wk
                &nbsp;·&nbsp; ⏱ {fr["time_delta_hours"]:.1f}h gap
                &nbsp;·&nbsp; {acct}
                &nbsp;·&nbsp; 🔗 {dist}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📄 Legal Evidence Report")
    if not evts:
        st.info("Run the stream first — the report will include all classified events.")
    else:
        phases_used = sorted({e["phase"] for e in evts})
        if st.button("📥 Generate & Download PDF Report", type="secondary"):
            with st.spinner("Generating legal evidence report..."):
                pdf_bytes = generate_pdf_report(evts, selected_victim, mode_key,
                                                phases_used, bt, BASELINE_BT)
            st.download_button(label="⬇️ Download Report PDF", data=pdf_bytes,
                               file_name=f"guardian_ai_victim{selected_victim:02d}_report.pdf",
                               mime="application/pdf", type="primary")

# ══════════════════════════════════════════════════════
#  TAB 2 — MODEL BEHAVIOUR
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown(f"### Confidence Stream — Victim {selected_victim:02d}")

    if not evts:
        st.info("Press ▶ Run to start streaming.")
    else:
        total   = len(evts)
        correct = sum(1 for e in evts if e["correct"])
        flagged = sum(1 for e in evts if e["predicted_label"]==1)
        tp      = sum(1 for e in evts if e["predicted_label"]==1 and e["true_label"]==1)
        prec    = tp / flagged if flagged > 0 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Events",    total)
        m2.metric("Accuracy",  f"{correct/total*100:.1f}%")
        m3.metric("Flagged",   flagged)
        m4.metric("Precision", f"{prec*100:.1f}%")

        # Phase-wise breakdown across all accumulated events
        all_phases_seen2 = sorted({e["phase"] for e in evts})
        if len(all_phases_seen2) > 1:
            st.markdown("#### Phase-Wise Accuracy Breakdown")
            rows = [["Phase", "Description", "Events", "Threats", "Accuracy"]]
            for p in all_phases_seen2:
                p_evts = [e for e in evts if e["phase"]==p]
                if not p_evts: continue
                p_corr = sum(1 for e in p_evts if e["correct"])
                p_flag = sum(1 for e in p_evts if e["predicted_label"]==1)
                rows.append([f"Phase {p}", PHASE_NAMES[p], len(p_evts), p_flag,
                              f"{p_corr/len(p_evts)*100:.1f}%"])
            acc_df = pd.DataFrame(rows[1:], columns=rows[0])
            st.dataframe(acc_df, use_container_width=True, hide_index=True)

        st.markdown("#### Confidence Scores")
        conf_vals = [e["confidence"]      for e in evts]
        true_labs = [e["true_label"]      for e in evts]
        pred_labs = [e["predicted_label"] for e in evts]
        phases_ev = [e["phase"]           for e in evts]

        fig, ax = plt.subplots(figsize=(11, 3.5))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#F5F6FA")

        dot_cols = ["#16A34A" if p==t else "#DC2626"
                    for p, t in zip(pred_labs, true_labs)]
        ax.scatter(range(len(conf_vals)), conf_vals,
                   c=dot_cols, s=22, alpha=0.9, zorder=3, edgecolors="white", linewidths=0.4)
        ax.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1.0, alpha=0.8)

        # Draw phase boundary lines across all events
        prev = phases_ev[0]
        for i, ph in enumerate(phases_ev):
            if ph != prev:
                ax.axvline(i, color="#4F6EF7", linestyle="--", linewidth=1.4, alpha=0.8)
                ax.text(i+1, 0.94, f"P{ph}", color="#4F6EF7", fontsize=8, fontweight="bold")
                prev = ph

        green_p = mpatches.Patch(color="#16A34A", label="Correct")
        red_p   = mpatches.Patch(color="#DC2626", label="Misclassified")
        ax.legend(handles=[green_p, red_p], loc="lower right",
                  facecolor="#FFFFFF", edgecolor="#DDE1F0", labelcolor="#111827", fontsize=9)
        ax.set_xlabel("Event index",          color="#374151", fontsize=10)
        ax.set_ylabel("Confidence in predicted class", color="#374151", fontsize=10)
        ax.tick_params(colors="#374151")
        ax.set_ylim(-0.05, 1.05)
        for sp in ax.spines.values():
            sp.set_edgecolor("#DDE1F0")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ══════════════════════════════════════════════════════
#  TAB 3 — EVENT INSPECTOR
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown("### Event Inspector — SHAP Feature Contributions")
    flagged_evts = [e for e in evts if e["predicted_label"]==1]

    if not flagged_evts:
        st.info("No flagged events yet — run the stream first.")
    else:
        event_labels = [
            f"Event {i+1}  ·  Phase {e['phase']} ({PHASE_NAMES[e['phase']]})  ·  "
            f"conf {e['confidence']:.1%}  ·  {'✓ correct' if e['correct'] else '✗ misclassified'}"
            for i, e in enumerate(flagged_evts)
        ]
        chosen_idx = st.selectbox("Select flagged event", range(len(flagged_evts)),
                                  format_func=lambda i: event_labels[i])
        chosen_evt = flagged_evts[chosen_idx]
        fr         = chosen_evt["features_raw"]

        shap_key = (f"shap_v{chosen_evt['victim_id']}_p{chosen_evt['phase']}"
                    f"_c{chosen_evt['confidence']}")
        if shap_key not in st.session_state:
            with st.spinner("Computing SHAP values (~10s)..."):
                explainer  = load_shap_explainer(model, device)
                feat_norm  = np.array([fr[c]/NORM[c] for c in FEATURE_COLS], dtype=np.float32)
                sv         = explainer.shap_values(feat_norm.reshape(1,-1), nsamples=100, silent=True)
                st.session_state[shap_key] = np.array(sv).reshape(-1)

        shap_vals = st.session_state[shap_key]
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("**Feature Contributions**")
            fig2, ax2 = plt.subplots(figsize=(6, 3.2))
            fig2.patch.set_facecolor("#FFFFFF")
            ax2.set_facecolor("#F5F6FA")
            # Fix 1: positive SHAP (pushes toward threat) = red, negative (away from threat) = green
            bar_cols = ["#DC2626" if v > 0 else "#16A34A" for v in shap_vals]
            bars = ax2.barh(FEATURE_COLS, shap_vals, color=bar_cols, height=0.55)
            ax2.axvline(0, color="#6B7280", linewidth=0.9)
            for bar, val in zip(bars, shap_vals):
                xpos = val + 0.003 if val >= 0 else val - 0.003
                ha   = "left" if val >= 0 else "right"
                ax2.text(xpos, bar.get_y() + bar.get_height()/2,
                         f"{val:+.3f}", va="center", ha=ha, color="#111827", fontsize=8)
            ax2.set_xlabel("SHAP contribution to threat score", color="#374151", fontsize=9)
            ax2.tick_params(colors="#374151", labelsize=9)
            ax2.set_title(f"Confidence: {chosen_evt['confidence']:.1%}", color="#1E2240", fontsize=10)
            for sp in ax2.spines.values():
                sp.set_edgecolor("#DDE1F0")
            # Legend so it's clear which colour means what
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor="#DC2626", label="Increases threat score"),
                               Patch(facecolor="#16A34A", label="Decreases threat score")]
            ax2.legend(handles=legend_elements, loc="lower right", fontsize=7,
                       facecolor="#FFFFFF", edgecolor="#DDE1F0", labelcolor="#111827")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        with col_b:
            st.markdown("**Plain-English Explanation**")
            lines = shap_explanation_lines(fr, shap_vals)
            for direction, text in lines:
                if direction == "up":
                    st.error(f"↑ {text}")
                else:
                    st.success(f"↓ {text}")

            st.markdown("---")
            label_str   = "Stalking ⚠️" if chosen_evt["true_label"]==1 else "Normal ✅"
            correct_str = "Correctly flagged" if chosen_evt["correct"] else "Misclassified — true label: Normal"
            st.markdown(f"""
<div style="display:flex; gap:12px; margin-top:12px;">
    <div style="flex:1; background:#FFFFFF; border:1px solid #DDE1F0; border-radius:10px; padding:12px 14px;">
        <div style="font-size:0.75em; font-weight:700; color:#4B5563;">True Label</div>
        <div style="font-size:1.1em; font-weight:800; color:#1E2240; margin-top:4px; word-wrap:break-word;">{label_str}</div>
    </div>
    <div style="flex:1; background:#FFFFFF; border:1px solid #DDE1F0; border-radius:10px; padding:12px 14px;">
        <div style="font-size:0.75em; font-weight:700; color:#4B5563;">Phase</div>
        <div style="font-size:1.1em; font-weight:800; color:#1E2240; margin-top:4px; word-wrap:break-word;">Phase {chosen_evt['phase']} — {PHASE_NAMES[chosen_evt['phase']]}</div>
    </div>
    <div style="flex:1; background:#FFFFFF; border:1px solid #DDE1F0; border-radius:10px; padding:12px 14px;">
        <div style="font-size:0.75em; font-weight:700; color:#4B5563;">Result</div>
        <div style="font-size:1.1em; font-weight:800; color:#1E2240; margin-top:4px; word-wrap:break-word;">{"✓ Correctly flagged" if chosen_evt["correct"] else "✗ Misclassified — true label: Normal"}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  STREAMING LOOP
# ══════════════════════════════════════════════════════
if (st.session_state.running
        and st.session_state.cursor < total_events
        and total_events > 0):

    row        = event_stream[st.session_state.cursor]
    pred, conf = infer(model, row, device)

    st.session_state.events.append({
        "victim_id"      : int(row["victim_id"]),
        "phase"          : int(row["phase"]),
        "true_label"     : int(row["label"]),
        "predicted_label": pred,
        "confidence"     : conf,
        "correct"        : pred == int(row["label"]),
        "features_raw"   : {col: row[col] for col in FEATURE_COLS},
    })
    st.session_state.cursor += 1

    progress_ph.progress(st.session_state.cursor / total_events)
    status_ph.caption(
        f"🔴 Streaming  —  {st.session_state.cursor}/{total_events} in segment  —  "
        f"Phase {row['phase']} ({PHASE_NAMES[int(row['phase'])]})  —  "
        f"Victim {selected_victim:02d}  —  {len(st.session_state.events)} total events"
    )
    
    st.rerun()

elif st.session_state.cursor >= total_events and total_events > 0:
    st.session_state.running = False
    status_ph.success(
        f"✓ Segment complete — {total_events} events classified · "
        f"{len(evts)} total events in timeline · Victim {selected_victim:02d}"
    )
'''

with open("app.py", "w") as f:
    f.write(app_code)

print("app.py written. Re-run your launch cell.")
