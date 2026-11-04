import os, time, cv2, numpy as np, torch, streamlit as st
import plotly.graph_objects as go
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
from cls_model import build_model
from config import CLS_SIZE, SEG_SIZE

CLS_PATH = "models/resnet18_pneumonia.pth"
SEG_PATH = "models/unet_seg.pth"
LABELS = ["NORMAL", "PNEUMONIA"]          # same order as ImageFolder
GREEN, RED, BLUE, AMBER = "#22c55e", "#ef4444", "#38bdf8", "#f59e0b"

st.set_page_config(page_title="X-ray Triage Copilot", page_icon="🩻", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
#MainMenu, footer {visibility: hidden;}
.hero {
    background: linear-gradient(120deg, #0ea5e9 0%, #6366f1 55%, #8b5cf6 100%);
    border-radius: 22px; padding: 30px 36px; margin-bottom: 18px;
    box-shadow: 0 12px 40px rgba(99,102,241,.25);
}
.hero h1 {color: #fff; font-size: 2.1rem; margin: 0 0 6px 0; font-weight: 800;}
.hero p  {color: rgba(255,255,255,.88); margin: 0; font-size: 1.02rem;}
.hero-title {display: flex; align-items: center; gap: 14px;}
.project-icon {width: 54px; height: 54px; flex: 0 0 54px;}
.badge {
    display: inline-block; padding: 4px 12px; margin: 12px 8px 0 0; border-radius: 999px;
    background: rgba(255,255,255,.18); color: #fff; font-size: .78rem; font-weight: 600;
    border: 1px solid rgba(255,255,255,.25);
}
.card {
    background: #111a2e; border: 1px solid #1f2b47; border-radius: 18px;
    padding: 18px 20px; height: 100%;
}
.card h4 {margin: 0 0 8px 0; color: #9fb3d9; font-size: .8rem; letter-spacing: .08em; text-transform: uppercase;}
.image-heading {margin-bottom: 14px;}
.big {font-size: 1.9rem; font-weight: 800; margin: 0;}
.sub {color: #8da2c9; font-size: .85rem; margin-top: 2px;}
.verdict {
    border-radius: 18px; padding: 20px 24px; margin-bottom: 14px;
    border: 1px solid; display: flex; align-items: center; gap: 16px;
}
.verdict .icon {font-size: 2.4rem;}
.verdict .t {font-size: 1.5rem; font-weight: 800; margin: 0;}
.verdict .s {margin: 2px 0 0 0; opacity: .85; font-size: .92rem;}
.v-bad  {background: rgba(239,68,68,.10); border-color: rgba(239,68,68,.45); color: #fecaca;}
.v-good {background: rgba(34,197,94,.10); border-color: rgba(34,197,94,.45); color: #bbf7d0;}
.step {text-align: center; padding: 8px;}
.step .n {font-size: 1.8rem;}
.side-card {
    background: #0f172a; border: 1px solid #22304f; border-radius: 14px;
    padding: 12px 14px; margin-bottom: 10px; font-size: .86rem; color: #cbd5e1;
}
.side-card b {color: #fff;}
.tag {color: #38bdf8; font-weight: 700; font-size: .72rem; letter-spacing: .08em; text-transform: uppercase;}
.stTabs [data-baseweb="tab"] {font-weight: 600;}
.stButton>button, .stDownloadButton>button {border-radius: 12px; font-weight: 700;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ models
@st.cache_resource
def load_models():
    clf = build_model()
    clf.load_state_dict(torch.load(CLS_PATH, map_location="cpu"))
    clf.eval()
    seg = None
    if os.path.exists(SEG_PATH):
        seg = smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=1)
        seg.load_state_dict(torch.load(SEG_PATH, map_location="cpu"))
        seg.eval()
    return clf, seg


tf = transforms.Compose([
    transforms.Resize((CLS_SIZE, CLS_SIZE)), transforms.Grayscale(3),
    transforms.ToTensor(), transforms.Normalize([0.485] * 3, [0.229] * 3)])


# ------------------------------------------------------------------ chart helpers
def style(fig, h=280):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e6edf7"), legend=dict(orientation="h", y=-0.05))
    return fig


def donut(labels, values, colors, center_text):
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.68, sort=False,
                           marker=dict(colors=colors, line=dict(color="#0b1220", width=3)),
                           textinfo="percent", textfont_size=13))
    fig.add_annotation(text=center_text, x=0.5, y=0.5, showarrow=False,
                       font=dict(size=17, color="#fff"))
    return style(fig)


def gauge(value):
    color = GREEN if value < 35 else AMBER if value < 65 else RED
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number=dict(suffix="%", font=dict(size=40)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor="#64748b"),
                   bar=dict(color=color, thickness=0.28), bgcolor="#0f172a", borderwidth=0,
                   steps=[dict(range=[0, 35], color="rgba(34,197,94,.18)"),
                          dict(range=[35, 65], color="rgba(245,158,11,.18)"),
                          dict(range=[65, 100], color="rgba(239,68,68,.18)")])))
    return style(fig, 260)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🩻 Triage Copilot")
    st.caption("AI-assisted chest X-ray analysis")

    st.markdown("""
    <div class="side-card"><span class="tag">Classification model</span><br>
    <b>ResNet18</b> (transfer learning, ImageNet)<br>
    Output: NORMAL / PNEUMONIA<br>Input size: <b>%d x %d</b></div>
    <div class="side-card"><span class="tag">Segmentation model</span><br>
    <b>U-Net</b> with ResNet18 encoder<br>
    Output: infection mask<br>Input size: <b>%d x %d</b></div>
    <div class="side-card"><span class="tag">Datasets</span><br>
    Kaggle Chest X-Ray Pneumonia (classification)<br>
    COVID-QU-Ex infection masks (segmentation)</div>
    """ % (CLS_SIZE, CLS_SIZE, SEG_SIZE, SEG_SIZE), unsafe_allow_html=True)

    st.markdown("### Controls")
    threshold = st.slider("Mask sensitivity", 0.10, 0.90, 0.50, 0.05,
                          help="Lower = larger highlighted area, higher = stricter.")
    opacity = st.slider("Overlay opacity", 0.1, 0.9, 0.5, 0.05)
    show_mask = st.toggle("Show segmentation", value=True)

    st.markdown("---")
    st.warning("Research and education only. Not a medical device. Not for clinical decisions.")

# ------------------------------------------------------------------ hero
st.markdown("""
<div class="hero">
  <div class="hero-title">
    <svg class="project-icon" viewBox="0 0 64 64" role="img" aria-label="Chest X-ray project icon" xmlns="http://www.w3.org/2000/svg">
      <rect x="2" y="2" width="60" height="60" rx="17" fill="rgba(255,255,255,.16)" stroke="rgba(255,255,255,.5)" stroke-width="2"/>
      <path d="M32 13v15m0-8-7 8m7-8 7 8" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M24 27c-5 1-9 7-10 15-1 7 2 11 7 10 5-1 9-7 9-14V29m10-2c5 1 9 7 10 15 1 7-2 11-7 10-5-1-9-7-9-14V29" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M32 28v23" fill="none" stroke="#bae6fd" stroke-width="2" stroke-linecap="round"/>
    </svg>
    <h1>Chest X-ray Pneumonia Analysis</h1>
  </div>
  <p>Upload a chest X-ray to get an AI classification and a highlighted view of the suspected infected region.</p>
  <span class="badge">Deep Learning</span><span class="badge">Classification</span>
  <span class="badge">Segmentation</span><span class="badge">PyTorch</span>
</div>
""", unsafe_allow_html=True)

if not os.path.exists(CLS_PATH):
    st.error(f"Missing {CLS_PATH}. Run classification_train.py first.")
    st.stop()

clf, seg = load_models()

file = st.file_uploader("Upload a chest X-ray (PNG / JPG)", type=["png", "jpg", "jpeg"],
                        label_visibility="collapsed")

# ------------------------------------------------------------------ empty state
if not file:
    cols = st.columns(4)
    steps = [("📤", "Upload", "Drop a chest X-ray image"),
             ("🧠", "Classify", "ResNet18: NORMAL or PNEUMONIA"),
             ("🎯", "Segment", "U-Net highlights suspected regions"),
             ("📊", "Review", "Charts, heatmap and downloadable result")]
    for c, (icon, title, text) in zip(cols, steps):
        c.markdown(f'<div class="card step"><div class="n">{icon}</div>'
                   f'<b>{title}</b><div class="sub">{text}</div></div>', unsafe_allow_html=True)
    st.stop()

# ------------------------------------------------------------------ inference (same logic as before)
img = Image.open(file).convert("L")
gray = np.array(img)

t0 = time.time()
with st.spinner("Analyzing X-ray..."):
    with torch.no_grad():
        prob = torch.softmax(clf(tf(img).unsqueeze(0)), 1)[0]
    cls_time = time.time() - t0

    prob_map, mask, overlay, heat = None, None, None, None
    if show_mask and seg is not None:
        g = cv2.resize(gray, (SEG_SIZE, SEG_SIZE))
        t = torch.tensor(g / 255., dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        with torch.no_grad():
            prob_map = torch.sigmoid(seg(t))[0, 0].numpy()
        mask = (prob_map > threshold).astype(np.uint8)

        base = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        overlay = base.copy()
        red = np.array([255, 60, 60])
        overlay[mask == 1] = ((1 - opacity) * base[mask == 1] + opacity * red).astype(np.uint8)

        hm = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        heat = cv2.addWeighted(base, 0.55, hm, 0.45, 0)

label = LABELS[prob.argmax().item()]
conf = prob.max().item() * 100
p_pneu = prob[1].item() * 100
bad = label == "PNEUMONIA"

# ------------------------------------------------------------------ verdict banner
st.markdown(f"""
<div class="verdict {'v-bad' if bad else 'v-good'}">
  <div class="icon">{'⚠️' if bad else '✅'}</div>
  <div><p class="t">{label} detected</p>
  <p class="s">Model confidence {conf:.1f}% &middot; analysed in {cls_time*1000:.0f} ms</p></div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🎯 Segmentation", "📝 Report"])

# ------------------------------------------------------------------ tab 1
with tab1:
    c1, c2, c3 = st.columns([1.05, 1, 1])
    with c1:
        st.markdown('<div class="card image-heading"><h4>Original X-ray</h4></div>', unsafe_allow_html=True)
        st.image(gray, use_container_width=True)
    with c2:
        st.markdown('<div class="card"><h4>Class probability</h4></div>', unsafe_allow_html=True)
        st.plotly_chart(donut(LABELS, [prob[0].item(), prob[1].item()], [GREEN, RED],
                              f"<b>{label}</b>"), use_container_width=True)
    with c3:
        st.markdown('<div class="card"><h4>Pneumonia risk score</h4></div>', unsafe_allow_html=True)
        st.plotly_chart(gauge(p_pneu), use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    tiles = [(m1, "Prediction", label, RED if bad else GREEN),
             (m2, "Confidence", f"{conf:.1f}%", BLUE),
             (m3, "Normal prob.", f"{prob[0].item()*100:.1f}%", GREEN),
             (m4, "Pneumonia prob.", f"{p_pneu:.1f}%", RED)]
    for col, name, val, color in tiles:
        col.markdown(f'<div class="card"><h4>{name}</h4>'
                     f'<p class="big" style="color:{color}">{val}</p></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ tab 2
with tab2:
    if not show_mask:
        st.info("Segmentation is switched off in the sidebar.")
    elif seg is None:
        st.info("Segmentation model not found. Run segmentation_train.py to enable this tab.")
    else:
        area = mask.mean() * 100
        a, b, c = st.columns(3)
        a.markdown('<div class="card image-heading"><h4>Input</h4></div>', unsafe_allow_html=True)
        a.image(cv2.resize(gray, (SEG_SIZE, SEG_SIZE)), use_container_width=True)
        b.markdown('<div class="card image-heading"><h4>Mask overlay</h4></div>', unsafe_allow_html=True)
        b.image(overlay, use_container_width=True)
        c.markdown('<div class="card image-heading"><h4>Probability heatmap</h4></div>', unsafe_allow_html=True)
        c.image(heat, use_container_width=True)

        d, e = st.columns([1, 1.2])
        with d:
            st.plotly_chart(donut(["Highlighted area", "Clear area"], [area, 100 - area],
                                  [RED, "#334155"], f"<b>{area:.1f}%</b>"), use_container_width=True)
        with e:
            st.markdown(f'<div class="card"><h4>Affected area estimate</h4>'
                        f'<p class="big" style="color:{RED if area > 10 else AMBER}">{area:.1f}% of image</p>'
                        f'<p class="sub">At sensitivity threshold {threshold:.2f}. '
                        f'Move the slider in the sidebar to see the mask change.</p></div>',
                        unsafe_allow_html=True)
            if not bad and area > 2:
                st.caption("Note: the mask model found some area even though the classifier says NORMAL. "
                           "The two models were trained on different datasets, so they can disagree.")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.markdown("### Analysis summary")
    summary = (f"Prediction: {label}\nConfidence: {conf:.1f}%\n"
               f"Normal probability: {prob[0].item()*100:.1f}%\nPneumonia probability: {p_pneu:.1f}%\n")
    if mask is not None:
        summary += f"Highlighted area: {mask.mean()*100:.1f}% (threshold {threshold:.2f})\n"
    summary += ("Models: ResNet18 classifier, U-Net (ResNet18 encoder) segmentation\n"
                "Disclaimer: research and education only, not for clinical use.\n")
    st.code(summary, language="text")

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download summary (.txt)", summary, "xray_summary.txt")
    if overlay is not None:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        d2.download_button("⬇️ Download overlay (.png)", buf.tobytes(), "xray_overlay.png", "image/png")

    with st.expander("How it works"):
        st.write("The classifier (ResNet18) looks at the whole image and outputs a probability for "
                 "NORMAL and PNEUMONIA. The segmentation network (U-Net) predicts, for every pixel, "
                 "how likely it is to belong to an infected region. The threshold slider decides how "
                 "strict that pixel decision is.")
