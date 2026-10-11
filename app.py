"""AgriWeed AI - Agricultural Weed Detection & Potential Yield Impact Estimator.

A clean, easy-to-use MVP that:
1. Detects and marks weeds vs crops in agricultural field images.
2. Flags infestation levels and counts.
3. Estimates crop yield impact based on detected weed presence and field size.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Configure page
st.set_page_config(
    page_title="AgriWeed AI — Weed Detection & Yield Estimator",
    page_icon="🌿",
    layout="wide",
)

# Constants & Paths
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"
MODEL_PATH = MODELS_DIR / "weedblaster_yolov8s.pt"
MODEL_URL = "https://huggingface.co/NvMayMay/weedblaster-vision-yolov8s/resolve/main/best.pt"
ACRE_TO_HECTARE = 0.40468564224

# Class definitions from CropAndWeed model
CLASS_NAMES = ["Maize", "Sugar Beet", "Soy", "Sunflower", "Potato", "Pea", "Bean", "Pumpkin", "Weed"]
WEED_CLASS_ID = 8

# Styling
st.markdown(
    """
    <style>
    .agri-header {
        background: linear-gradient(135deg, #133824 0%, #1B4332 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .agri-header h1 {
        color: #FFFFFF !important;
        margin: 0 !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    .agri-header p {
        color: #A7F3D0 !important;
        margin: 0.3rem 0 0 0 !important;
        font-size: 1rem !important;
    }
    .metric-card {
        background-color: white;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.2rem 0;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #9CA3AF;
    }
    .val-weed { color: #DC2626; }
    .val-crop { color: #166534; }
    .val-gain { color: #059669; }
    .val-loss { color: #DC2626; }
    .val-neutral { color: #1F2937; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Weed Detection Model...")
def load_model():
    """Load pretrained YOLOv8 crop-weed detection model."""
    if not MODEL_PATH.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        import requests
        with st.spinner("Downloading pretrained weed model weights (~22 MB)..."):
            resp = requests.get(MODEL_URL, stream=True, timeout=60)
            resp.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    try:
        from ultralytics import YOLO
    except ImportError:
        st.error(
            "⚠️ **`ultralytics` is not installed in the currently active Python environment!**\n\n"
            "You are running Streamlit with Python: `" + sys.executable + "`\n\n"
            "**To fix this, run Streamlit using your project's virtual environment:**\n"
            "```powershell\n"
            ".venv\\Scripts\\activate\n"
            "streamlit run app.py\n"
            "```\n"
            "**Or run directly:**\n"
            "```powershell\n"
            ".venv\\Scripts\\python.exe -m streamlit run app.py\n"
            "```"
        )
        st.stop()

    model = YOLO(str(MODEL_PATH))
    return model



def annotate_image(image_rgb: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes: Red for weeds, Green for crops."""
    ann_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    h, w = ann_bgr.shape[:2]
    thickness = max(2, int(max(h, w) / 500))

    for det in detections:
        box = [int(v) for v in det["box"]]
        x1 = max(0, min(box[0], w - 1))
        y1 = max(0, min(box[1], h - 1))
        x2 = max(0, min(box[2], w - 1))
        y2 = max(0, min(box[3], h - 1))

        is_weed = det["is_weed"]
        color = (45, 52, 239) if is_weed else (80, 185, 34)  # BGR
        label = f"{det['class_name']} {det['conf']:.0%}"

        # Draw box
        cv2.rectangle(ann_bgr, (x1, y1), (x2, y2), color, thickness)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(ann_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            ann_bgr,
            label,
            (x1 + 3, max(th, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)


def calculate_yield(
    field_area: float,
    area_unit: str,
    baseline_yield_t_ha: float,
    loss_pct: float,
    control_pct: float,
):
    """Calculate production, loss, and gain in tonnes."""
    area_ha = field_area * ACRE_TO_HECTARE if area_unit.lower() == "acre" else field_area
    r = loss_pct / 100.0
    e = control_pct / 100.0

    baseline_prod = area_ha * baseline_yield_t_ha
    uncontrolled_prod = baseline_prod * (1.0 - r)
    controlled_prod = baseline_prod * (1.0 - (r * (1.0 - e)))
    potential_loss = baseline_prod * r
    potential_gain = baseline_prod * r * e

    return {
        "area_ha": round(area_ha, 3),
        "baseline_prod": round(baseline_prod, 2),
        "uncontrolled_prod": round(uncontrolled_prod, 2),
        "controlled_prod": round(controlled_prod, 2),
        "potential_loss": round(potential_loss, 2),
        "potential_gain": round(potential_gain, 2),
    }


# Header
st.markdown(
    """
    <div class="agri-header">
        <h1>🌿 AgriWeed AI</h1>
        <p>Simple Agricultural Weed Detection & Crop Yield Impact Estimator</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 1. Image Selection
st.markdown("### 1. Select Field Photograph")
c_upload, c_samples = st.columns([1.5, 1.0], gap="medium")

with c_samples:
    st.markdown("**Quick Test Samples:**")
    btn_maize = st.button("🌽 Load Sample Maize Field", use_container_width=True)
    btn_beet = st.button("🍃 Load Sample Sugar Beet Field", use_container_width=True)

    if btn_maize:
        st.session_state["chosen_file"] = "sample_field_maize.jpg"
    elif btn_beet:
        st.session_state["chosen_file"] = "sample_field_sugarbeet.jpg"

with c_upload:
    uploaded = st.file_uploader("Upload field image (JPG / PNG):", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        st.session_state["chosen_file"] = uploaded

# Resolve selected image
image_source = st.session_state.get("chosen_file")
image_pil = None
image_name = "None"

if image_source is not None:
    if isinstance(image_source, str):
        sample_path = ASSETS_DIR / image_source
        if sample_path.exists():
            image_pil = Image.open(sample_path).convert("RGB")
            image_name = image_source
    else:
        image_pil = Image.open(image_source).convert("RGB")
        image_name = getattr(image_source, "name", "uploaded_image.jpg")

if image_pil is None:
    st.info("👆 Please upload a field photo or click one of the quick test sample buttons above.")
else:
    # 2. Weed Detection
    st.markdown("---")
    st.markdown(f"### 2. Weed & Crop Detection — `{image_name}`")

    conf_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help="Higher values show only high-confidence detections.",
    )

    # Run Detection
    model = load_model()
    results = model.predict(image_pil, conf=conf_threshold, verbose=False)

    detections = []
    weed_count = 0
    crop_count = 0
    weed_area_px = 0.0
    img_w, img_h = image_pil.size
    total_area_px = img_w * img_h

    if results and len(results) > 0:
        first = results[0]
        if first.boxes is not None and len(first.boxes) > 0:
            boxes = first.boxes.xyxy.cpu().numpy()
            confs = first.boxes.conf.cpu().numpy()
            cls_ids = first.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cid in zip(boxes, confs, cls_ids):
                cname = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else f"Class_{cid}"
                is_weed = (cid == WEED_CLASS_ID) or ("weed" in cname.lower())
                bw = max(0.0, float(box[2] - box[0]))
                bh = max(0.0, float(box[3] - box[1]))
                b_area = bw * bh

                if is_weed:
                    weed_count += 1
                    weed_area_px += b_area
                else:
                    crop_count += 1

                detections.append({
                    "box": box,
                    "conf": float(conf),
                    "class_name": cname,
                    "is_weed": is_weed,
                })

    weed_cover_pct = min(100.0, (weed_area_px / total_area_px) * 100.0)

    # Flag Infestation Severity
    if weed_count == 0:
        flag_status = "🌱 Weed-Free"
        suggested_loss = 0.0
    elif weed_count <= 5 and weed_cover_pct < 5.0:
        flag_status = "🟢 Low Infestation"
        suggested_loss = 10.0
    elif weed_count <= 20 or weed_cover_pct < 15.0:
        flag_status = "🟡 Moderate Infestation"
        suggested_loss = 22.0
    elif weed_count <= 50 or weed_cover_pct < 25.0:
        flag_status = "🟠 High Infestation"
        suggested_loss = 35.0
    else:
        flag_status = "🔴 Severe Infestation"
        suggested_loss = 45.0

    # Display KPI Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Target Weeds</div>'
            f'<div class="metric-val val-weed">{weed_count}</div>'
            f'<div class="metric-sub">Flagged competitors</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Protected Crops</div>'
            f'<div class="metric-val val-crop">{crop_count}</div>'
            f'<div class="metric-sub">Crop instances</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Weed Canopy Area</div>'
            f'<div class="metric-val val-neutral">{weed_cover_pct:.1f}%</div>'
            f'<div class="metric-sub">Of image area</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Infestation Status</div>'
            f'<div class="metric-val val-neutral" style="font-size:1.3rem;">{flag_status}</div>'
            f'<div class="metric-sub">Severity level</div></div>',
            unsafe_allow_html=True,
        )

    # Annotate and Display Image
    annotated_rgb = annotate_image(np.array(image_pil), detections)
    st.image(
        annotated_rgb,
        caption=f"Detections: {weed_count} Weeds (Red) | {crop_count} Crops (Green)",
        use_container_width=True,
    )

    # 3. Crop Yield Impact Estimator
    st.markdown("---")
    st.markdown("### 3. Crop Yield Impact Estimator")
    st.caption("Estimates potential harvest loss and recovery based on field area and detected weed competition.")

    yc1, yc2 = st.columns([1.0, 1.4], gap="large")

    with yc1:
        st.markdown("**Field Parameters:**")
        crop_type = st.selectbox(
            "Crop Type",
            ["Corn / Maize", "Sugar Beet", "Soybean", "Potato", "Sunflower", "Wheat", "Other"],
            index=0,
        )

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            field_area = st.number_input("Field Area", min_value=0.1, value=10.0, step=1.0)
        with f_col2:
            area_unit = st.selectbox("Unit", ["acre", "hectare"], index=0)

        # Typical default yields
        default_yields = {
            "Corn / Maize": 9.5,
            "Sugar Beet": 65.0,
            "Soybean": 3.2,
            "Potato": 35.0,
            "Sunflower": 2.4,
            "Wheat": 4.5,
            "Other": 5.0,
        }
        baseline_yield = st.number_input(
            "Expected Weed-Free Yield (t/ha)",
            min_value=0.1,
            value=float(default_yields.get(crop_type, 9.5)),
            step=0.5,
        )

        st.markdown("**Weed Impact Scenarios:**")
        loss_pct = st.slider(
            "Assumed Yield Loss if Weeds Uncontrolled (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(suggested_loss),
            step=1.0,
            help="Suggested automatically based on detected weed severity; adjustable.",
        )

        control_pct = st.slider(
            "Assumed Weed-Control Effectiveness (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0,
            help="Percentage of weed loss recovered through weed removal/spraying.",
        )

    # Calculate Yield Metrics
    yield_results = calculate_yield(
        field_area=field_area,
        area_unit=area_unit,
        baseline_yield_t_ha=baseline_yield,
        loss_pct=loss_pct,
        control_pct=control_pct,
    )

    with yc2:
        st.markdown("**Estimated Harvest Outcomes:**")
        yk1, yk2 = st.columns(2)
        with yk1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Weed-Free Baseline</div>'
                f'<div class="metric-val val-neutral">{yield_results["baseline_prod"]} t</div>'
                f'<div class="metric-sub">{field_area} {area_unit}s @ {baseline_yield} t/ha</div></div>',
                unsafe_allow_html=True,
            )
        with yk2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Potential Loss (Weeds Left)</div>'
                f'<div class="metric-val val-loss">-{yield_results["potential_loss"]} t</div>'
                f'<div class="metric-sub">Loss at {loss_pct}% reduction</div></div>',
                unsafe_allow_html=True,
            )

        yk3, yk4 = st.columns(2)
        with yk3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Uncontrolled Harvest</div>'
                f'<div class="metric-val val-loss">{yield_results["uncontrolled_prod"]} t</div>'
                f'<div class="metric-sub">Residual production</div></div>',
                unsafe_allow_html=True,
            )
        with yk4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Recovered with Control</div>'
                f'<div class="metric-val val-gain">+{yield_results["potential_gain"]} t</div>'
                f'<div class="metric-sub">{control_pct}% loss recovered ({yield_results["controlled_prod"]} t total)</div></div>',
                unsafe_allow_html=True,
            )

        st.info(
            f"ℹ️ **Formula Summary**: Baseline = Area ({yield_results['area_ha']} ha) × {baseline_yield} t/ha. "
            f"If weeds reduce yield by {loss_pct}%, potential harvest drops by {yield_results['potential_loss']} tonnes. "
            f"Applying {control_pct}% effective weed control recovers **{yield_results['potential_gain']} tonnes**."
        )
