import streamlit as st
import pandas as pd
import os
import uuid
import time
from PIL import Image, UnidentifiedImageError
import requests
import io
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import re
import random
from io import BytesIO
import datetime

# === Core Configuration ===
st.set_option('client.showErrorDetails', True)
st.set_page_config(page_title="AI-Assisted Dermatological Diagnosis Research", page_icon="🩺", layout="centered")

# Global CSS styles, SCI-paper modern UI, color blocks, card containers, and mobile optimization
st.markdown("""
<style>
/* === 1. Global & Typography Styles === */
.stApp {
    background-color: #F0F4F8 !important;
    color: #1a1a1a !important;
    font-family: Inter, Roboto, Arial, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
}

/* Base text color & line heights */
p, label, span, div, .stMarkdown {
    color: #1a1a1a !important;
    line-height: 1.6 !important;
    font-family: Inter, Roboto, Arial, sans-serif !important;
}

/* Heading hierarchies & tight letter spacing */
h1, h2, h3, h4, h5, h6 {
    color: #1a1a1a !important;
    font-family: Inter, Roboto, Arial, sans-serif !important;
    letter-spacing: -0.3px !important;
    line-height: 1.3 !important;
}

h1 {
    font-weight: 700 !important;
    font-size: 26px !important;
}

h2, h3 {
    font-weight: 600 !important;
}

/* Mobile responsive heading adjustments */
@media (max-width: 768px) {
    h1 {
        font-size: 20px !important;
    }
    h2, h3 {
        font-size: 18px !important;
    }
    .main .block-container {
        padding-left: 16px !important;
        padding-right: 16px !important;
    }
}

/* === 2. Streamlit Native Elements Clean-up === */
#MainMenu, footer, header {visibility: hidden !important;}
.stDeployButton {display: none !important;}
div[data-testid="stToolbar"] {display: none !important;}
div[data-testid="stDecoration"] {display: none !important;}
div[data-testid="stStatusWidget"] {display: none !important;}

h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
    display: none !important;
}

.streamlit-footer-mask {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 45px;
    background-color: #F0F4F8;
    z-index: 999999;
    pointer-events: none;
}

div[data-baseweb="select"] input {
    caret-color: transparent !important;
}

div[data-baseweb="select"], .stSelectbox, .stSlider {
    width: 100% !important;
}

/* === 3. SCI-Paper UI Components (Cards & Header Blocks) === */
.card-container {
    background-color: #FFFFFF;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
    border: 1px solid #E2E8F0;
}

.header-block {
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
    color: #FFFFFF;
    padding: 12px 18px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 16px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: -0.2px;
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
}

/* === 4. Button Styling Upgrades === */
div.stButton > button, div.stFormSubmitButton > button {
    background-color: #DC2626 !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    width: 100% !important;
    height: 48px !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 4px rgba(220, 38, 38, 0.2);
    transition: all 0.2s ease-in-out;
}

div.stButton > button:hover, div.stFormSubmitButton > button:hover {
    background-color: #B91C1C !important;
    box-shadow: 0 4px 8px rgba(220, 38, 38, 0.3);
    transform: translateY(-1px);
}

div.stButton > button:active, div.stFormSubmitButton > button:active {
    transform: scale(0.98);
}

/* Secondary / Back button styling */
div.stButton.secondary-btn > button {
    background-color: #F3F4F6 !important;
    color: #374151 !important;
    border: 1px solid #D1D5DB !important;
}
div.stButton.secondary-btn > button:hover {
    background-color: #E5E7EB !important;
    color: #1F2937 !important;
}

/* === 5. Slider Customization === */
div[data-baseweb="slider"] {
    padding-top: 10px;
    padding-bottom: 10px;
}
.stSlider span[role="slider"] {
    background-color: #DC2626 !important;
    border-color: #DC2626 !important;
    width: 16px !important;
    height: 16px !important;
}

/* === 6. Warning Box Refinement === */
.custom-warning-box {
    background-color: #FFF8E1;
    border-left: 4px solid #FFA000;
    padding: 14px 18px;
    border-radius: 8px;
    margin: 16px 0;
    color: #1a1a1a;
    font-size: 14px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
.custom-warning-box b {
    color: #1a1a1a;
}
</style>
<div class="streamlit-footer-mask"></div>
""", unsafe_allow_html=True)

# Performance optimization configuration
REQUEST_TIMEOUT = 2
CACHE_TTL = 3600
IMAGE_COMPRESS_WIDTH = 600
IMAGE_QUALITY = 85

# GitHub Configuration
GITHUB_USERNAME = "grass134"
GITHUB_REPO = "skin-question"
GOLD_TXT = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/boosted_final_detail4.UTF-8.txt"

# ========== Google Sheets Configuration ==========
GOOGLE_SHEET_NAME = "Skin Diagnosis Data"
LOCAL_GOOGLE_CREDENTIALS_FILE = "google_credentials.json"

# GitHub Image Folder
GITHUB_IMAGE_FOLDER = "experiment_pool"
GITHUB_BRANCH = "main"

# Unified Disease Labels & Grouping Dictionary
DISEASE_GROUPS = {
    "Melanocytic Lesions": {
        "MEL": "Melanoma",
        "NV": "Nevus (Melanocytic Nevus)"
    },
    "Epithelial Tumors": {
        "BCC": "Basal Cell Carcinoma",
        "SCC": "Squamous Cell Carcinoma",
        "AK": "Actinic Keratosis"
    },
    "Benign Lesions": {
        "BKL": "Benign Keratosis",
        "DF": "Dermatofibroma",
        "VASC": "Vascular Lesion"
    },
    "Inflammatory/Other": {
        "Vitiligo": "Vitiligo",
        "Pityrasis-Alba": "Pityriasis Alba",
        "Psoriasis": "Psoriasis"
    },
    "Unknown Category": {
        "UNK": "Unknown Category"
    }
}

# Flat Dictionary for reverse lookups
DISEASE_LABELS = {}
for group_dict in DISEASE_GROUPS.values():
    for k, v in group_dict.items():
        DISEASE_LABELS[k] = v

ALL_CLASSES = list(DISEASE_LABELS.values())
TEST_COUNT = 10

# === Deduplication Function (Preserve order, exclude "N/A") ===
def deduplicate_preserve_order(lst):
    seen = set()
    result = []
    for x in lst:
        if x not in seen and x != "N/A":
            seen.add(x)
            result.append(x)
    return result

# === Get CST Time Function ===
def get_cst_time():
    cst_tz = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(cst_tz).strftime("%Y-%m-%d %H:%M:%S")

# === Google Sheets Initialization ===
@st.cache_resource(ttl=CACHE_TTL, show_spinner=False)
def init_google_sheets_once():
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]

        try:
            creds_dict = dict(st.secrets["GOOGLE_CREDENTIALS"])
            if "private_key" in creds_dict:
                creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        except KeyError:
            if not os.path.exists(LOCAL_GOOGLE_CREDENTIALS_FILE):
                return None, "❌ Credentials file google_credentials.json not found"
            creds = ServiceAccountCredentials.from_json_keyfile_name(LOCAL_GOOGLE_CREDENTIALS_FILE, scope)

        client = gspread.authorize(creds)
        sheet = client.open(GOOGLE_SHEET_NAME).sheet1

        required_headers = [
            "doctor_id", "hospital_level", "work_years", "daily_patients", "prior_ai_trust",
            "image_id", "true_label", "ai_label", "ai_is_correct", "initial_top1", "initial_top2",
            "initial_top3", "initial_confidence", "is_initial_top1_correct", "is_initial_top3_correct",
            "interaction_type", "action_taken", "use_ai", "final_top1", "final_top2", "final_top3",
            "final_top4", "is_final_top1_correct", "is_final_top3_correct", "is_final_top4_correct",
            "final_confidence", "confidence_gain", "decision_path", "is_misled", "is_rescued",
            "time_baseline", "time_post_ai", "submit_time"
        ]
        headers = sheet.row_values(1)
        if not headers or len(headers) != len(required_headers):
            sheet.clear()
            sheet.append_row(required_headers)

        return sheet, None

    except gspread.exceptions.SpreadsheetNotFound:
        return None, f"❌ Spreadsheet not found: {GOOGLE_SHEET_NAME}"
    except Exception as e:
        return None, f"❌ Google Sheets initialization failed: {str(e)}"

# === Session State Initialization ===
def init_session_state():
    default_states = {
        "step": "profile",
        "current_idx": 0,
        "show_ai": False,
        "user_results": [],
        "test_set": None,
        "doctor_info": {},
        "ai_suggestion": {},
        "initial_top": ["Select Diagnosis", "N/A", "N/A"],
        "initial_conf": 5,
        "final_top1": "", "final_top2": "", "final_top3": "", "final_top4": "",
        "final_conf": 5,
        "question_start": None,
        "time_baseline": 0,
        "doctor_id": "",
        "ai_same_as_initial": False,
        "answered_image_ids": [],
        "show_lightbox": False,
    }
    for k, v in default_states.items():
        if k not in st.session_state:
            st.session_state[k] = v

# === Test Set Loading ===
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_gold_data_cached():
    try:
        resp = requests.get(GOLD_TXT, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), encoding="utf-8")
        req_cols = ["image_id", "Top1_预测", "真实病名"]
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            return None, f"Missing fields: {missing}"

        df["true_cn"] = df["真实病名"].map(DISEASE_LABELS).fillna("Unknown")
        df["ai_cn"] = df["Top1_预测"].map(DISEASE_LABELS).fillna("Unknown")
        df["ai_correct"] = df["true_cn"] == df["ai_cn"]
        df = df[(df["true_cn"] != "Unknown") & (df["ai_cn"] != "Unknown")]
        return df, None
    except Exception as e:
        return None, f"Loading failed: {str(e)}"

# === Random Case Sampling Rule Optimization ===
def load_balanced_test_set(df, completed_image_ids=None):
    if completed_image_ids is None:
        completed_image_ids = []
    
    available_df = df[~df["image_id"].isin(completed_image_ids)]
    if len(available_df) < TEST_COUNT:
        available_df = df  
        
    correct_sample = pd.DataFrame()
    incorrect_sample = pd.DataFrame()
    ai_correct = available_df[available_df["ai_correct"]]
    ai_incorrect = available_df[~available_df["ai_correct"]]

    n_correct = min(6, len(ai_correct))
    n_incorrect = TEST_COUNT - n_correct
    if n_incorrect > len(ai_incorrect):
        n_incorrect = len(ai_incorrect)
        n_correct = TEST_COUNT - n_incorrect

    if len(ai_correct) > 0:
        correct_sample = ai_correct.sample(n_correct, replace=False)
    if len(ai_incorrect) > 0:
        incorrect_sample = ai_incorrect.sample(n_incorrect, replace=False)

    if correct_sample.empty and incorrect_sample.empty:
        return df.head(TEST_COUNT)

    test_set = pd.concat([correct_sample, incorrect_sample]).sample(frac=1).reset_index(drop=True)
    return test_set.head(TEST_COUNT)

# === Save to Sheets ===
def save_results_to_gs():
    with st.spinner("Saving data to Google Sheets..."):
        sheet, err = init_google_sheets_once()
        if err:
            st.error(err)
            return False

        if not st.session_state.user_results:
            st.warning("No results available to save")
            return False

        rows = []
        for r in st.session_state.user_results:
            row = [
                r["doctor_id"], r["hospital_level"], r["work_years"], r["daily_patients"], r["prior_ai_trust"],
                r["image_id"], r["true_label"], r["ai_label"], r["ai_is_correct"],
                r["initial_top1"], r["initial_top2"], r["initial_top3"], r["initial_confidence"],
                r["is_initial_top1_correct"], r["is_initial_top3_correct"],
                r["interaction_type"], r["action_taken"], r["use_ai"],
                r["final_top1"], r["final_top2"], r["final_top3"], r["final_top4"],
                r["is_final_top1_correct"], r["is_final_top3_correct"], r["is_final_top4_correct"],
                r["final_confidence"], r["confidence_gain"], r["decision_path"],
                r["is_misled"], r["is_rescued"], r["time_baseline"], r["time_post_ai"],
                r["submit_time"]
            ]
            rows.append(row)

        try:
            sheet.append_rows(rows)
            st.success("✅ Successfully saved records")
            return True
        except Exception as e:
            st.error(f"❌ Write failed: {str(e)}")
            return False

# === Single Question State Reset ===
def reset_test_state():
    st.session_state.show_ai = False
    st.session_state.initial_top = ["Select Diagnosis", "N/A", "N/A"]
    st.session_state.initial_conf = 5
    st.session_state.final_top1 = ""
    st.session_state.final_top2 = ""
    st.session_state.final_top3 = ""
    st.session_state.final_top4 = "N/A"
    st.session_state.final_conf = 5
    st.session_state.time_baseline = 0
    st.session_state.ai_same_as_initial = False
    st.session_state.question_start = None
    st.session_state.show_lightbox = False

# === Image Compression ===
def compress_image(image_url):
    try:
        r = requests.get(image_url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content))
        if img.mode in ("RGBA", "P", "L"):
            img = img.convert("RGB")
        w, h = img.size
        ratio = IMAGE_COMPRESS_WIDTH / w
        new_h = int(h * ratio)
        img = img.resize((IMAGE_COMPRESS_WIDTH, new_h), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=IMAGE_QUALITY, optimize=True)
        buf.seek(0)
        return buf
    except UnidentifiedImageError:
        blank = Image.new("RGB", (600, 400), "#eee")
        buf = BytesIO()
        blank.save(buf, "JPEG")
        buf.seek(0)
        return buf
    except:
        try:
            return BytesIO(requests.get(image_url, timeout=REQUEST_TIMEOUT).content)
        except:
            blank = Image.new("RGB", (600, 400), "#eee")
            buf = BytesIO()
            blank.save(buf, "JPEG")
            buf.seek(0)
            return buf

# === Image URL Retrieval ===
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_image_url_cached(image_id):
    clean_id = re.sub(r"\.(jpg|png)$", "", image_id)
    paths = []
    lower_id = clean_id.lower()
    if "pity" in lower_id:
        paths.append(f"{GITHUB_IMAGE_FOLDER}/pityriasis-alba-images/{clean_id}.jpg")
    elif "psoriasis" in lower_id:
        paths.append(f"{GITHUB_IMAGE_FOLDER}/PSORIASIS/{clean_id}.jpg")
    elif "vitiligo" in lower_id:
        paths.append(f"{GITHUB_IMAGE_FOLDER}/vitiligo/{clean_id}.jpg")
    elif clean_id.startswith("ISIC_"):
        paths.append(f"{GITHUB_IMAGE_FOLDER}/{clean_id}.jpg")
    paths.append(f"{GITHUB_IMAGE_FOLDER}/{clean_id}.jpg")

    base = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/"
    for p in paths[:4]:
        u = base + p
        try:
            if requests.head(u, timeout=REQUEST_TIMEOUT).status_code == 200:
                return u
        except:
            continue
    fallback = random.choice(["ISIC_0034334", "ISIC_0034402", "ISIC_0034411"])
    return f"{base}{GITHUB_IMAGE_FOLDER}/{fallback}.jpg"

# === Custom Grouped Selectbox Helper ===
def grouped_selectbox(label, options_list, key, help_text=None, placeholder="Select Diagnosis"):
    flat_options = [placeholder]
    for group_name, diseases in DISEASE_GROUPS.items():
        flat_options.append(f"── {group_name} ──")
        for d_code, d_name in diseases.items():
            if d_name in options_list:
                flat_options.append(d_name)
    
    selected = st.selectbox(label, flat_options, key=key, help=help_text)
    if selected and selected.startswith("──"):
        st.warning("Please select a valid disease category option, not a group header.")
        return placeholder
    return selected

# === Doctor Profile Page ===
def profile_step():
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("# AI-Assisted Dermatological Diagnosis Research Survey")
    st.markdown("""
Dear Doctor:
Thank you for taking time out of your busy clinical schedule to participate in this survey!
This test consists of 10 multiple-choice questions and takes about 3-7 minutes to complete. Every careful judgment and genuine feedback from you carries great value and responsibility for medical research.
We will properly and strictly anonymize all your data, allowing your professional experience to make a greater impact.
Once again, our sincere gratitude to you, and we wish you all the best!
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("""
    <div class="header-block">
        <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>
        Step 1: Basic Information Collection (Anonymous)
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("profile_form"):
        hospital_level = st.selectbox(
            "Professional Background",
            ["Tertiary-A Hospital Dermatologist", "Secondary Hospital Dermatologist", "Community Hospital / Resident Physician"],
            help="Please select your practicing hospital level and professional background"
        )
        work_years = st.selectbox(
            "Years of Dermatological Practice",
            ["<=5 years", "5-10 years", "10-15 years", ">15 years", "No Clinical Experience (Intern)"],
            help="Please select your years of experience in dermatology"
        )
        daily_patients = st.selectbox(
            "Average Daily Dermatological Patients",
            ["<=15 patients", "15-30 patients", ">30 patients", "No Outpatient Experience"],
            help="Please select your average daily patient volume range"
        )
        prior_ai_trust = st.selectbox(
            "Initial Confidence in AI-Assisted Diagnosis (Score 1-10)",
            options=list(range(1, 11)),
            index=4,
            help="1: Completely distrust, 10: Completely trust"
        )

        submit_btn = st.form_submit_button("Submit & Start Test")

    if submit_btn:
        prefix = "A" if "Tertiary-A" in hospital_level else "B" if "Secondary" in hospital_level else "C"
        doctor_id = f"{prefix}_DR_{uuid.uuid4().hex[:6].upper()}"
        st.session_state.doctor_id = doctor_id

        st.session_state.doctor_info = {
            "doctor_id": doctor_id,
            "hospital_level": hospital_level,
            "work_years": work_years,
            "daily_patients": daily_patients,
            "prior_ai_trust": prior_ai_trust
        }

        with st.spinner("Loading test cases..."):
            df, err = load_gold_data_cached()
            if err:
                st.error(err)
                return
            if work_years == ">15 years" and len(df[~df["ai_correct"]]) >= 2:
                add_samples = df[~df["ai_correct"]].sample(2)
                df = pd.concat([df, add_samples]).drop_duplicates()
            st.session_state.test_set = load_balanced_test_set(df, st.session_state.answered_image_ids)

        st.session_state.step = "test"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# === Test Question Page ===
def test_step():
    ts = st.session_state.test_set
    if ts is None or ts.empty:
        st.error("Failed to load test set, please refresh and try again")
        return

    idx = st.session_state.current_idx
    if idx >= TEST_COUNT:
        save_results_to_gs()
        st.components.v1.html("""
            <script>
                try {
                    sessionStorage.removeItem("skin_survey_progress");
                } catch(e) {}
            </script>
        """, height=0)
        st.session_state.step = "result"
        st.rerun()

    cur = ts.iloc[idx]
    img_id = cur["image_id"]
    truth = cur["true_cn"]
    ai_lbl = cur["ai_cn"]
    ai_ok = ai_lbl == truth

    if st.session_state.question_start is None:
        st.session_state.question_start = time.time()

    # Progress & Title Card
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown(f"### Case Diagnosis - Question {idx+1} of {TEST_COUNT}")
    st.markdown(f"**Case {idx+1} / 10**")
    st.progress((idx+1)/TEST_COUNT)
    st.markdown('</div>', unsafe_allow_html=True)

    # Lesion Image Card
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("""
    <div class="header-block">
        <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path><path stroke-linecap="round" stroke-linejoin="round" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
        Lesion Image
    </div>
    """, unsafe_allow_html=True)
    
    img_url = get_image_url_cached(img_id)
    compressed_img = compress_image(img_url)
    
    st.image(compressed_img, use_container_width=True)

    # Lightbox Modal Integration via st.components.v1.html & Custom Button
    if st.button("🔍 View Full Resolution"):
        st.session_state.show_lightbox = True

    if st.session_state.get("show_lightbox", False):
        lightbox_html = f"""
        <div id="custom-lightbox" style="position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.85);z-index:999999;display:flex;justify-content:center;align-items:center;cursor:pointer;" onclick="document.getElementById('custom-lightbox').style.display='none';">
            <div style="position:relative; max-width:90%; max-height:90%;" onclick="event.stopPropagation();">
                <img src="{img_url}" style="width:100%; height:auto; max-height:85vh; border-radius:8px; object-fit:contain; box-shadow: 0 10px 25px rgba(0,0,0,0.5);" />
                <button onclick="document.getElementById('custom-lightbox').style.display='none';" style="position:absolute;top:-15px;right:-15px;background:#DC2626;color:white;border:none;border-radius:50%;width:36px;height:36px;font-size:18px;cursor:pointer;font-weight:bold;">&times;</button>
            </div>
        </div>
        """
        st.components.v1.html(lightbox_html, height=0)

    st.markdown('</div>', unsafe_allow_html=True)

    # Independent Diagnosis Card
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("""
    <div class="header-block">
        <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
        I. Independent Diagnosis
    </div>
    """, unsafe_allow_html=True)

    with st.form(f"initial_diagnosis_form_{idx}"):
        t1 = grouped_selectbox(
            "Primary Diagnosis",
            ALL_CLASSES,
            key=f"t1_{idx}",
            help_text="Please select your most likely primary diagnosis from the categorized list (Required)",
            placeholder="Select Diagnosis"
        )
        
        t2_options = ["N/A"] + [x for x in ALL_CLASSES if x != t1]
        t2 = grouped_selectbox(
            "Secondary Diagnosis",
            t2_options,
            key=f"t2_{idx}",
            help_text="Optional secondary differential diagnosis",
            placeholder="N/A"
        )
        
        t3_options = ["N/A"] + [x for x in ALL_CLASSES if x not in [t1, t2]]
        t3 = grouped_selectbox(
            "Tertiary Diagnosis",
            t3_options,
            key=f"t3_{idx}",
            help_text="Optional tertiary differential diagnosis",
            placeholder="N/A"
        )
        
        conf_i = st.slider(
            "Confidence in This Diagnosis (Score 1-10)",
            1, 10, 5,
            key=f"ci_{idx}",
            help="1: Completely uncertain, 10: Completely certain"
        )

        submit_initial = st.form_submit_button("Submit & View AI Suggestion")
        if submit_initial:
            if t1 == "Select Diagnosis" or t1.startswith("──"):
                st.error("Please select a valid primary diagnosis before submitting")
            else:
                time_baseline = round(time.time() - st.session_state.question_start, 2)
                st.session_state.time_baseline = time_baseline

                st.session_state.initial_top = [t1, t2, t3]
                st.session_state.initial_conf = conf_i
                st.session_state.ai_suggestion = {"label": ai_lbl}
                st.session_state.ai_same_as_initial = (t1 == ai_lbl)
                st.session_state.show_ai = True
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.show_ai:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown("""
        <div class="header-block">
            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path></svg>
            II. AI-Assisted Decision Making
        </div>
        """, unsafe_allow_html=True)
        
        init1, init2, init3 = st.session_state.initial_top
        same_with_ai = init1 == ai_lbl

        if same_with_ai:
            st.success(f"Your initial diagnosis matches the AI recommendation: {init1}")
            with st.form(f"final_decision_form_{idx}"):
                final_conf = st.slider(
                    "Final Diagnosis Confidence (Score 1-10)",
                    1, 10, st.session_state.initial_conf,
                    key=f"cf_{idx}",
                    help="Rate your confidence after reviewing AI alignment"
                )
                
                col_back, col_submit = st.columns([1, 3])
                with col_back:
                    back_btn = st.form_submit_button("Back", help="Return to modify initial diagnosis")
                with col_submit:
                    submit_final = st.form_submit_button("Confirm & Next Case")

                if back_btn:
                    st.session_state.show_ai = False
                    st.rerun()

                if submit_final:
                    t_post = round(time.time() - st.session_state.question_start, 2)
                    gain = final_conf - st.session_state.initial_conf
                    ini_ok = (init1 == truth)
                    fin_ok = (init1 == truth)
                    use_ai = 0

                    final_list = deduplicate_preserve_order([init1, init2, init3])
                    while len(final_list) < 3:
                        final_list.append("N/A")
                    final_list = final_list[:3]
                    final1, final2, final3 = final_list
                    final4 = "N/A"

                    is_final_top3_correct = truth in [final1, final2, final3]
                    is_final_top4_correct = truth in [final1, final2, final3, final4]

                    if ini_ok and fin_ok:
                        path, misled, rescued = "Consistent & Maintained", False, False
                    else:
                        path, misled, rescued = "Persisted Incorrect", False, False

                    result = {
                        **st.session_state.doctor_info,
                        "image_id": img_id,
                        "true_label": truth,
                        "ai_label": ai_lbl,
                        "ai_is_correct": ai_ok,
                        "initial_top1": init1,
                        "initial_top2": init2,
                        "initial_top3": init3,
                        "initial_confidence": st.session_state.initial_conf,
                        "is_initial_top1_correct": ini_ok,
                        "is_initial_top3_correct": truth in [init1, init2, init3],
                        "interaction_type": "Consistent",
                        "action_taken": "No Action Needed (Initial matches AI)",
                        "use_ai": use_ai,
                        "final_top1": final1,
                        "final_top2": final2,
                        "final_top3": final3,
                        "final_top4": final4,
                        "is_final_top1_correct": fin_ok,
                        "is_final_top3_correct": is_final_top3_correct,
                        "is_final_top4_correct": is_final_top4_correct,
                        "final_confidence": final_conf,
                        "confidence_gain": gain,
                        "decision_path": path,
                        "is_misled": misled,
                        "is_rescued": rescued,
                        "time_baseline": st.session_state.time_baseline,
                        "time_post_ai": t_post,
                        "submit_time": get_cst_time()
                    }

                    st.session_state.user_results.append(result)
                    if img_id not in st.session_state.answered_image_ids:
                        st.session_state.answered_image_ids.append(img_id)

                    reset_test_state()
                    st.session_state.current_idx += 1
                    
                    progress_data = {
                        "current_idx": st.session_state.current_idx,
                        "answered_image_ids": st.session_state.answered_image_ids,
                        "user_results": st.session_state.user_results,
                        "doctor_info": st.session_state.doctor_info,
                        "doctor_id": st.session_state.doctor_id,
                        "step": st.session_state.step
                    }
                    st.components.v1.html(f"""
                        <script>
                            try {{
                                sessionStorage.setItem("skin_survey_progress", JSON.stringify({json.dumps(progress_data)}));
                            }} catch(e) {{}}
                        </script>
                    """, height=0)

                    st.rerun()

        else:
            st.markdown(f"""
            <div class="custom-warning-box">
                <b>⚠️ Your diagnosis differs from the AI recommendation.</b><br>
                Yours: <b>{init1}</b> &nbsp;|&nbsp; AI suggests: <b>{ai_lbl}</b>
            </div>
            """, unsafe_allow_html=True)
            
            ai_in_top3 = ai_lbl in [init1, init2, init3]

            with st.form(f"final_decision_form_{idx}"):
                if not ai_in_top3:
                    act = st.radio(
                        "AI suggestion is not in your top three diagnoses. Your choice is:",
                        ["Maintain Original Diagnosis", "Replace as Primary (Top 1)", "Add as 4th Diagnosis"],
                        key=f"act_{idx}",
                        help="Choose how to incorporate or override with the AI suggestion"
                    )
                else:
                    act = st.radio(
                        "Final Decision Choice",
                        ["Maintain Original Diagnosis", "Replace as Primary (Top 1)"],
                        key=f"act_{idx}",
                        help="Choose whether to adopt AI suggestion already present in top differentials"
                    )

                final4 = "N/A"
                use_ai = 0

                if act == "Maintain Original Diagnosis":
                    final_list = deduplicate_preserve_order([init1, init2, init3])
                    while len(final_list) < 3:
                        final_list.append("N/A")
                    final_list = final_list[:3]
                    final1, final2, final3 = final_list

                elif act == "Replace as Primary (Top 1)":
                    temp = [ai_lbl, init1, init2, init3]
                    final_list = deduplicate_preserve_order(temp)
                    while len(final_list) < 3:
                        final_list.append("N/A")
                    final_list = final_list[:3]
                    final1, final2, final3 = final_list
                    use_ai = 1

                elif act == "Add as 4th Diagnosis":
                    final_list = deduplicate_preserve_order([init1, init2, init3])
                    while len(final_list) < 3:
                        final_list.append("N/A")
                    final_list = final_list[:3]
                    final1, final2, final3 = final_list
                    final4 = ai_lbl
                    use_ai = 1

                else:
                    final_list = deduplicate_preserve_order([init1, init2, init3])
                    while len(final_list) < 3:
                        final_list.append("N/A")
                    final_list = final_list[:3]
                    final1, final2, final3 = final_list

                is_final_top1_correct = (final1 == truth)
                is_final_top3_correct = truth in [final1, final2, final3]
                is_final_top4_correct = truth in [final1, final2, final3, final4]

                final_conf = st.slider(
                    "Final Diagnosis Confidence (Score 1-10)",
                    1, 10, st.session_state.initial_conf,
                    key=f"cf_{idx}",
                    help="Final confidence score after evaluating the AI recommendation"
                )

                col_back, col_submit = st.columns([1, 3])
                with col_back:
                    back_btn = st.form_submit_button("Back", help="Return to modify initial diagnosis")
                with col_submit:
                    submit_final = st.form_submit_button("Confirm & Next Case")

                if back_btn:
                    st.session_state.show_ai = False
                    st.rerun()

                if submit_final:
                    t_post = round(time.time() - st.session_state.question_start, 2)
                    gain = final_conf - st.session_state.initial_conf
                    ini_ok = (init1 == truth)
                    fin_ok = is_final_top1_correct

                    if ini_ok and not fin_ok:
                        path, misled, rescued = "Misled", True, False
                    elif not ini_ok and fin_ok:
                        path, misled, rescued = "Rescued", False, True
                    elif ini_ok and fin_ok:
                        path, misled, rescued = "Consistent & Maintained", False, False
                    else:
                        path, misled, rescued = "Persisted Incorrect", False, False

                    result = {
                        **st.session_state.doctor_info,
                        "image_id": img_id,
                        "true_label": truth,
                        "ai_label": ai_lbl,
                        "ai_is_correct": ai_ok,
                        "initial_top1": init1,
                        "initial_top2": init2,
                        "initial_top3": init3,
                        "initial_confidence": st.session_state.initial_conf,
                        "is_initial_top1_correct": ini_ok,
                        "is_initial_top3_correct": truth in [init1, init2, init3],
                        "ai_in_initial_top3": ai_in_top3,
                        "interaction_type": "Conflict",
                        "action_taken": act,
                        "use_ai": use_ai,
                        "final_top1": final1,
                        "final_top2": final2,
                        "final_top3": final3,
                        "final_top4": final4,
                        "is_final_top1_correct": fin_ok,
                        "is_final_top3_correct": is_final_top3_correct,
                        "is_final_top4_correct": is_final_top4_correct,
                        "final_confidence": final_conf,
                        "confidence_gain": gain,
                        "decision_path": path,
                        "is_misled": misled,
                        "is_rescued": rescued,
                        "time_baseline": st.session_state.time_baseline,
                        "time_post_ai": t_post,
                        "submit_time": get_cst_time()
                    }

                    st.session_state.user_results.append(result)
                    if img_id not in st.session_state.answered_image_ids:
                        st.session_state.answered_image_ids.append(img_id)

                    reset_test_state()
                    st.session_state.current_idx += 1

                    progress_data = {
                        "current_idx": st.session_state.current_idx,
                        "answered_image_ids": st.session_state.answered_image_ids,
                        "user_results": st.session_state.user_results,
                        "doctor_info": st.session_state.doctor_info,
                        "doctor_id": st.session_state.doctor_id,
                        "step": st.session_state.step
                    }
                    st.components.v1.html(f"""
                        <script>
                            try {{
                                sessionStorage.setItem("skin_survey_progress", JSON.stringify({json.dumps(progress_data)}));
                            }} catch(e) {{}}
                        </script>
                    """, height=0)

                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# === Result Page ===
def result_step():
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown("# Test Completed")
    st.success(f"Your Test ID: {st.session_state.doctor_id}")
    st.info("All data has been successfully written to Google Sheets. You can check the complete records in the spreadsheet.")
    st.markdown('</div>', unsafe_allow_html=True)

    if len(st.session_state.user_results) > 0:
        df = pd.DataFrame(st.session_state.user_results)

        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown("### Diagnostic Accuracy Comparison")
        initial_acc = df["is_initial_top1_correct"].mean() * 100
        final_acc = df["is_final_top1_correct"].mean() * 100

        acc_data = pd.DataFrame({
            "Accuracy (%)": [initial_acc, final_acc]
        }, index=["Initial Diagnosis (Without AI)", "Final Diagnosis (AI-Assisted)"])

        st.bar_chart(acc_data, color="#3498db", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown("### AI Adoption Effectiveness Analysis")
        ai_used = df[df["use_ai"] == 1]
        ai_not_used = df[df["use_ai"] == 0]

        ai_used_acc = ai_used["is_final_top1_correct"].mean() * 100 if len(ai_used) > 0 else 0
        ai_not_used_acc = ai_not_used["is_final_top1_correct"].mean() * 100 if len(ai_not_used) > 0 else 0

        ai_data = pd.DataFrame({
            "Accuracy (%)": [ai_used_acc, ai_not_used_acc]
        }, index=["Adopted AI Recommendation", "Did Not Adopt AI Recommendation (Including initial match with AI)"])

        st.bar_chart(ai_data, color="#e74c3c", width="stretch")
        st.caption(f"Adopted AI: {len(ai_used)} questions | Did not adopt AI: {len(ai_not_used)} questions")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown("### Summary of Core Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Initial Accuracy", f"{initial_acc:.1f}%")
        with col2:
            st.metric("Final Accuracy", f"{final_acc:.1f}%", delta=f"{final_acc-initial_acc:.1f}%")
        with col3:
            st.metric("AI Adoptions", len(ai_used))
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Restart Test", type="primary"):
        init_session_state()
        st.components.v1.html("""
            <script>
                try {
                    sessionStorage.removeItem("skin_survey_progress");
                } catch(e) {}
            </script>
        """, height=0)
        st.session_state.step = "profile"
        st.rerun()

# === Main Function ===
def main():
    init_session_state()

    if not st.session_state.get("checked_storage", False):
        st.session_state.checked_storage = True
        st.components.v1.html("""
            <script>
                try {
                    const saved = sessionStorage.getItem("skin_survey_progress");
                    if (saved) {
                    }
                } catch(e) {}
            </script>
        """, height=0)

    step = st.session_state.step
    if step == "profile":
        profile_step()
    elif step == "test":
        test_step()
    elif step == "result":
        result_step()

if __name__ == "__main__":
    main()
