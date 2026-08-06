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

# Global CSS styles, pure white bottom mask, mobile keyboard optimization, and button styling
st.markdown("""
<style>
/* Set page background to pure white */
.stApp {
    background-color: #FFFFFF !important;
}

/* Pure white fixed bottom mask to cover the residual Streamlit banner in WeChat */
.streamlit-footer-mask {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 45px;
    background-color: #FFFFFF;
    z-index: 999999;
    pointer-events: none;
}

/* Disable dropdown built-in search input and mobile keyboard popup */
div[data-baseweb="select"] input {
    caret-color: transparent !important;
}

/* Global styling for all submission and primary action buttons */
div.stButton > button, div.stFormSubmitButton > button {
    background-color: #E63946 !important;
    color: #FFFFFF !important;
    border-color: #E63946 !important;
    font-weight: bold !important;
    width: 100% !important;
    border-radius: 6px !important;
}

div.stButton > button:hover, div.stFormSubmitButton > button:hover {
    background-color: #D62828 !important;
    color: #FFFFFF !important;
    border-color: #D62828 !important;
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

# Disease Labels
DISEASE_LABELS = {
    "MEL": "Melanoma", "NV": "Nevus (Melanocytic Nevus)", "BCC": "Basal Cell Carcinoma", "AK": "Actinic Keratosis",
    "BKL": "Benign Keratosis (e.g., Seborrheic Keratosis)", "DF": "Dermatofibroma", "VASC": "Vascular Lesion", "SCC": "Squamous Cell Carcinoma",
    "Vitiligo": "Vitiligo", "Pityrasis-Alba": "Pityriasis Alba", "Psoriasis": "Psoriasis", "UNK": "Unknown Category"
}
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
        "initial_top": ["Please Select", "N/A", "N/A"],
        "initial_conf": 5,
        "final_top1": "", "final_top2": "", "final_top3": "", "final_top4": "",
        "final_conf": 5,
        "question_start": None,
        "time_baseline": 0,
        "doctor_id": "",
        "ai_same_as_initial": False,
        "answered_image_ids": [],
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

# === Random Case Sampling Rule Optimization (Excludes answered cases, ensures AI correct/incorrect ratio meets 6-7:3-4 requirement) ===
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
            st.success(f"✅ Successfully saved {len(rows)} records")
            return True
        except Exception as e:
            st.error(f"❌ Write failed: {str(e)}")
            return False

# === Single Question State Reset ===
def reset_test_state():
    st.session_state.show_ai = False
    st.session_state.initial_top = ["Please Select", "N/A", "N/A"]
    st.session_state.initial_conf = 5
    st.session_state.final_top1 = ""
    st.session_state.final_top2 = ""
    st.session_state.final_top3 = ""
    st.session_state.final_top4 = "N/A"
    st.session_state.final_conf = 5
    st.session_state.time_baseline = 0
    st.session_state.ai_same_as_initial = False
    st.session_state.question_start = None

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

# === Doctor Profile Page ===
def profile_step():
    st.title("🩺 AI-Assisted Dermatological Diagnosis Research Survey")
    st.markdown("""
Dear Doctor:
Thank you for taking time out of your busy clinical schedule to participate in this survey!
This test consists of 10 multiple-choice questions and takes about 3-7 minutes to complete. Every careful judgment and genuine feedback from you carries great value and responsibility for medical research.
We will properly and strictly anonymize all your data, allowing your professional experience to make a greater impact.
Once again, our sincere gratitude to you, and we wish you all the best!
""")
    st.subheader("Step 1: Basic Information Collection (Anonymous)")
    with st.form("profile_form"):
        hospital_level = st.selectbox(
            "Hospital Level",
            ["Tertiary Grade A Hospital Specialist", "Secondary Hospital Specialist", "Community Hospital / Intern"],
            help="Please select your practicing hospital level"
        )
        work_years = st.selectbox(
            "Years of Dermatological Practice",
            ["<=5 years", "5-10 years", "10-15 years", ">15 years", "No Clinical Experience (Intern)"],
            help="Please select your years of experience in dermatology"
        )
        daily_patients = st.selectbox(
            "Average Daily Dermatological Patients",
            ["<=15 cases", "15-30 cases", ">30 cases", "No Outpatient Experience"],
            help="Please select your average daily patient volume range"
        )
        prior_ai_trust = st.selectbox(
            "Initial Trust in AI-Assisted Diagnosis (Score 1-10)",
            options=list(range(1, 11)),
            index=4,
            help="1: Completely distrust, 10: Completely trust"
        )

        submit_btn = st.form_submit_button("✅ Submit Information and Start Test")

    if submit_btn:
        prefix = "A" if "Tertiary" in hospital_level else "B" if "Secondary" in hospital_level else "C"
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

    st.title(f"📷 Case Diagnosis - Question {idx+1} of {TEST_COUNT}")
    st.progress((idx+1)/TEST_COUNT)

    st.subheader("Lesion Image")
    img_url = get_image_url_cached(img_id)
    compressed_img = compress_image(img_url)
    st.image(compressed_img, use_container_width=True)

    st.markdown("### I. Independent Diagnosis")
    with st.form(f"initial_diagnosis_form_{idx}"):
        t1 = st.selectbox(
            "Primary Diagnosis",
            ["Please Select"] + ALL_CLASSES,
            key=f"t1_{idx}",
            help="Please select your most likely diagnosis (Required)"
        )
        t2_opt = ["N/A"] + [x for x in ALL_CLASSES if x != t1]
        t2 = st.selectbox("Secondary Diagnosis", t2_opt, key=f"t2_{idx}")
        t3_opt = ["N/A"] + [x for x in ALL_CLASSES if x not in [t1, t2]]
        t3 = st.selectbox("Tertiary Diagnosis", t3_opt, key=f"t3_{idx}")
        conf_i = st.slider(
            "Confidence in This Diagnosis (Score 1-10)",
            1, 10, 5,
            key=f"ci_{idx}",
            help="1: Completely uncertain, 10: Completely certain"
        )

        submit_initial = st.form_submit_button("🔍 Submit Diagnosis & View AI Recommendation")
        if submit_initial:
            if t1 == "Please Select":
                st.error("Please select at least the primary diagnosis before submitting")
            else:
                time_baseline = round(time.time() - st.session_state.question_start, 2)
                st.session_state.time_baseline = time_baseline

                st.session_state.initial_top = [t1, t2, t3]
                st.session_state.initial_conf = conf_i
                st.session_state.ai_suggestion = {"label": ai_lbl}
                st.session_state.ai_same_as_initial = (t1 == ai_lbl)
                st.session_state.show_ai = True
                st.rerun()

    if st.session_state.show_ai:
        st.markdown("### II. AI-Assisted Decision Making")
        st.info(f"🤖 AI Diagnostic Recommendation: **{ai_lbl}**")

        init1, init2, init3 = st.session_state.initial_top
        same_with_ai = init1 == ai_lbl

        if same_with_ai:
            st.success(f"✅ Your initial diagnosis matches the AI recommendation: {init1}")
            with st.form(f"final_decision_form_{idx}"):
                final_conf = st.slider(
                    "Final Diagnosis Confidence (Score 1-10)",
                    1, 10, st.session_state.initial_conf,
                    key=f"cf_{idx}"
                )
                submit_final = st.form_submit_button("✅ Confirm Final Diagnosis & Proceed to Next Question")
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
            st.warning(f"⚠️ Your initial diagnosis ({init1}) differs from the AI recommendation ({ai_lbl})")
            ai_in_top3 = ai_lbl in [init1, init2, init3]

            with st.form(f"final_decision_form_{idx}"):
                if not ai_in_top3:
                    act = st.radio(
                        "AI suggestion is not in your top three diagnoses. Your choice is:",
                        ["Maintain Original Diagnosis", "Replace as Primary (Top 1)", "Add as 4th Diagnosis"],
                        key=f"act_{idx}"
                    )
                else:
                    act = st.radio(
                        "Final Decision Choice",
                        ["Maintain Original Diagnosis", "Replace as Primary (Top 1)"],
                        key=f"act_{idx}"
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
                    key=f"cf_{idx}"
                )

                submit_final = st.form_submit_button("✅ Confirm Final Diagnosis & Proceed to Next Question")
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
                        "is_final_top1_correct": is_final_top1_correct,
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

# === Result Page ===
def result_step():
    st.title("📊 Test Completed")
    st.success(f"Your Test ID: {st.session_state.doctor_id}")
    st.info("All data has been successfully written to Google Sheets. You can check the complete records in the spreadsheet.")

    if len(st.session_state.user_results) > 0:
        df = pd.DataFrame(st.session_state.user_results)

        st.subheader("📈 Diagnostic Accuracy Comparison")
        initial_acc = df["is_initial_top1_correct"].mean() * 100
        final_acc = df["is_final_top1_correct"].mean() * 100

        acc_data = pd.DataFrame({
            "Accuracy (%)": [initial_acc, final_acc]
        }, index=["Initial Diagnosis (Without AI)", "Final Diagnosis (AI-Assisted)"])

        st.bar_chart(acc_data, color="#3498db", width="stretch")

        st.subheader("💡 AI Adoption Effectiveness Analysis")
        ai_used = df[df["use_ai"] == 1]
        ai_not_used = df[df["use_ai"] == 0]

        ai_used_acc = ai_used["is_final_top1_correct"].mean() * 100 if len(ai_used) > 0 else 0
        ai_not_used_acc = ai_not_used["is_final_top1_correct"].mean() * 100 if len(ai_not_used) > 0 else 0

        ai_data = pd.DataFrame({
            "Accuracy (%)": [ai_used_acc, ai_not_used_acc]
        }, index=["Adopted AI Recommendation", "Did Not Adopt AI Recommendation (Including initial match with AI)"])

        st.bar_chart(ai_data, color="#e74c3c", width="stretch")
        st.caption(f"Adopted AI: {len(ai_used)} questions | Did not adopt AI: {len(ai_not_used)} questions")

        st.subheader("📊 Summary of Core Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Initial Accuracy", f"{initial_acc:.1f}%")
        with col2:
            st.metric("Final Accuracy", f"{final_acc:.1f}%", delta=f"{final_acc-initial_acc:.1f}%")
        with col3:
            st.metric("AI Adoptions", len(ai_used))

    if st.button("🔄 Restart Test", type="primary"):
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
