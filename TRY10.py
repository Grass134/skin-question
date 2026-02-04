import streamlit as st
import pandas as pd
import os
import uuid
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
import io
import json  
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import re  
import random
from io import BytesIO

# === 核心配置 ===
st.set_option('client.showErrorDetails', True)  
st.set_page_config(page_title="皮肤病AI辅助诊断研究", page_icon="🩺", layout="centered")

# 性能优化配置
REQUEST_TIMEOUT = 1  # 图片请求超时1秒
CACHE_TTL = 3600     # 缓存有效期1小时
IMAGE_COMPRESS_WIDTH = 600  # 手机端更适合的图片宽度
IMAGE_QUALITY = 85     # 图片压缩质量（1-100）

# 你的GitHub信息
GITHUB_USERNAME = "Grass134"
GITHUB_REPO = "skin-question"
GOLD_TXT = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/boosted_final_detail4.UTF-8.txt"

# ========== Google Sheets配置 ==========
GOOGLE_SHEET_NAME = "皮肤诊断数据"  
LOCAL_GOOGLE_CREDENTIALS_FILE = "google_credentials.json"

# GitHub图片文件夹配置
GITHUB_IMAGE_FOLDER = "experiment_pool"
GITHUB_BRANCH = "main"

# 疾病标签映射
DISEASE_LABELS = {
    "MEL": "黑色素瘤", "NV": "痣（色素痣）", "BCC": "基底细胞癌", "AK": "光化性角化病",
    "BKL": "良性角化病（脂溢性角化等）", "DF": "皮肤纤维瘤", "VASC": "血管病变", "SCC": "鳞状细胞癌",
    "Vitiligo": "白癜风", "Pityrasis-Alba": "白色糠疹", "Psoriasis": "银屑病", "UNK": "未知类别"
}
ALL_CLASSES = list(DISEASE_LABELS.values())
TEST_COUNT = 10

# === 性能优化：全局缓存Google Sheets连接（延迟初始化） ===
@st.cache_resource(ttl=CACHE_TTL, show_spinner=False)
def init_google_sheets_once():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        # 优先从Streamlit Secrets读取
        try:
            creds_dict = dict(st.secrets["GOOGLE_CREDENTIALS"])
            if "private_key" in creds_dict:
                creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            
            required_fields = ["type", "project_id", "private_key", "client_email"]
            missing_fields = [f for f in required_fields if f not in creds_dict]
            if missing_fields:
                return None, f"❌ 密钥缺少必要字段：{missing_fields}"
            
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        except KeyError:
            if not os.path.exists(LOCAL_GOOGLE_CREDENTIALS_FILE):
                return None, "❌ 本地凭证文件不存在"
            creds = ServiceAccountCredentials.from_json_keyfile_name(LOCAL_GOOGLE_CREDENTIALS_FILE, scope)
        
        client = gspread.authorize(creds)
        try:
            sheet = client.open(GOOGLE_SHEET_NAME).sheet1
            # 初始化表头（仅一次）
            headers = sheet.row_values(1)
            required_headers = [
                "doctor_id", "hospital_level", "work_years", "daily_patients", "prior_ai_trust",
                "image_id", "true_label", "ai_label", "ai_is_correct", "initial_top1", "initial_top2",
                "initial_top3", "initial_confidence", "is_initial_top1_correct", "is_initial_top3_correct",
                "interaction_type", "action_taken", "use_ai", "final_top1", "final_top2", "final_top3",
                "final_top4", "is_final_top1_correct", "is_final_top3_correct", "is_final_top4_correct",
                "final_confidence", "confidence_gain", "decision_path", "is_misled", "is_rescued",
                "time_baseline", "time_post_ai", "submit_time"
            ]
            if not headers or len(headers) != len(required_headers):
                sheet.clear()
                sheet.append_row(required_headers)
            return sheet, None
        except gspread.exceptions.SpreadsheetNotFound:
            return None, f"❌ 未找到Google表格：{GOOGLE_SHEET_NAME}"
    except Exception as e:
        return None, f"⚠️ Google Sheets初始化失败：{str(e)}"

# === 会话状态初始化（延迟加载Google Sheets） ===
def init_session_state():
    default_states = {
        "step": "profile",
        "current_idx": 0,
        "show_ai": False,
        "user_results": [],  # 本地临时存储
        "test_set": None,
        "doctor_info": {},
        "ai_suggestion": {},
        "initial_top": ["请选择", "无", "无"],
        "initial_conf": 5,
        "final_top1": "",
        "final_top2": "",
        "final_top3": "",
        "final_top4": "",
        "final_conf": 5,
        "question_start": 0,
        "time_baseline": 0,
        "doctor_id": "",
        "ai_same_as_initial": False,
        "gs_sheet": None,  # 延迟初始化
        "gs_error": None
    }
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

# === 性能优化：缓存测试数据（避免st.stop阻塞） ===
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_gold_data_cached():
    try:
        response = requests.get(GOLD_TXT, timeout=5)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), encoding="utf-8")
        
        required_cols = ["image_id", "Top1_预测", "真实病名"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None, f"⚠️ 缺失必要字段：{', '.join(missing_cols)}"
        
        df["true_cn"] = df["真实病名"].map(DISEASE_LABELS).fillna("未知")
        df["ai_cn"] = df["Top1_预测"].map(DISEASE_LABELS).fillna("未知")
        df["ai_correct"] = df["true_cn"] == df["ai_cn"]
        df = df[df["true_cn"] != "未知"]
        df = df[df["ai_cn"] != "未知"]
        return df, None
    except Exception as e:
        return None, f"⚠️ 测试数据加载失败：{str(e)}"

def load_balanced_test_set(df):
    ai_correct = df[df["ai_correct"]]
    ai_incorrect = df[~df["ai_correct"]]
    correct_sample = ai_correct.sample(min(6, len(ai_correct)))
    incorrect_sample = ai_incorrect.sample(min(4, len(incorrect_sample)))
    if len(correct_sample) < 6:
        correct_sample = pd.concat([correct_sample, ai_correct.sample(6 - len(correct_sample))])
    if len(incorrect_sample) < 4:
        incorrect_sample = pd.concat([incorrect_sample, ai_incorrect.sample(4 - len(incorrect_sample))])
    test_set = pd.concat([correct_sample, incorrect_sample]).sample(frac=1).reset_index(drop=True)
    return test_set.head(TEST_COUNT)

# === 最终批量保存（移除自动保存） ===
def save_results_batch():
    if st.session_state.gs_sheet is None:
        st.error(st.session_state.gs_error)
        return
    if len(st.session_state.user_results) == 0:
        return
    
    try:
        rows = []
        for result in st.session_state.user_results:
            row_data = [
                result["doctor_id"], result["hospital_level"], result["work_years"],
                result["daily_patients"], result["prior_ai_trust"], result["image_id"],
                result["true_label"], result["ai_label"], result["ai_is_correct"],
                result["initial_top1"], result["initial_top2"], result["initial_top3"],
                result["initial_confidence"], result["is_initial_top1_correct"],
                result["is_initial_top3_correct"], result["interaction_type"],
                result["action_taken"], result["use_ai"], result["final_top1"],
                result["final_top2"], result["final_top3"], result["final_top4"],
                result["is_final_top1_correct"], result["is_final_top3_correct"],
                result["is_final_top4_correct"], result["final_confidence"],
                result["confidence_gain"], result["decision_path"], result["is_misled"],
                result["is_rescued"], result["time_baseline"], result["time_post_ai"],
                result["submit_time"]
            ]
            rows.append(row_data)
        
        with st.spinner("💾 正在保存所有数据..."):
            st.session_state.gs_sheet.append_rows(rows)
        st.toast(f"✅ 成功保存{len(rows)}条数据到Google Sheets", icon="✅")
    except Exception as e:
        st.error(f"❌ 数据保存失败：{str(e)}")

# === 重置答题状态 ===
def reset_test_state():
    st.session_state.show_ai = False
    st.session_state.initial_top = ["请选择", "无", "无"]
    st.session_state.initial_conf = 5
    st.session_state.final_top1 = ""
    st.session_state.final_top2 = ""
    st.session_state.final_top3 = ""
    st.session_state.final_top4 = ""
    st.session_state.final_conf = 5
    st.session_state.time_baseline = 0
    st.session_state.ai_same_as_initial = False

# === 图片压缩函数 ===
def compress_image(image_url):
    try:
        response = requests.get(image_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        
        w, h = img.size
        ratio = IMAGE_COMPRESS_WIDTH / w
        new_height = int(h * ratio)
        img = img.resize((IMAGE_COMPRESS_WIDTH, new_height), Image.Resampling.LANCZOS)
        
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=IMAGE_QUALITY, optimize=True)
        buf.seek(0)
        return buf
    except Exception as e:
        st.toast(f"⚠️ 图片压缩失败：{str(e)[:20]}", icon="⚠️")
        response = requests.get(image_url, timeout=REQUEST_TIMEOUT)
        return BytesIO(response.content)

# === 性能优化：简化图片加载 + 压缩 ===
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_image_url_cached(image_id):
    possible_paths = []
    image_id_clean = re.sub(r'\.(jpg|png)$', '', image_id)
    
    if 'pityriasis-alba' in image_id_clean.lower() or 'pityrasis-alba' in image_id_clean.lower():
        possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/pityriasis-alba-images/{image_id_clean}.jpg")
    elif 'psoriasis' in image_id_clean.lower():
        possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/PSORIASIS/{image_id_clean}.jpg")
    elif 'vitiligo' in image_id_clean.lower():
        possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/vitiligo/{image_id_clean}.jpg")
    elif image_id_clean.startswith('ISIC_'):
        possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/{image_id_clean}.jpg")
    else:
        possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/{image_id_clean}.jpg")

    for path in possible_paths[:3]:
        raw_url = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{path}"
        try:
            response = requests.head(raw_url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                return raw_url
        except:
            continue

    isic_fallback = ["ISIC_0034334", "ISIC_0034402", "ISIC_0034411"]
    return f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_IMAGE_FOLDER}/{random.choice(isic_fallback)}.jpg"

# === 医生信息采集（适配手机） ===
def profile_step():
    st.title("🩺 皮肤病AI辅助诊断研究")
    st.subheader("第一步：医生信息采集（匿名）")
    
    with st.form("profile_form", clear_on_submit=True):
        hospital_level = st.selectbox(
            "1. 医院等级", 
            ["三甲医院专科医生", "二级医院专科医生", "社区医院医生（含实习生）"]
        )
        work_years = st.selectbox(
            "2. 工作年限", 
            ["≤5年", "5-15年", ">15年", "无经验（实习生）"]
        )
        daily_patients = st.selectbox(
            "3. 日均接诊量", 
            ["≤30例", "30-50例", ">50例", "无接诊经验"]
        )
        prior_ai_trust = st.slider(
            "4. 对AI的信任度（1-5）", 
            1, 5, 3, help="1=极不信任，5=极度信任"
        )
        
        if st.form_submit_button("✅ 提交并开始测试", type="primary"):
            # 修复KeyError：字典键和选项文本完全匹配
            level_prefix = {
                "三甲医院专科医生": "A", 
                "二级医院专科医生": "B", 
                "社区医院医生（含实习生）": "C"
            }
            st.session_state.doctor_id = f"{level_prefix[hospital_level]}_DR_{uuid.uuid4().hex[:6].upper()}"
            
            st.session_state.doctor_info = {
                "doctor_id": st.session_state.doctor_id,
                "hospital_level": hospital_level,
                "work_years": work_years,
                "daily_patients": daily_patients,
                "prior_ai_trust": prior_ai_trust
            }
            
            # 加载测试数据（处理异常）
            with st.spinner("加载测试数据..."):
                gold_df, error = load_gold_data_cached()
                if gold_df is None:
                    st.error(error)
                    st.stop()
            
            if ">15年" in work_years:
                more_trap = gold_df[~gold_df["ai_correct"]].sample(min(2, len(gold_df[~gold_df["ai_correct"]])))
                gold_df = pd.concat([gold_df, more_trap]).drop_duplicates()
            st.session_state.test_set = load_balanced_test_set(gold_df)
            st.session_state.step = "test"
            st.rerun()

# === 答题流程（适配手机） ===
def test_step():
    if st.session_state.test_set is None:
        st.error("⚠️ 测试数据未加载")
        if st.button("🔄 返回重新开始", type="primary"):
            init_session_state()
            st.session_state.step = "profile"
            st.rerun()
        return
    
    idx = st.session_state.current_idx
    test_set = st.session_state.test_set
    
    if idx >= len(test_set):
        # 初始化Google Sheets（延迟到保存时）
        with st.spinner("初始化数据存储..."):
            sheet, error = init_google_sheets_once()
            st.session_state.gs_sheet = sheet
            st.session_state.gs_error = error
        save_results_batch()  # 完成后一次性保存
        st.session_state.step = "result"
        st.rerun()
    
    current_data = test_set.iloc[idx]
    image_id = current_data["image_id"]
    true_label = current_data["true_cn"]
    ai_label = current_data["ai_cn"]
    ai_is_correct = (ai_label == true_label)
    
    st.title(f"📝 测试题 {idx + 1}/{TEST_COUNT}")
    st.progress((idx + 1) / TEST_COUNT, text=f"进度：{idx + 1}/{TEST_COUNT}")
    st.subheader("皮肤镜图像")
    
    image_url = get_image_url_cached(image_id)
    compressed_img = compress_image(image_url)
    try:
        st.image(compressed_img, use_container_width=True, caption=f"图片ID：{image_id}")
    except:
        st.image("https://via.placeholder.com/600x400?text=皮肤镜示例图", use_container_width=True)
    
    st.markdown("### 第一阶段：独立诊断")
    top1 = st.selectbox("首选 (Top-1) [必填]", ["请选择"] + ALL_CLASSES, key=f"t1_{idx}")
    top2_options = ["无"] + [c for c in ALL_CLASSES if c != top1]
    top2 = st.selectbox("次选 (Top-2) [可选]", top2_options, key=f"t2_{idx}", index=0)
    top3_options = ["无"] + [c for c in ALL_CLASSES if c not in [top1, top2]]
    top3 = st.selectbox("备选 (Top-3) [可选]", top3_options, key=f"t3_{idx}", index=0)
    conf_init = st.slider("初始信心（1-10）", 1, 10, 5, key=f"c1_{idx}")
    
    is_valid = top1 != "请选择"
    if not st.session_state.show_ai:
        if st.button("🔍 获取AI辅助建议", disabled=not is_valid, type="secondary"):
            st.session_state.initial_top = [top1, top2, top3]
            st.session_state.initial_conf = conf_init
            st.session_state.ai_suggestion = {"label": ai_label, "is_correct": ai_is_correct}
            st.session_state.ai_same_as_initial = (top1 == ai_label)
            st.session_state.question_start = time.time()
            st.session_state.time_baseline = round(time.time() - st.session_state.question_start, 2)
            st.session_state.show_ai = True
            st.rerun()
        if not is_valid:
            st.caption("⚠️ 请先选择Top1诊断结果")
    
    if st.session_state.show_ai:
        st.markdown("### 第二阶段：AI辅助决策")
        ai_sug = st.session_state.ai_suggestion["label"]
        initial_top1 = st.session_state.initial_top[0]
        
        if st.session_state.ai_same_as_initial:
            st.success(f"✅ 你的初始诊断（{initial_top1}）与AI建议（{ai_sug}）一致！")
            
            if st.button("✅ 确认并进入下一题", key=f"btn_{idx}", type="primary"):
                time_post_ai = round(time.time() - st.session_state.question_start, 2)
                is_initial_top1_correct = (initial_top1 == true_label)
                
                result = {
                    "doctor_id": st.session_state.doctor_id,
                    "hospital_level": st.session_state.doctor_info["hospital_level"],
                    "work_years": st.session_state.doctor_info["work_years"],
                    "daily_patients": st.session_state.doctor_info["daily_patients"],
                    "prior_ai_trust": st.session_state.doctor_info["prior_ai_trust"],
                    "image_id": image_id,
                    "true_label": true_label,
                    "ai_label": ai_sug,
                    "ai_is_correct": ai_is_correct,
                    "initial_top1": initial_top1,
                    "initial_top2": st.session_state.initial_top[1],
                    "initial_top3": st.session_state.initial_top[2],
                    "initial_confidence": st.session_state.initial_conf,
                    "is_initial_top1_correct": is_initial_top1_correct,
                    "is_initial_top3_correct": (true_label in st.session_state.initial_top),
                    "interaction_type": "一致",
                    "action_taken": "无需选择（AI与初始一致）",
                    "use_ai": 0,
                    "final_top1": initial_top1,
                    "final_top2": st.session_state.initial_top[1],
                    "final_top3": st.session_state.initial_top[2],
                    "final_top4": "无",
                    "is_final_top1_correct": is_initial_top1_correct,
                    "is_final_top3_correct": (true_label in st.session_state.initial_top),
                    "is_final_top4_correct": (true_label in st.session_state.initial_top),
                    "final_confidence": st.session_state.initial_conf,
                    "confidence_gain": 0,
                    "decision_path": "一致（诊断相同）",
                    "is_misled": False,
                    "is_rescued": False,
                    "time_baseline": st.session_state.time_baseline,
                    "time_post_ai": time_post_ai,
                    "submit_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.session_state.user_results.append(result)
                reset_test_state()
                st.session_state.current_idx = idx + 1
                st.rerun()
        
        else:
            st.warning(f"⚠️ 你的初始诊断（{initial_top1}）与AI建议（{ai_sug}）不一致！")
            action = st.radio(
                "如何处理AI建议？",
                ["坚持原诊断", "替换为AI建议"],
                key=f"act_{idx}"
            )
            
            final_top1 = initial_top1 if action == "坚持原诊断" else ai_sug
            conf_final = st.slider("最终信心（1-10）", 1, 10, st.session_state.initial_conf, key=f"c2_{idx}")
            
            if st.button("✅ 确认并进入下一题", key=f"btn_{idx}", type="primary"):
                time_post_ai = round(time.time() - st.session_state.question_start, 2)
                confidence_gain = conf_final - st.session_state.initial_conf
                is_initial_top1_correct = (initial_top1 == true_label)
                is_final_top1_correct = (final_top1 == true_label)
                use_ai = 1 if action == "替换为AI建议" else 0
                
                decision_path = ""
                is_misled = False
                is_rescued = False
                if is_initial_top1_correct and not is_final_top1_correct:
                    decision_path = "误导（对改错）"
                    is_misled = True
                elif not is_initial_top1_correct and is_final_top1_correct:
                    decision_path = "纠正（错改对）"
                    is_rescued = True
                elif is_initial_top1_correct and is_final_top1_correct:
                    decision_path = "同对（坚持）"
                else:
                    decision_path = "盲从（错改错）"
                
                result = {
                    "doctor_id": st.session_state.doctor_id,
                    "hospital_level": st.session_state.doctor_info["hospital_level"],
                    "work_years": st.session_state.doctor_info["work_years"],
                    "daily_patients": st.session_state.doctor_info["daily_patients"],
                    "prior_ai_trust": st.session_state.doctor_info["prior_ai_trust"],
                    "image_id": image_id,
                    "true_label": true_label,
                    "ai_label": ai_sug,
                    "ai_is_correct": ai_is_correct,
                    "initial_top1": initial_top1,
                    "initial_top2": st.session_state.initial_top[1],
                    "initial_top3": st.session_state.initial_top[2],
                    "initial_confidence": st.session_state.initial_conf,
                    "is_initial_top1_correct": is_initial_top1_correct,
                    "is_initial_top3_correct": (true_label in st.session_state.initial_top),
                    "interaction_type": "冲突",
                    "action_taken": action,
                    "use_ai": use_ai,
                    "final_top1": final_top1,
                    "final_top2": st.session_state.initial_top[1],
                    "final_top3": st.session_state.initial_top[2],
                    "final_top4": "无",
                    "is_final_top1_correct": is_final_top1_correct,
                    "is_final_top3_correct": (true_label in [final_top1, st.session_state.initial_top[1], st.session_state.initial_top[2]]),
                    "is_final_top4_correct": (true_label in [final_top1, st.session_state.initial_top[1], st.session_state.initial_top[2]]),
                    "final_confidence": conf_final,
                    "confidence_gain": confidence_gain,
                    "decision_path": decision_path,
                    "is_misled": is_misled,
                    "is_rescued": is_rescued,
                    "time_baseline": st.session_state.time_baseline,
                    "time_post_ai": time_post_ai,
                    "submit_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.session_state.user_results.append(result)
                reset_test_state()
                st.session_state.current_idx = idx + 1
                st.rerun()

# === 结果展示（适配手机） ===
def result_step():
    st.title("🏁 测试完成！")
    st.success(f"✅ 你的唯一标识ID：{st.session_state.doctor_id}")
    
    if len(st.session_state.user_results) > 0:
        user_df = pd.DataFrame(st.session_state.user_results)
        
        st.subheader("📊 你的诊断表现")
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_acc = user_df["is_initial_top1_correct"].mean() * 100
            st.metric("初始准确率", f"{initial_acc:.1f}%")
        with col2:
            final_acc = user_df["is_final_top1_correct"].mean() * 100
            st.metric("最终准确率", f"{final_acc:.1f}%", delta=f"{final_acc - initial_acc:.1f}%")
        with col3:
            ai_usage = user_df["use_ai"].sum()
            st.metric("采纳AI次数", ai_usage)
        
        st.subheader("📋 答题记录")
        display_df = user_df[["image_id", "true_label", "initial_top1", "final_top1", "ai_label", "decision_path"]]
        display_df.columns = ["图片ID", "真实诊断", "初始诊断", "最终诊断", "AI建议", "决策路径"]
        st.dataframe(display_df, use_container_width=True)
    
    st.button("🔄 重新开始测试", on_click=init_session_state, type="primary")

# === 主函数 ===
def main():
    # 先检查依赖
    missing_deps = []
    try:
        import gspread
    except ImportError:
        missing_deps.append("gspread")
    try:
        import oauth2client
    except ImportError:
        missing_deps.append("oauth2client")
    try:
        from PIL import Image
    except ImportError:
        missing_deps.append("pillow")
    
    if missing_deps:
        st.error(f"⚠️ 缺少依赖库，请运行：pip install {' '.join(missing_deps)}")
        st.stop()
    
    # 确保会话状态初始化
    if "step" not in st.session_state:
        init_session_state()
    
    # 执行对应步骤
    if st.session_state.step == "profile":
        profile_step()
    elif st.session_state.step == "test":
        test_step()
    elif st.session_state.step == "result":
        result_step()

if __name__ == "__main__":
    main()
