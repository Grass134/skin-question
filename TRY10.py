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

# === 核心配置 ===
st.set_option('client.showErrorDetails', True)  
st.set_page_config(page_title="皮肤病AI辅助诊断研究", page_icon="🩺", layout="wide")

# 你的GitHub信息
GITHUB_USERNAME = "Grass134"
GITHUB_REPO = "skin-question"
GOLD_TXT = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/boosted_final_detail4.UTF-8.txt"

# ========== Google Sheets配置（唯一存储方式） ==========
GOOGLE_SHEET_NAME = "皮肤诊断数据"  
LOCAL_GOOGLE_CREDENTIALS_FILE = "google_credentials.json"

# GitHub图片文件夹配置
GITHUB_IMAGE_FOLDER = "experiment_pool"
GITHUB_BRANCH = "main"

# 备用图片池（严格匹配你的重命名规则）
FALLBACK_IMAGE_POOLS = {
    "vitiligo": [f"vitiligo-{str(i).zfill(4)}.jpg" for i in range(1, 500)] + 
                [f"vitiligo-{str(i).zfill(4)}-{j}.jpg" for i in range(1, 500) for j in range(1, 10)],
    "pityriasis-alba": [f"pityriasis-alba-{str(i).zfill(4)}.jpg" for i in range(1, 300)] + 
                       [f"pityriasis-alba-{str(i).zfill(4)}-{j}.jpg" for i in range(1, 300) for j in range(1, 10)],
    "psoriasis": [f"psoriasis-{str(i).zfill(4)}.jpg" for i in range(1, 300)] + 
                 [f"psoriasis-{str(i).zfill(4)}-{j}.jpg" for i in range(1, 300) for j in range(1, 10)],
    "general": [f"skin-image-{str(i).zfill(4)}.jpg" for i in range(1, 500)] + 
               [f"skin-image-{str(i).zfill(4)}-{j}.jpg" for i in range(1, 500) for j in range(1, 10)]
}

# 疾病标签映射
DISEASE_LABELS = {
    "MEL": "黑色素瘤", "NV": "痣（色素痣）", "BCC": "基底细胞癌", "AK": "光化性角化病",
    "BKL": "良性角化病（脂溢性角化等）", "DF": "皮肤纤维瘤", "VASC": "血管病变", "SCC": "鳞状细胞癌",
    "Vitiligo": "白癜风", "Pityrasis-Alba": "白色糠疹", "Psoriasis": "银屑病", "UNK": "未知类别"
}
ALL_CLASSES = list(DISEASE_LABELS.values())
TEST_COUNT = 10

# === 初始化Google Sheets连接（强制唯一存储） ===
def init_google_sheets():
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # 优先从Streamlit Secrets读取（推荐线上部署）
        try:
            creds_dict = dict(st.secrets["GOOGLE_CREDENTIALS"])
            # 修复private_key换行符（防止复制丢失）
            if "private_key" in creds_dict:
                creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            
            # 校验必要字段
            required_fields = ["type", "project_id", "private_key", "client_email"]
            missing_fields = [f for f in required_fields if f not in creds_dict]
            if missing_fields:
                st.error(f"❌ 密钥缺少必要字段：{missing_fields}")
                st.error("请检查Streamlit Secrets中的GOOGLE_CREDENTIALS配置")
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            st.success("✅ 从Streamlit Secrets加载Google凭证成功")
        
        # 本地运行时使用本地凭证文件
        except KeyError:
            st.info("ℹ️ 未检测到Streamlit Secrets，尝试加载本地凭证文件")
            if not os.path.exists(LOCAL_GOOGLE_CREDENTIALS_FILE):
                st.error(f"❌ 本地凭证文件 {LOCAL_GOOGLE_CREDENTIALS_FILE} 不存在")
                raise FileNotFoundError(f"Local credentials file not found")
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                LOCAL_GOOGLE_CREDENTIALS_FILE, scope
            )
            st.success("✅ 从本地文件加载Google凭证成功")
        
        # 连接表格并初始化表头
        client = gspread.authorize(creds)
        try:
            sheet = client.open(GOOGLE_SHEET_NAME).sheet1
            # 检查表头是否存在，不存在则创建
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
                sheet.clear()  # 清空错误表头
                sheet.append_row(required_headers)
                st.success(f"✅ 初始化Google表格表头成功")
            st.success(f"✅ 成功连接Google表格：{GOOGLE_SHEET_NAME}")
            return sheet
        
        except gspread.exceptions.SpreadsheetNotFound:
            st.error(f"❌ 未找到Google表格：{GOOGLE_SHEET_NAME}")
            st.error("请确认表格名称完全一致，且服务账号已获得编辑权限")
            raise
        except Exception as e:
            st.error(f"❌ 连接Google表格失败：{str(e)}")
            raise
    
    except Exception as e:
        st.error(f"⚠️ Google Sheets初始化失败：{str(e)}")
        st.error("❌ 数据无法存储，请修复凭证配置后重试")
        st.stop()  # 强制停止，避免无存储情况下继续运行
        return None

# === 会话状态初始化 ===
def init_session_state():
    default_states = {
        "step": "profile",
        "current_idx": 0,
        "show_ai": False,
        "user_results": [],  # 临时存储答题结果，最终统一提交
        "test_set": None,
        "doctor_info": {},
        "ai_suggestion": {},
        "initial_top": ["请选择", "无", "无"],
        "initial_conf": 5,
        "final_top1": "",
        "final_top2": "",
        "final_top3": "",
        "final_top4": "",
        "final_decision": "",
        "final_conf": 5,
        "question_start": 0,
        "time_baseline": 0,
        "doctor_id": "",
        "ai_same_as_initial": False,
        "gs_sheet": None  # 存储Google Sheets连接对象
    }
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

# === 数据加载 ===
@st.cache_data(ttl=300)
def load_gold_data():
    try:
        response = requests.get(GOLD_TXT, timeout=15)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), encoding="utf-8")
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ 测试数据加载失败：{str(e)}")
        st.error("请检查GitHub链接是否正确")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("⚠️ CSV文件为空，请检查文件内容")
        st.stop()
    
    required_cols = ["image_id", "Top1_预测", "真实病名"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"⚠️ CSV文件缺失必要字段：{', '.join(missing_cols)}")
        st.stop()
    
    df["true_cn"] = df["真实病名"].map(DISEASE_LABELS).fillna("未知")
    df["ai_cn"] = df["Top1_预测"].map(DISEASE_LABELS).fillna("未知")
    df["ai_correct"] = df["true_cn"] == df["ai_cn"]
    df = df[df["true_cn"] != "未知"]
    df = df[df["ai_cn"] != "未知"]
    if len(df) < TEST_COUNT:
        st.error(f"⚠️ 有效测试数据不足{TEST_COUNT}条")
        st.stop()
    return df

def load_balanced_test_set(df):
    ai_correct = df[df["ai_correct"]]
    ai_incorrect = df[~df["ai_correct"]]
    correct_sample = ai_correct.sample(min(6, len(ai_correct)))
    incorrect_sample = ai_incorrect.sample(min(4, len(ai_incorrect)))
    if len(correct_sample) < 6:
        correct_sample = pd.concat([correct_sample, ai_correct.sample(6 - len(correct_sample))])
    if len(incorrect_sample) < 4:
        incorrect_sample = pd.concat([incorrect_sample, ai_incorrect.sample(4 - len(incorrect_sample))])
    test_set = pd.concat([correct_sample, incorrect_sample]).sample(frac=1).reset_index(drop=True)
    return test_set.head(TEST_COUNT)

# === 数据保存（仅Google Sheets，无本地存储） ===
def save_result_to_backend(result):
    # 确保Google Sheets连接已初始化
    if st.session_state.gs_sheet is None:
        st.session_state.gs_sheet = init_google_sheets()
    
    try:
        # 拼接行数据
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
        # 写入Google Sheets
        st.session_state.gs_sheet.append_row(row_data)
        st.toast("✅ 数据已成功保存到Google Sheets", icon="✅")
    except Exception as e:
        st.error(f"❌ 数据保存失败：{str(e)}")
        st.error("请检查网络连接和Google Sheets权限")
        raise  # 保存失败时终止流程，确保数据不丢失

# === 重置答题状态 ===
def reset_test_state():
    st.session_state.show_ai = False
    st.session_state.initial_top = ["请选择", "无", "无"]
    st.session_state.initial_conf = 5
    st.session_state.final_top1 = ""
    st.session_state.final_top2 = ""
    st.session_state.final_top3 = ""
    st.session_state.final_top4 = ""
    st.session_state.final_decision = ""
    st.session_state.final_conf = 5
    st.session_state.time_baseline = 0
    st.session_state.ai_same_as_initial = False

# === 获取备用图片URL ===
def get_fallback_image_url():
    pool_types = list(FALLBACK_IMAGE_POOLS.keys())
    random.shuffle(pool_types)
    
    for pool_type in pool_types:
        image_list = FALLBACK_IMAGE_POOLS[pool_type].copy()
        random.shuffle(image_list)
        
        for image_name in image_list[:50]:
            if pool_type == "pityriasis-alba":
                path = f"{GITHUB_IMAGE_FOLDER}/pityriasis-alba-images/{image_name}"
            elif pool_type == "psoriasis":
                path = f"{GITHUB_IMAGE_FOLDER}/PSORIASIS/{image_name}"
            elif pool_type == "vitiligo":
                path = f"{GITHUB_IMAGE_FOLDER}/vitiligo/{image_name}"
            else:
                path = f"{GITHUB_IMAGE_FOLDER}/{image_name}"
            
            raw_url = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{path}"
            try:
                response = requests.head(raw_url, timeout=2)
                if response.status_code == 200:
                    st.toast(f"ℹ️ 图片加载失败，已替换为{pool_type}备用图片", icon="ℹ️")
                    return raw_url
            except:
                continue
    
    return "https://via.placeholder.com/600x400?text=图片加载失败"

# === 图片加载（匹配重命名规则） ===
def get_github_image_url(image_id):
    possible_paths = []
    image_id_clean = re.sub(r'\.(jpg|png|jpeg|gif|bmp)$', '', image_id)
    
    # 匹配 pityriasis-alba (带i)
    if 'pityriasis-alba' in image_id_clean.lower():
        number_match = re.search(r'(\d{4})', image_id_clean)
        if number_match:
            file_number = number_match.group(1)
            possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/pityriasis-alba-images/pityriasis-alba-{file_number}.jpg")
            for suffix in range(1, 11):
                possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/pityriasis-alba-images/pityriasis-alba-{file_number}-{suffix}.jpg")
    # 兼容旧拼写
    elif 'pityrasis-alba' in image_id_clean.lower():
        number_match = re.search(r'(\d{4})', image_id_clean)
        if number_match:
            file_number = number_match.group(1)
            possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/pityriasis-alba-images/pityriasis-alba-{file_number}.jpg")
            for suffix in range(1, 11):
                possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/pityriasis-alba-images/pityriasis-alba-{file_number}-{suffix}.jpg")
    # 匹配 psoriasis
    elif 'psoriasis' in image_id_clean.lower():
        number_match = re.search(r'(\d{4})', image_id_clean)
        if number_match:
            file_number = number_match.group(1)
            possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/PSORIASIS/psoriasis-{file_number}.jpg")
            for suffix in range(1, 11):
                possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/PSORIASIS/psoriasis-{file_number}-{suffix}.jpg")
    # 匹配 vitiligo
    elif 'vitiligo' in image_id_clean.lower():
        number_match = re.search(r'(\d{4})', image_id_clean)
        if number_match:
            file_number = number_match.group(1)
            possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/vitiligo/vitiligo-{file_number}.jpg")
            for suffix in range(1, 11):
                possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/vitiligo/vitiligo-{file_number}-{suffix}.jpg")
    # 匹配通用皮肤图片
    elif 'skin-image' in image_id_clean.lower():
        number_match = re.search(r'(\d{4})', image_id_clean)
        if number_match:
            file_number = number_match.group(1)
            possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/skin-image-{file_number}.jpg")
            for suffix in range(1, 11):
                possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/skin-image-{file_number}-{suffix}.jpg")
    # ISIC原始文件
    elif image_id_clean.startswith('ISIC_'):
        possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/{image_id_clean}.jpg")
        possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/{image_id_clean}.png")
    
    # 兜底：直接尝试原文件名
    possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/{image_id}.jpg")
    possible_paths.append(f"{GITHUB_IMAGE_FOLDER}/{image_id}.png")

    # 尝试加载
    for path in possible_paths:
        raw_url = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{path}"
        try:
            response = requests.head(raw_url, timeout=3)
            if response.status_code == 200:
                return raw_url
        except:
            continue

    # 调用备用图片
    return get_fallback_image_url()

# === 医生信息采集 ===
def profile_step():
    st.title("🩺 皮肤病AI辅助诊断研究")
    st.subheader("第一步：医生信息采集（匿名）")
    
    # 提前初始化Google Sheets连接（确保后续保存正常）
    if st.session_state.gs_sheet is None:
        st.session_state.gs_sheet = init_google_sheets()
    
    with st.form("profile_form", clear_on_submit=True):
        hospital_level = st.selectbox(
            "1. 医院等级（注：实习生/规培生属于社区医院）", 
            ["三甲医院专科医生", "二级医院专科医生", "社区医院医生（含实习生/规培生）"]
        )
        work_years = st.selectbox(
            "2. 工作年限", 
            ["≤5年（低年限）", "5-15年", ">15年（高年限）", "无工作经验（实习生）"]
        )
        daily_patients = st.selectbox(
            "3. 日均接诊量", 
            ["≤30例", "30-50例", ">50例", "无接诊经验（实习生）"]
        )
        prior_ai_trust = st.slider(
            "4. 实验前对AI辅助诊断的信任度", 
            1, 5, 3,
            help="1=极不信任，3=中立，5=极度信任"
        )
        st.caption("💡 提示：请滑动滑块选择信任度（1-5分）")
        
        if st.form_submit_button("✅ 提交信息并开始测试"):
            level_prefix = {
                "三甲医院专科医生": "A",
                "二级医院专科医生": "B",
                "社区医院医生（含实习生/规培生）": "C"
            }[hospital_level]
            st.session_state.doctor_id = f"{level_prefix}_DR_{uuid.uuid4().hex[:6].upper()}"
            
            st.session_state.doctor_info = {
                "doctor_id": st.session_state.doctor_id,
                "hospital_level": hospital_level,
                "work_years": work_years,
                "daily_patients": daily_patients,
                "prior_ai_trust": prior_ai_trust,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            try:
                gold_df = load_gold_data()
                if ">15年" in work_years:
                    more_trap = gold_df[~gold_df["ai_correct"]].sample(min(2, len(gold_df[~gold_df["ai_correct"]])))
                    gold_df = pd.concat([gold_df, more_trap]).drop_duplicates()
                st.session_state.test_set = load_balanced_test_set(gold_df)
                st.session_state.step = "test"
                st.rerun()
            except Exception as e:
                st.error(f"测试数据加载失败：{str(e)}")

# === 答题流程 ===
def test_step():
    if st.session_state.test_set is None:
        st.error("⚠️ 测试数据未加载，请返回重新开始")
        if st.button("🔄 返回重新开始"):
            init_session_state()
            st.session_state.step = "profile"
            st.rerun()
        return
    
    # 确保Google Sheets连接有效
    if st.session_state.gs_sheet is None:
        st.session_state.gs_sheet = init_google_sheets()
    
    idx = st.session_state.current_idx
    test_set = st.session_state.test_set
    
    if idx >= len(test_set):
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
    
    image_url = get_github_image_url(image_id)
    try:
        st.image(image_url, use_container_width=True, caption=f"当前图片：{image_url.split('/')[-1]}")
    except Exception as e:
        st.image("https://via.placeholder.com/600x400?text=图片加载异常", use_container_width=True)
        st.toast(f"⚠️ 图片加载异常：{str(e)}", icon="⚠️")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 第一阶段：独立诊断")
        st.caption("💡 提示：至少选择Top1，Top2/3可选“无”")
        top1 = st.selectbox("首选 (Top-1) [必填]", ["请选择"] + ALL_CLASSES, key=f"t1_{idx}")
        top2_options = ["无"] + [c for c in ALL_CLASSES if c != top1]
        top2 = st.selectbox("次选 (Top-2) [可选]", top2_options, key=f"t2_{idx}", index=0)
        top3_options = ["无"] + [c for c in ALL_CLASSES if c not in [top1, top2]]
        top3 = st.selectbox("备选 (Top-3) [可选]", top3_options, key=f"t3_{idx}", index=0)
        conf_init = st.slider("初始信心自评（1-10分）", 1, 10, 5, key=f"c1_{idx}")
        
        is_valid = top1 != "请选择"
        if not st.session_state.show_ai:
            if st.button("🔍 获取AI辅助建议", disabled=not is_valid):
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
    
    with col2:
        if st.session_state.show_ai:
            st.markdown("### 第二阶段：AI辅助决策")
            ai_sug = st.session_state.ai_suggestion["label"]
            initial_top1 = st.session_state.initial_top[0]
            
            if st.session_state.ai_same_as_initial:
                st.success(f"✅ 您的初始诊断（{initial_top1}）与AI建议（{ai_sug}）一致！")
                
                if st.button("✅ 确认结果并进入下一题", key=f"btn_{idx}"):
                    time_post_ai = round(time.time() - st.session_state.question_start, 2)
                    confidence_gain = 0
                    is_initial_top1_correct = (initial_top1 == true_label)
                    is_initial_top3_correct = (true_label in [t for t in st.session_state.initial_top if t != "无"])
                    
                    final_top1 = initial_top1
                    final_top2 = st.session_state.initial_top[1]
                    final_top3 = st.session_state.initial_top[2]
                    final_top4 = "无"
                    is_final_top1_correct = is_initial_top1_correct
                    is_final_top3_correct = is_initial_top3_correct
                    is_final_top4_correct = (true_label in [final_top1, final_top2, final_top3])
                    use_ai = 0
                    
                    initial_correct = is_initial_top1_correct
                    final_correct = is_final_top1_correct
                    decision_path = "一致"
                    is_misled = False
                    is_rescued = False
                    
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
                        "is_initial_top3_correct": is_initial_top3_correct,
                        "interaction_type": "一致",
                        "action_taken": "无需选择（AI与初始一致）",
                        "use_ai": use_ai,
                        "final_top1": final_top1,
                        "final_top2": final_top2,
                        "final_top3": final_top3,
                        "final_top4": final_top4,
                        "is_final_top1_correct": is_final_top1_correct,
                        "is_final_top3_correct": is_final_top3_correct,
                        "is_final_top4_correct": is_final_top4_correct,
                        "final_confidence": st.session_state.initial_conf,
                        "confidence_gain": confidence_gain,
                        "decision_path": decision_path,
                        "is_misled": is_misled,
                        "is_rescued": is_rescued,
                        "time_baseline": st.session_state.time_baseline,
                        "time_post_ai": time_post_ai,
                        "submit_time": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # 保存数据到Google Sheets
                    save_result_to_backend(result)
                    st.session_state.user_results.append(result)
                    
                    reset_test_state()
                    st.session_state.current_idx = idx + 1
                    st.rerun()
            
            else:
                st.warning(f"⚠️ 您的初始诊断（{initial_top1}）与AI建议（{ai_sug}）不一致！")
                interaction_type = "冲突"
                
                st.markdown("#### 交互动作选择")
                action = st.radio(
                    "您希望如何处理AI建议？",
                    ["坚持原诊断", "替换为AI建议"],
                    key=f"act_{idx}"
                )
                
                final_top1 = initial_top1 if action == "坚持原诊断" else ai_sug
                final_top2 = st.session_state.initial_top[1]
                final_top3 = st.session_state.initial_top[2]
                final_top4 = "无"
                conf_final = st.slider("最终信心自评（1-10分）", 1, 10, st.session_state.initial_conf, key=f"c2_{idx}")
                
                if st.button("✅ 确认结果并进入下一题", key=f"btn_{idx}"):
                    time_post_ai = round(time.time() - st.session_state.question_start, 2)
                    confidence_gain = conf_final - st.session_state.initial_conf
                    is_initial_top1_correct = (initial_top1 == true_label)
                    is_initial_top3_correct = (true_label in [t for t in st.session_state.initial_top if t != "无"])
                    
                    is_final_top1_correct = (final_top1 == true_label)
                    final_options = [t for t in [final_top1, final_top2, final_top3] if t != "无"]
                    is_final_top3_correct = (true_label in final_options[:3])
                    is_final_top4_correct = (true_label in final_options)
                    use_ai = 1 if action == "替换为AI建议" else 0
                    
                    initial_correct = is_initial_top1_correct
                    final_correct = is_final_top1_correct
                    decision_path = ""
                    is_misled = False
                    is_rescued = False
                    if initial_correct and not final_correct:
                        decision_path = "误导"
                        is_misled = True
                    elif not initial_correct and final_correct:
                        decision_path = "纠正"
                        is_rescued = True
                    elif initial_correct and final_correct:
                        decision_path = "固执"
                    else:
                        decision_path = "盲从"
                    
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
                        "is_initial_top3_correct": is_initial_top3_correct,
                        "interaction_type": interaction_type,
                        "action_taken": action,
                        "use_ai": use_ai,
                        "final_top1": final_top1,
                        "final_top2": final_top2,
                        "final_top3": final_top3,
                        "final_top4": final_top4,
                        "is_final_top1_correct": is_final_top1_correct,
                        "is_final_top3_correct": is_final_top3_correct,
                        "is_final_top4_correct": is_final_top4_correct,
                        "final_confidence": conf_final,
                        "confidence_gain": confidence_gain,
                        "decision_path": decision_path,
                        "is_misled": is_misled,
                        "is_rescued": is_rescued,
                        "time_baseline": st.session_state.time_baseline,
                        "time_post_ai": time_post_ai,
                        "submit_time": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # 保存数据到Google Sheets
                    save_result_to_backend(result)
                    st.session_state.user_results.append(result)
                    
                    reset_test_state()
                    st.session_state.current_idx = idx + 1
                    st.rerun()

# === 结果展示（从Google Sheets读取数据） ===
def result_step():
    st.title("🏁 测试完成！研究数据可视化报告")
    st.success(f"✅ 您的测试已完成！唯一标识ID：{st.session_state.doctor_id}")
    st.info("📌 所有数据已唯一存储到Google Sheets")
    
    # 从Google Sheets读取当前用户数据
    try:
        if st.session_state.gs_sheet is None:
            st.session_state.gs_sheet = init_google_sheets()
        
        # 读取所有数据并筛选当前用户
        all_data = st.session_state.gs_sheet.get_all_records()
        df = pd.DataFrame(all_data)
        user_df = df[df["doctor_id"] == st.session_state.doctor_id]
        
        if len(user_df) == 0:
            st.warning("⚠️ 未查询到您的答题数据")
            st.warning("可能是数据存储延迟，请稍后刷新或重新测试")
            if st.button("🔄 重新开始测试"):
                init_session_state()
                st.rerun()
            return
        
        # 1. 核心诊断指标
        st.subheader("📊 你的诊断表现")
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_acc = user_df["is_initial_top1_correct"].mean() * 100
            st.metric("初始诊断准确率", f"{initial_acc:.1f}%")
        with col2:
            final_acc = user_df["is_final_top1_correct"].mean() * 100
            st.metric("最终诊断准确率", f"{final_acc:.1f}%", delta=f"{final_acc - initial_acc:.1f}%")
        with col3:
            ai_usage = user_df["use_ai"].sum()
            st.metric("采纳AI建议次数", ai_usage)
        
        # 2. 信心变化趋势
        st.subheader("📈 诊断信心变化")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(user_df.index + 1, user_df["initial_confidence"], marker='o', label='初始信心', color='#4285F4')
        ax.plot(user_df.index + 1, user_df["final_confidence"], marker='s', label='最终信心', color='#34A853')
        ax.set_xlabel("题目序号")
        ax.set_ylabel("信心评分（1-10）")
        ax.set_title("每道题的诊断信心变化")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # 3. AI交互分析
        st.subheader("🤖 AI交互分析")
        conflict_df = user_df[user_df["interaction_type"] == "冲突"]
        if len(conflict_df) > 0:
            misled_count = conflict_df["is_misled"].sum()
            rescued_count = conflict_df["is_rescued"].sum()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("被AI误导次数", misled_count)
            with col2:
                st.metric("被AI纠正次数", rescued_count)
        
        # 4. 详细答题数据
        st.subheader("📋 详细答题记录")
        display_df = user_df[["image_id", "true_label", "initial_top1", "final_top1", "ai_label", "action_taken"]]
        display_df.columns = ["图片ID", "真实诊断", "你的初始诊断", "你的最终诊断", "AI建议", "处理方式"]
        st.dataframe(display_df, use_container_width=True)
        
        # 5. 数据下载（从Google Sheets导出）
        st.subheader("📥 数据导出")
        user_csv = user_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="下载你的答题数据（CSV）",
            data=user_csv,
            file_name=f"skin_diagnosis_{st.session_state.doctor_id}_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    except Exception as e:
        st.error(f"⚠️ 数据展示失败：{str(e)}")
        st.error("请检查网络连接和Google Sheets权限")
    
    if st.button("🔄 重新开始测试"):
        init_session_state()
        st.rerun()

# === 主函数 ===
def main():
    try:
        import gspread
        import oauth2client
    except ImportError:
        st.error("⚠️ 缺少依赖库，请运行：pip install gspread oauth2client")
        st.stop()
    
    init_session_state()
    if st.session_state.step == "profile":
        profile_step()
    elif st.session_state.step == "test":
        test_step()
    elif st.session_state.step == "result":
        result_step()

if __name__ == "__main__":
    main()
