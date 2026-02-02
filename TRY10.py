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
import json  # 新增：用于解析Secrets中的JSON字符串
# 新增：Google Sheets相关导入
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# === 核心配置 ===
st.set_option('client.showErrorDetails', True)  # 修改：开启错误详情，方便调试
st.set_page_config(page_title="皮肤病AI辅助诊断研究", page_icon="🩺", layout="wide")

# 你的GitHub信息
GITHUB_USERNAME = "Grass134"
GITHUB_REPO = "skin-question"
GOLD_TXT = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/boosted_final_detail4.UTF-8.txt"

# ========== 本地CSV配置 ==========
BACKEND_CSV_PATH = "skin_diagnosis_backend_data.csv"

# ========== Google Sheets配置（关键修改：移除本地密钥文件配置） ==========
GOOGLE_SHEET_NAME = "皮肤诊断数据"  # 确认你的Google表格名称完全一致！
# 本地运行时的备用密钥文件（线上部署时不会用到）
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

# === 初始化Google Sheets连接（核心修改：修复Secrets读取逻辑） ===
def init_google_sheets():
    """初始化Google Sheets连接，返回表格对象
    优先从Streamlit Secrets读取密钥，本地运行时fallback到本地文件
    """
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # ========== 关键修改1：增加详细调试信息 ==========
        st.write("📝 调试信息 - Secrets中的所有键：", list(st.secrets.keys()))
        if "GOOGLE_CREDENTIALS" in st.secrets:
            st.write("✅ 检测到GOOGLE_CREDENTIALS键")
            st.write("🔍 密钥类型：", type(st.secrets["GOOGLE_CREDENTIALS"]))
            # 显示前100个字符（避免泄露完整密钥）
            st.write("🔍 密钥内容片段：", str(st.secrets["GOOGLE_CREDENTIALS"])[:100])
        
        # ========== 关键修改2：简化并修复Secrets读取逻辑 ==========
        # 第一步：尝试从Streamlit Secrets读取（线上部署）
        try:
            # 从Secrets读取内容
            creds_content = st.secrets["GOOGLE_CREDENTIALS"]
            
            # 处理不同格式：如果是字符串则解析为JSON，否则直接使用字典
            if isinstance(creds_content, str):
                try:
                    creds_dict = json.loads(creds_content)
                    st.success("✅ JSON字符串解析成功")
                except json.JSONDecodeError as e:
                    st.error(f"❌ JSON解析失败：{str(e)}")
                    st.error("🔍 请检查Secrets中的JSON格式是否正确（是否有多余/缺失的逗号、引号）")
                    raise
            else:
                creds_dict = creds_content
            
            # 验证必要字段
            required_fields = ["type", "project_id", "private_key", "client_email"]
            missing_fields = [f for f in required_fields if f not in creds_dict]
            if missing_fields:
                st.error(f"❌ 密钥缺少必要字段：{missing_fields}")
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # 从字典加载凭证
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            st.success("✅ 从Streamlit Secrets加载Google凭证成功")
        
        # 第二步：Secrets读取失败时，尝试本地文件（本地运行）
        except KeyError:
            st.info("ℹ️ 未检测到Streamlit Secrets中的GOOGLE_CREDENTIALS键，尝试加载本地密钥文件")
            if not os.path.exists(LOCAL_GOOGLE_CREDENTIALS_FILE):
                raise FileNotFoundError(f"本地密钥文件 {LOCAL_GOOGLE_CREDENTIALS_FILE} 不存在")
            # 从本地文件加载凭证
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                LOCAL_GOOGLE_CREDENTIALS_FILE, scope
            )
            st.success("✅ 从本地文件加载Google凭证成功")
        
        # ========== 关键修改3：增加表格打开的错误处理 ==========
        # 授权并打开表格（确认表格名称完全一致）
        client = gspread.authorize(creds)
        try:
            sheet = client.open(GOOGLE_SHEET_NAME).sheet1
            st.success(f"✅ 成功打开Google表格：{GOOGLE_SHEET_NAME}")
        except gspread.exceptions.SpreadsheetNotFound:
            st.error(f"❌ 未找到Google表格：{GOOGLE_SHEET_NAME}")
            st.error("🔍 请检查表格名称是否完全一致（包括空格、中文标点），且该服务账号有访问权限")
            raise
        
        return sheet
    
    except Exception as e:
        st.warning(f"⚠️ Google Sheets连接失败：{str(e)}")
        st.warning("将仅保存到本地CSV文件，请检查凭证配置")
        return None

# === 会话状态初始化 ===
def init_session_state():
    default_states = {
        "step": "profile",
        "current_idx": 0,
        "show_ai": False,
        "user_results": [],
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
    
    # ========== 关键修改4：延迟初始化Google Sheets，确保Secrets已加载 ==========
    # 不在初始化时立即连接，而是在首次保存数据时初始化
    # 避免页面加载时过早尝试读取Secrets

# === 数据加载（稳定版本） ===
@st.cache_data(ttl=300)
def load_gold_data():
    try:
        response = requests.get(GOLD_TXT, timeout=15)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), encoding="utf-8")
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ 数据加载失败：{str(e)}")
        st.error("请检查GitHub链接是否正确，或稍后重试")
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
        st.error(f"⚠️ 有效数据不足{TEST_COUNT}条")
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

# === 数据保存（本地CSV + Google Sheets同步） ===
def save_result_to_backend(result):
    """保存数据到本地CSV，并同步到Google Sheets"""
    # ========== 关键修改5：在保存数据时初始化Google Sheets ==========
    # 首次保存时初始化Google Sheets连接
    if st.session_state.gs_sheet is None:
        st.session_state.gs_sheet = init_google_sheets()
    
    # 1. 保存到本地CSV
    try:
        pd.DataFrame([result]).to_csv(
            BACKEND_CSV_PATH,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8-sig"
        )
        st.success("✅ 数据已保存到本地CSV")
    except Exception as e:
        st.warning(f"本地CSV保存失败：{str(e)}")
    
    # 2. 同步到Google Sheets
    if st.session_state.gs_sheet is not None:
        try:
            # 将字典转为列表（按表头顺序）
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
            # 追加到Google Sheets
            st.session_state.gs_sheet.append_row(row_data)
            st.success("✅ 数据已同步到Google Sheets")
        except Exception as e:
            st.warning(f"Google Sheets同步失败：{str(e)}")

# === 重置答题状态（不重置test_set） ===
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
    st.session_state.user_results = []

# === 图片加载（核心修改：适配P2/P3等新图片名） ===
def get_github_image_url(image_id):
    """
    适配修改后的图片名（P2/P3等），优先匹配：
    1. vitiligo文件夹下的P2/P3等图片
    2. pityrasis-alba-images文件夹下的P2/P3等图片
    3. PSORIASIS文件夹下的图片
    4. 根文件夹下的图片
    """
    # 核心修改：先尝试将原始image_id映射为P2/P3（如果需要固定映射）
    # 如果你的image_id本身已经是P2/P3，可注释掉下面的映射逻辑
    image_mapping = {
        # 示例：原始image_id -> 新图片名（根据你的实际映射关系修改）
        "vitiligo_original_001": "P2",
        "pityrasis_alba_original_001": "P3",
        "vitiligo_original_002": "P4",
        "pityrasis_alba_original_002": "P5"
    }
    
    # 使用映射后的图片名（如果有映射），否则用原始image_id
    new_image_id = image_mapping.get(image_id, image_id)
    
    possible_paths = [
        # 优先查找vitiligo文件夹下的P2/P3等图片
        f"{GITHUB_IMAGE_FOLDER}/vitiligo/{new_image_id}.jpg",
        f"{GITHUB_IMAGE_FOLDER}/vitiligo/{new_image_id}.png",
        # 其次查找pityrasis-alba-images文件夹下的P2/P3等图片
        f"{GITHUB_IMAGE_FOLDER}/pityrasis-alba-images/{new_image_id}.jpg",
        f"{GITHUB_IMAGE_FOLDER}/pityrasis-alba-images/{new_image_id}.png",
        # 保留PSORIASIS文件夹
        f"{GITHUB_IMAGE_FOLDER}/PSORIASIS/{new_image_id}.jpg",
        f"{GITHUB_IMAGE_FOLDER}/PSORIASIS/{new_image_id}.png",
        # 根文件夹兜底
        f"{GITHUB_IMAGE_FOLDER}/{new_image_id}.jpg",
        f"{GITHUB_IMAGE_FOLDER}/{new_image_id}.png"
    ]
    
    for path in possible_paths:
        raw_url = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{path}"
        try:
            response = requests.head(raw_url, timeout=3)
            if response.status_code == 200:
                return raw_url
        except:
            continue
    
    # 调试：显示尝试过的图片路径
    st.warning(f"⚠️ 图片加载失败 - 尝试过的路径：{possible_paths}")
    return "https://via.placeholder.com/600x400?text=图片未找到"

# === 医生信息采集 ===
def profile_step():
    st.title("🩺 皮肤病AI辅助诊断研究")
    st.subheader("第一步：医生信息采集（匿名）")
    
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
            help="请滑动滑块选择信任度：1=极不信任，3=中立，5=极度信任"
        )
        st.caption("💡 提示：请滑动上方滑块选择您对AI辅助诊断的初始信任度（1-5分）")
        
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
        st.image(image_url, use_container_width=True, caption=f"图片ID：{image_id}（实际加载：{image_url.split('/')[-1]}）")
    except:
        st.image("https://via.placeholder.com/600x400?text=图片加载失败", use_container_width=True)
        st.warning(f"⚠️ 图片ID {image_id} 加载失败，请检查GitHub路径")
    
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
                st.caption("请先选择Top1")
    
    with col2:
        if st.session_state.show_ai:
            st.markdown("### 第二阶段：AI辅助决策")
            ai_sug = st.session_state.ai_suggestion["label"]
            initial_top1 = st.session_state.initial_top[0]
            
            if st.session_state.ai_same_as_initial:
                st.success(f"✅ 您的初始诊断（{initial_top1}）与AI建议（{ai_sug}）一致！无需额外选择")
                
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
                    
                    st.session_state.user_results.append(result)
                    save_result_to_backend(result)
                    
                    reset_test_state()
                    st.session_state.current_idx = idx + 1
                    st.rerun()
            
            else:
                st.warning(f"⚠️ 您的初始诊断（{initial_top1}）与AI建议（{ai_sug}）不一致！请选择处理方式")
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
                    
                    st.session_state.user_results.append(result)
                    save_result_to_backend(result)
                    
                    reset_test_state()
                    st.session_state.current_idx = idx + 1
                    st.rerun()

# === 结果展示 + 数据下载 ===
def result_step():
    st.title("🏁 测试完成！研究数据可视化报告")
    st.success(f"✅ 您的测试已完成！您的唯一标识ID：{st.session_state.doctor_id}")
    st.info("📌 所有数据均匿名存储，已同步到Google Sheets")
    
    results = st.session_state.user_results
    if not results:
        st.warning("暂无答题结果")
        if st.button("🔄 重新开始测试"):
            init_session_state()
            st.rerun()
        return
    
    df = pd.DataFrame(results)
    
    # 1. 机构层级准确率
    st.subheader("1. 机构层级：不同医院的诊断准确率")
    hospital_group = df.groupby("hospital_level").agg(
        initial_top1=("is_initial_top1_correct", "mean"),
        final_top1=("is_final_top1_correct", "mean"),
        initial_top3=("is_initial_top3_correct", "mean"),
        final_top3=("is_final_top3_correct", "mean")
    ).reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(hospital_group["hospital_level"]))
    width = 0.35
    ax1.bar(x-width/2, hospital_group["initial_top1"], width, label="初始诊断", color="#4285F4")
    ax1.bar(x+width/2, hospital_group["final_top1"], width, label="AI辅助后", color="#34A853")
    ax1.set_title("Top-1准确率（按机构）")
    ax1.set_xlabel("机构类型")
    ax1.set_ylabel("准确率")
    ax1.set_xticks(x)
    ax1.set_xticklabels(hospital_group["hospital_level"], rotation=15)
    ax1.legend()
    ax2.bar(x-width/2, hospital_group["initial_top3"], width, label="初始诊断", color="#FBBC05")
    ax2.bar(x+width/2, hospital_group["final_top3"], width, label="AI辅助后", color="#EA4335")
    ax2.set_title("Top-3准确率（按机构）")
    ax2.set_xlabel("机构类型")
    ax2.set_ylabel("准确率")
    ax2.set_xticks(x)
    ax2.set_xticklabels(hospital_group["hospital_level"], rotation=15)
    ax2.legend()
    st.pyplot(fig)

    # 2. 经验水平表现
    st.subheader("2. 经验水平：不同年限医生的表现")
    df["year_group"] = df["work_years"].map(lambda x: "低年限(≤5年)" if "≤5年" in x else "中年限(5-15年)" if "5-15年" in x else "高年限(>15年)" if ">15年" in x else "无经验(实习生)")
    year_group = df.groupby("year_group").agg(
        initial_top1=("is_initial_top1_correct", "mean"),
        final_top1=("is_final_top1_correct", "mean"),
        use_ai=("use_ai", "mean")
    ).reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(year_group["year_group"], year_group["initial_top1"], label="初始诊断", color="#4285F4")
    ax1.bar(year_group["year_group"], year_group["final_top1"], bottom=year_group["initial_top1"], label="AI辅助后提升", color="#34A853")
    ax1.set_title("Top-1准确率（按经验）")
    ax1.set_xlabel("经验水平")
    ax1.set_ylabel("准确率")
    ax1.set_xticklabels(year_group["year_group"], rotation=15)
    ax1.legend()
    ax2.bar(year_group["year_group"], year_group["use_ai"], color="#FBBC05")
    ax2.set_title("AI使用频率（按经验）")
    ax2.set_xlabel("经验水平")
    ax2.set_ylabel("使用频率")
    ax2.set_xticklabels(year_group["year_group"], rotation=15)
    st.pyplot(fig)

    # 3. 接诊量与AI依赖度
    st.subheader("3. 接诊量：不同接诊量的AI依赖度")
    load_group = df.groupby("daily_patients")["use_ai"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(load_group["daily_patients"], load_group["use_ai"], color="#4285F4")
    ax.set_title("AI依赖度（按接诊量）")
    ax.set_xlabel("日均接诊量")
    ax.set_ylabel("AI依赖度")
    ax.set_xticklabels(load_group["daily_patients"], rotation=15)
    st.pyplot(fig)

    # 4. 初始信任度与AI采纳率
    st.subheader("4. 信任度：初始信任度与AI采纳率")
    trust_group = df.groupby("prior_ai_trust")["use_ai"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(trust_group["prior_ai_trust"], trust_group["use_ai"], marker="o", color="#34A853")
    ax.set_title("初始信任度与AI采纳率的关系")
    ax.set_xlabel("初始信任度（1-5分）")
    ax.set_ylabel("AI采纳率")
    st.pyplot(fig)

    # 新增：数据下载功能
    st.subheader("📥 数据导出")
    col1, col2 = st.columns(2)
    with col1:
        # 导出本地CSV
        if os.path.exists(BACKEND_CSV_PATH):
            with open(BACKEND_CSV_PATH, "r", encoding="utf-8-sig") as f:
                csv_data = f.read()
            st.download_button(
                label="下载本地完整数据（CSV）",
                data=csv_data,
                file_name=f"skin_diagnosis_local_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("暂无本地数据可下载")
    with col2:
        # 导出当前用户数据
        user_csv = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="下载本次答题数据（CSV）",
            data=user_csv,
            file_name=f"skin_diagnosis_user_{st.session_state.doctor_id}_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    if st.button("🔄 重新开始测试"):
        init_session_state()
        st.rerun()

# === 主函数 ===
def main():
    # 安装依赖提示（首次运行）
    try:
        import gspread
        import oauth2client
    except ImportError:
        st.error("⚠️ 缺少Google Sheets依赖库，请先运行：pip install gspread oauth2client")
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
