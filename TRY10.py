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
import re
import random
from io import BytesIO

# === 核心配置 ===
st.set_option('client.showErrorDetails', True)
st.set_page_config(page_title="皮肤病AI辅助诊断研究", page_icon="🩺", layout="centered")

# 性能优化配置
REQUEST_TIMEOUT = 1
CACHE_TTL = 3600
IMAGE_COMPRESS_WIDTH = 600
IMAGE_QUALITY = 85

# GitHub 配置
GITHUB_USERNAME = "Grass1121"
GITHUB_REPO = "skin-question"
GOLD_TXT = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/boosted_final_detail4.UTF-8.txt"

# ========== Google Sheets 强制开启配置 ==========
GOOGLE_SHEET_NAME = "皮肤诊断数据"
LOCAL_GOOGLE_CREDENTIALS_FILE = "google_credentials.json"

# GitHub 图片路径
GITHUB_IMAGE_FOLDER = "experiment_pool"
GITHUB_BRANCH = "main"

# 疾病标签
DISEASE_LABELS = {
    "MEL": "黑色素瘤", "NV": "痣（色素痣）", "BCC": "基底细胞癌", "AK": "光化性角化病",
    "BKL": "良性角化病（脂溢性角化等）", "DF": "皮肤纤维瘤", "VASC": "血管病变", "SCC": "鳞状细胞癌",
    "Vitiligo": "白癜风", "Pityrasis-Alba": "白色糠疹", "Psoriasis": "银屑病", "UNK": "未知类别"
}
ALL_CLASSES = list(DISEASE_LABELS.values())
TEST_COUNT = 10

# === Google Sheets 初始化 ===
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
                return None, "❌ 未找到凭证文件 google_credentials.json"
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
        return None, f"❌ 未找到表格：{GOOGLE_SHEET_NAME}"
    except Exception as e:
        return None, f"❌ Google Sheets 初始化失败：{str(e)}"

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
        "final_top1": "", "final_top2": "", "final_top3": "", "final_top4": "",
        "final_conf": 5,
        "question_start": 0,
        "time_baseline": 0,
        "doctor_id": "",
        "ai_same_as_initial": False,
    }
    for k, v in default_states.items():
        if k not in st.session_state:
            st.session_state[k] = v

# === 测试集加载 ===
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_gold_data_cached():
    try:
        resp = requests.get(GOLD_TXT, timeout=8)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), encoding="utf-8")
        req_cols = ["image_id", "Top1_预测", "真实病名"]
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            return None, f"缺失字段：{missing}"

        df["true_cn"] = df["真实病名"].map(DISEASE_LABELS).fillna("未知")
        df["ai_cn"] = df["Top1_预测"].map(DISEASE_LABELS).fillna("未知")
        df["ai_correct"] = df["true_cn"] == df["ai_cn"]
        df = df[(df["true_cn"] != "未知") & (df["ai_cn"] != "未知")]
        return df, None
    except Exception as e:
        return None, f"加载失败：{str(e)}"

# === 均衡采样 ===
def load_balanced_test_set(df):
    correct_sample = pd.DataFrame()
    incorrect_sample = pd.DataFrame()
    ai_correct = df[df["ai_correct"]]
    ai_incorrect = df[~df["ai_correct"]]

    if len(ai_correct) > 0:
        correct_sample = ai_correct.sample(min(6, len(ai_correct)), replace=False)
        need = max(0, 6 - len(correct_sample))
        if need > 0 and len(ai_correct) >= need:
            correct_sample = pd.concat([correct_sample, ai_correct.sample(need, replace=False)])

    if len(ai_incorrect) > 0:
        incorrect_sample = ai_incorrect.sample(min(4, len(ai_incorrect)), replace=False)
        need = max(0, 4 - len(incorrect_sample))
        if need > 0 and len(ai_incorrect) >= need:
            incorrect_sample = pd.concat([incorrect_sample, ai_incorrect.sample(need, replace=False)])

    if correct_sample.empty and incorrect_sample.empty:
        return df.head(TEST_COUNT)

    test_set = pd.concat([correct_sample, incorrect_sample]).sample(frac=1).reset_index(drop=True)
    return test_set.head(TEST_COUNT)

# === 强制保存到 Sheets ===
def save_results_to_gs():
    with st.spinner("正在保存数据到 Google Sheets..."):
        sheet, err = init_google_sheets_once()
    if err:
        st.error(err)
        return False

    if not st.session_state.user_results:
        st.warning("无结果可保存")
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
        st.success(f"✅ 已保存 {len(rows)} 条记录")
        return True
    except Exception as e:
        st.error(f"❌ 写入失败：{str(e)}")
        return False

# === 单题状态重置 ===
def reset_test_state():
    st.session_state.show_ai = False
    st.session_state.initial_top = ["请选择", "无", "无"]
    st.session_state.initial_conf = 5
    st.session_state.final_top1 = ""
    st.session_state.final_top2 = ""
    st.session_state.final_top3 = ""
    st.session_state.final_top4 = "无"
    st.session_state.final_conf = 5
    st.session_state.time_baseline = 0
    st.session_state.ai_same_as_initial = False

# === 图片压缩（修复PIL识别错误）===
def compress_image(image_url):
    try:
        r = requests.get(image_url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content))
        # 转换为RGB模式以避免格式问题
        if img.mode in ("RGBA", "P"):
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
        # 图片无法识别时返回一张默认占位图
        buf = BytesIO()
        placeholder = Image.new("RGB", (IMAGE_COMPRESS_WIDTH, int(IMAGE_COMPRESS_WIDTH*0.75)), color="#DDDDDD")
        placeholder.save(buf, format="JPEG")
        buf.seek(0)
        return buf
    except Exception:
        # 其他加载失败情况返回占位图
        buf = BytesIO()
        placeholder = Image.new("RGB", (IMAGE_COMPRESS_WIDTH, int(IMAGE_COMPRESS_WIDTH*0.75)), color="#DDDDDD")
        placeholder.save(buf, format="JPEG")
        buf.seek(0)
        return buf

# === 图片URL获取 ===
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

# === 医生信息页 ===
def profile_step():
    st.title("🩺 皮肤病AI辅助诊断研究问卷")
    st.subheader("第一步：医生基本信息采集（匿名）")
    with st.form("profile_form"):
        # 医院等级
        hospital_level = st.selectbox(
            "所在医院等级",
            ["三甲医院专科医生", "二级医院专科医生", "社区医院/实习生"],
            help="请选择你的执业医院等级"
        )
        # 工作年限
        work_years = st.selectbox(
            "从事皮肤科工作年限",
            ["≤5年", "5-10年", "10-15年", ">15年", "无临床经验（实习生）"],
            help="请选择你的皮肤科工作年限"
        )
        # 日均接诊量（调整为15、30为界）
        daily_patients = st.selectbox(
            "日均接诊皮肤病患者数量",
            ["≤15例", "15-30例", ">30例", "无接诊经验"],
            help="请选择你的日均接诊量范围"
        )
        # AI信任度
        prior_ai_trust = st.slider(
            "对AI辅助诊断的初始信任度（1-5分）",
            1, 5, 3,
            help="1分：完全不信任，5分：完全信任"
        )
        
        submit_btn = st.form_submit_button("✅ 提交信息，开始测试")
        
        if submit_btn:
            # 生成标准化ID（三甲A/二级B/社区C开头）
            prefix = "A" if "三甲" in hospital_level else "B" if "二级" in hospital_level else "C"
            doctor_id = f"{prefix}_DR_{uuid.uuid4().hex[:6].upper()}"
            
            st.session_state.doctor_info = {
                "doctor_id": doctor_id,
                "hospital_level": hospital_level,
                "work_years": work_years,
                "daily_patients": daily_patients,
                "prior_ai_trust": prior_ai_trust
            }
            
            # 加载测试集
            with st.spinner("正在加载测试病例..."):
                df, err = load_gold_data_cached()
                if err:
                    st.error(err)
                    return
                # 针对>15年高年资医生补充AI错误病例
                if work_years == ">15年" and len(df[~df["ai_correct"]]) >= 2:
                    add_samples = df[~df["ai_correct"]].sample(2)
                    df = pd.concat([df, add_samples]).drop_duplicates()
                st.session_state.test_set = load_balanced_test_set(df)
            
            st.session_state.step = "test"
            st.rerun()

# === 测试答题页（只选一个就能提交）===
def test_step():
    ts = st.session_state.test_set
    if ts is None or ts.empty:
        st.error("测试集加载失败，请刷新页面重试")
        return
    
    idx = st.session_state.current_idx
    if idx >= TEST_COUNT:
        save_results_to_gs()
        st.session_state.step = "result"
        st.rerun()
    
    cur = ts.iloc[idx]
    img_id = cur["image_id"]
    truth = cur["true_cn"]
    ai_lbl = cur["ai_cn"]
    ai_ok = ai_lbl == truth

    # 页面标题
    st.title(f"📝 病例诊断 - 第 {idx+1}/{TEST_COUNT} 题")
    st.progress((idx+1)/TEST_COUNT)

    # 显示压缩后的图片（隐藏ID + 错误处理）
    st.subheader("皮损图像")
    img_url = get_image_url_cached(img_id)
    compressed_img = compress_image(img_url)
    st.image(compressed_img, use_container_width=True)

    # 初始诊断区域
    st.markdown("### 一、独立诊断")
    with st.form(f"initial_diagnosis_form_{idx}"):
        # 首选诊断（必选）
        t1 = st.selectbox(
            "首选诊断结果",
            ["请选择"] + ALL_CLASSES,
            key=f"t1_{idx}",
            help="请选择你认为最可能的诊断结果（必填）"
        )
        # 二选诊断（可选）
        t2_opt = ["无"] + [x for x in ALL_CLASSES if x != t1]
        t2 = st.selectbox("第二诊断结果（可选）", t2_opt, key=f"t2_{idx}")
        # 三选诊断（可选）
        t3_opt = ["无"] + [x for x in ALL_CLASSES if x not in [t1, t2]]
        t3 = st.selectbox("第三诊断结果（可选）", t3_opt, key=f"t3_{idx}")
        # 初始信心（默认5）
        conf_i = st.slider(
            "对本次诊断的信心值（1-10分）",
            1, 10, 5,
            key=f"ci_{idx}",
            help="1分：完全不确定，10分：完全确定"
        )
        
        # 提交按钮（仅校验首选诊断）
        submit_initial = st.form_submit_button("🔍 提交诊断，查看AI建议")
        if submit_initial:
            if t1 == "请选择":
                st.error("请至少选择首选诊断结果后提交")
            else:
                # 记录初始诊断信息
                st.session_state.initial_top = [t1, t2, t3]
                st.session_state.initial_conf = conf_i
                st.session_state.ai_suggestion = {"label": ai_lbl}
                st.session_state.ai_same_as_initial = (t1 == ai_lbl)
                st.session_state.question_start = time.time()
                st.session_state.time_baseline = round(time.time() - st.session_state.question_start, 2)
                st.session_state.show_ai = True
                st.rerun()

    # AI建议展示及最终决策
    if st.session_state.show_ai:
        st.markdown("### 二、AI辅助决策")
        st.info(f"📌 AI辅助诊断建议：**{ai_lbl}**")
        
        # 判断初始诊断与AI是否一致
        init1 = st.session_state.initial_top[0]
        same_with_ai = init1 == ai_lbl
        
        if same_with_ai:
            st.success(f"✅ 你的初始诊断与AI建议一致：{init1}")
        else:
            st.warning(f"⚠️ 你的初始诊断（{init1}）与AI建议（{ai_lbl}）不一致")
        
        # 最终决策表单
        with st.form(f"final_decision_form_{idx}"):
            # 决策选择
            act = st.radio(
                "最终决策选择",
                ["坚持原诊断", "采纳AI建议"],
                key=f"act_{idx}"
            )
            
            # 最终诊断结果
            final1 = init1 if act == "坚持原诊断" else ai_lbl
            st.session_state.final_top1 = final1
            
            # 最终信心（默认与初始一致）
            final_conf = st.slider(
                "最终诊断信心值（1-10分）",
                1, 10, st.session_state.initial_conf,
                key=f"cf_{idx}",
                help="1分：完全不确定，10分：完全确定"
            )
            
            # 提交按钮
            submit_final = st.form_submit_button("✅ 确认最终诊断，进入下一题")
            if submit_final:
                # 计算耗时
                t_post = round(time.time() - st.session_state.question_start, 2)
                gain = final_conf - st.session_state.initial_conf
                
                # 判断诊断正确性
                ini_ok = (init1 == truth)
                fin_ok = (final1 == truth)
                use_ai = 1 if act == "采纳AI建议" else 0
                
                # 决策路径分类
                if ini_ok and not fin_ok:
                    path, misled, rescued = "误导", True, False
                elif not ini_ok and fin_ok:
                    path, misled, rescued = "纠正", False, True
                elif ini_ok and fin_ok:
                    path, misled, rescued = "同对坚持", False, False
                else:
                    path, misled, rescued = "错上改错", False, False
                
                # 记录结果
                result = {
                    **st.session_state.doctor_info,
                    "image_id": img_id,
                    "true_label": truth,
                    "ai_label": ai_lbl,
                    "ai_is_correct": ai_ok,
                    "initial_top1": init1,
                    "initial_top2": st.session_state.initial_top[1],
                    "initial_top3": st.session_state.initial_top[2],
                    "initial_confidence": st.session_state.initial_conf,
                    "is_initial_top1_correct": ini_ok,
                    "is_initial_top3_correct": truth in st.session_state.initial_top,
                    "interaction_type": "一致" if same_with_ai else "冲突",
                    "action_taken": act,
                    "use_ai": use_ai,
                    "final_top1": final1,
                    "final_top2": st.session_state.initial_top[1],
                    "final_top3": st.session_state.initial_top[2],
                    "final_top4": "无",
                    "is_final_top1_correct": fin_ok,
                    "is_final_top3_correct": truth in [final1, st.session_state.initial_top[1], st.session_state.initial_top[2]],
                    "is_final_top4_correct": False,
                    "final_confidence": final_conf,
                    "confidence_gain": gain,
                    "decision_path": path,
                    "is_misled": misled,
                    "is_rescued": rescued,
                    "time_baseline": st.session_state.time_baseline,
                    "time_post_ai": t_post,
                    "submit_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.session_state.user_results.append(result)
                
                # 重置当前题状态
                reset_test_state()
                
                # 进入下一题
                st.session_state.current_idx += 1
                st.rerun()

# === 结果页（修复width参数为stretch）===
def result_step():
    st.title("🏁 测试完成")
    st.success(f"你的测试ID：{st.session_state.doctor_id}")
    st.info("所有数据已成功写入 Google Sheets，可前往表格查看完整记录")

    # 数据预处理
    if len(st.session_state.user_results) > 0:
        df = pd.DataFrame(st.session_state.user_results)
        
        # 1. 诊断准确率对比
        st.subheader("📊 诊断准确率对比")
        initial_acc = df["is_initial_top1_correct"].mean() * 100
        final_acc = df["is_final_top1_correct"].mean() * 100
        
        acc_data = pd.DataFrame({
            "准确率（%）": [initial_acc, final_acc]
        }, index=["初始诊断（无AI）", "最终诊断（AI辅助）"])
        
        # 修复width参数为stretch，兼容所有Streamlit版本
        st.bar_chart(acc_data, color="#3498db", width="stretch")

        # 2. AI采纳效果分析
        st.subheader("📊 AI采纳效果分析")
        # 筛选采纳/未采纳AI的记录
        ai_used = df[df["use_ai"] == 1]
        ai_not_used = df[df["use_ai"] == 0]
        
        # 计算准确率
        ai_used_acc = ai_used["is_final_top1_correct"].mean() * 100 if len(ai_used) > 0 else 0
        ai_not_used_acc = ai_not_used["is_final_top1_correct"].mean() * 100 if len(ai_not_used) > 0 else 0
        
        ai_data = pd.DataFrame({
            "准确率（%）": [ai_used_acc, ai_not_used_acc]
        }, index=["采纳AI建议", "未采纳AI建议"])
        
        st.bar_chart(ai_data, color="#e74c3c", width="stretch")
        # 显示样本数
        st.caption(f"采纳AI建议：{len(ai_used)}题 | 未采纳AI建议：{len(ai_not_used)}题")

        # 关键指标汇总
        st.subheader("📈 核心指标汇总")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("初始准确率", f"{initial_acc:.1f}%")
        with col2:
            st.metric("最终准确率", f"{final_acc:.1f}%", delta=f"{final_acc-initial_acc:.1f}%")
        with col3:
            st.metric("采纳AI次数", len(ai_used))

    # 重新测试按钮（修复状态重置逻辑）
    if st.button("🔄 重新开始测试", type="primary"):
        # 强制重置所有状态并跳转
        init_session_state()
        st.session_state.step = "profile"
        st.rerun()

# === 主函数 ===
def main():
    init_session_state()
    
    step = st.session_state.step
    if step == "profile":
        profile_step()
    elif step == "test":
        test_step()
    elif step == "result":
        result_step()

if __name__ == "__main__":
    main()
