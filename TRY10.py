import streamlit as st
import pandas as pd
import os
import uuid
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import requests
import json  # 新增：读取JSON

# === 核心配置：隐藏错误详情 + 云端路径 ===
st.set_option('client.showErrorDetails', False)
st.set_page_config(page_title="皮肤病AI辅助诊断", page_icon="🩺", layout="wide")

# -------------------------- 替换你的GitHub信息 --------------------------
GITHUB_USERNAME = "Grass134"  # 直接填你的用户名
GITHUB_REPO = "skin-question"
# -----------------------------------------------------------------

# JSON的GitHub Raw链接（替换成JSON文件）
GOLD_JSON = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/main/boosted_final_detail4.json"
RESULT_CSV = f"/tmp/diagnosis_results_{uuid.uuid4().hex[:6]}.csv"

# 疾病标签映射
DISEASE_LABELS = {
    "MEL": "黑色素瘤", "NV": "痣", "BCC": "基底细胞癌", "AK": "光化性角化病",
    "BKL": "良性角化病", "DF": "皮肤纤维瘤", "VASC": "血管病变", "SCC": "鳞状细胞癌",
    "Vitiligo": "白癜风", "Pityrasis-Alba": "白色糠疹", "Psoriasis": "银屑病"
}
ALL_CLASSES = list(DISEASE_LABELS.values())
TEST_COUNT = 10

# === 1. 会话状态初始化 ===
def init_session_state():
    default_states = {
        "step": "profile",
        "current_idx": 0,
        "show_ai": False,
        "user_results": [],
        "test_set": None,
        "doctor_info": {},
        "ai_suggestion": {},
        "initial_top": ["请选择", "请选择", "请选择"],
        "initial_conf": 5,
        "final_top1": "",
        "final_decision": "",
        "final_conf": 5,
        "question_start": 0,
        "doctor_id": f"DR_{uuid.uuid4().hex[:6].upper()}"
    }
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

# === 2. 数据加载（用JSON，彻底解决编码问题） ===
@st.cache_data
def load_gold_data():
    try:
        # 读取JSON（JSON天然兼容所有编码）
        response = requests.get(GOLD_JSON, timeout=10)
        data = json.loads(response.text)
        df = pd.DataFrame(data)
    except Exception as e:
        st.error(f"⚠️ 读取云端JSON失败：{str(e)}")
        st.error("请检查：1.JSON文件是否上传到仓库 2.链接是否正确")
        st.stop()
    
    # 检查必要字段
    required_cols = ["image_id", "Top1_预测", "真实病名", "image_url"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"⚠️ JSON缺少字段：{', '.join(missing_cols)}")
        st.stop()
    
    # 处理标签
    df["true_cn"] = df["真实病名"].map(DISEASE_LABELS).fillna("未知")
    df["ai_cn"] = df["Top1_预测"].map(DISEASE_LABELS).fillna("未知")
    df["ai_correct"] = df["true_cn"] == df["ai_cn"]
    
    # 过滤数据
    df = df[df["true_cn"] != "未知"]
    df = df[df["ai_cn"] != "未知"]
    if len(df) < TEST_COUNT:
        st.error(f"⚠️ 有效数据不足{TEST_COUNT}条")
        st.stop()
    
    return df

@st.cache_data
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

# === 3. 辅助函数 ===
def save_result(result):
    try:
        df = pd.DataFrame([result])
        if not os.path.exists(RESULT_CSV):
            df.to_csv(RESULT_CSV, mode="w", header=True, index=False, encoding="utf-8-sig")
        else:
            df.to_csv(RESULT_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")
    except Exception as e:
        st.warning(f"结果保存失败：{str(e)}")

def reset_test_state():
    st.session_state.show_ai = False
    st.session_state.initial_top = ["请选择", "请选择", "请选择"]
    st.session_state.initial_conf = 5
    st.session_state.final_top1 = ""
    st.session_state.final_decision = ""
    st.session_state.final_conf = 5

# === 4. 医生信息采集 ===
def profile_step():
    st.title("🩺 皮肤病AI辅助诊断研究")
    st.subheader("第一步：医生信息采集（匿名）")
    with st.form("profile_form", clear_on_submit=True):
        st.info(f"📌 您的匿名编号：**{st.session_state.doctor_id}**")
        hospital_level = st.selectbox("1. 医院等级", ["三甲医院", "二级医院", "社区医院/基层"])
        work_years = st.selectbox("2. 工作年限", ["≤3年", "3-10年", ">10年"])
        monthly_cases = st.selectbox("3. 月接诊量（皮肤病）", ["≤30例", "30-100例", ">100例"])
        ai_trust = st.slider("4. 对AI辅助诊断的初始信任度（1-5分）", 1, 5, 3)
        if st.form_submit_button("✅ 提交信息并开始测试"):
            st.session_state.doctor_info = {
                "doctor_id": st.session_state.doctor_id,
                "hospital_level": hospital_level,
                "work_years": work_years,
                "monthly_cases": monthly_cases,
                "initial_ai_trust": ai_trust,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            try:
                gold_df = load_gold_data()
                st.session_state.test_set = load_balanced_test_set(gold_df)
                st.session_state.step = "test"
                st.rerun()
            except Exception as e:
                st.error(f"测试数据加载失败：{str(e)}")

# === 5. 答题流程 ===
def test_step():
    if st.session_state.test_set is None:
        st.error("⚠️ 测试数据未加载，请返回重新开始")
        if st.button("🔄 返回重新开始"):
            init_session_state()
            st.rerun()
        return
    idx = st.session_state.current_idx
    test_set = st.session_state.test_set
    if idx >= len(test_set):
        st.session_state.step = "result"
        st.rerun()
    current_data = test_set.iloc[idx]
    image_url = current_data["image_url"]
    true_label = current_data["true_cn"]
    ai_label = current_data["ai_cn"]
    
    st.title(f"📝 测试题 {idx + 1}/{TEST_COUNT}")
    st.progress((idx + 1) / TEST_COUNT, text=f"进度：{idx + 1}/{TEST_COUNT}")
    st.subheader("皮肤镜图像")
    try:
        if image_url and image_url.startswith("https://raw.githubusercontent.com/"):
            st.image(image_url, use_container_width=True, caption=f"图片ID：{current_data['image_id']}")
        else:
            st.image("https://via.placeholder.com/600x400?text=图像链接缺失", use_container_width=True)
    except Exception as e:
        st.image("https://via.placeholder.com/600x400?text=图像加载失败", use_container_width=True)
        st.warning(f"图片加载失败：{str(e)}")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 第一阶段：独立诊断")
        top1 = st.selectbox("首选 (Top-1)", ["请选择"] + ALL_CLASSES, key=f"t1_{idx}")
        top2 = st.selectbox("次选 (Top-2)", ["请选择"] + ALL_CLASSES, key=f"t2_{idx}")
        top3 = st.selectbox("备选 (Top-3)", ["请选择"] + ALL_CLASSES, key=f"t3_{idx}")
        conf_init = st.slider("初始信心（1-10分）", 1, 10, 5, key=f"c1_{idx}")
        choices = [top1, top2, top3]
        is_valid = "请选择" not in choices and len(set(choices)) == 3
        if not st.session_state.show_ai:
            if st.button("🔍 获取AI辅助建议", disabled=not is_valid):
                st.session_state.initial_top = choices
                st.session_state.initial_conf = conf_init
                st.session_state.ai_suggestion = {"label": ai_label}
                st.session_state.question_start = time.time()
                st.session_state.show_ai = True
                st.rerun()
            if not is_valid:
                st.caption("⚠️ 请完成Top-1/2/3选择（不可重复）")
    
    with col2:
        if st.session_state.show_ai:
            st.markdown("### 第二阶段：AI辅助决策")
            st.info(f"🤖 AI诊断建议：**{st.session_state.ai_suggestion['label']}**")
            initial_top1 = st.session_state.initial_top[0]
            ai_sug = st.session_state.ai_suggestion["label"]
            if initial_top1 == ai_sug:
                st.success("✅ 您的首选与AI建议一致！")
                final_top1 = initial_top1
                final_decision = "坚持原诊断（与AI一致）"
                if st.button("✅ 确认结果并进入下一题"):
                    result = {
                        **st.session_state.doctor_info,
                        "image_id": current_data["image_id"],
                        "true_label": true_label,
                        "ai_label": ai_sug,
                        "initial_top1": initial_top1,
                        "initial_top2": st.session_state.initial_top[1],
                        "initial_top3": st.session_state.initial_top[2],
                        "final_top1": final_top1,
                        "final_decision": final_decision,
                        "initial_conf": st.session_state.initial_conf,
                        "final_conf": conf_init,
                        "time_used": round(time.time() - st.session_state.question_start, 2),
                        "is_correct": (final_top1 == true_label)
                    }
                    st.session_state.user_results.append(result)
                    save_result(result)
                    reset_test_state()
                    st.session_state.current_idx += 1
                    st.rerun()
            else:
                st.warning("⚠️ 您的诊断与AI建议不一致")
                conf_final = st.slider("最终信心（1-10分）", 1, 10, 5, key=f"c2_{idx}")
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("🔄 采纳AI建议作为首选"):
                        st.session_state.final_top1 = ai_sug
                        st.session_state.final_decision = "采纳AI建议"
                with col_btn2:
                    if st.button("🛡️ 坚持我的原诊断"):
                        st.session_state.final_top1 = initial_top1
                        st.session_state.final_decision = "坚持原诊断"
                if st.session_state.final_top1:
                    if st.button("✅ 确认结果并进入下一题"):
                        result = {
                            **st.session_state.doctor_info,
                            "image_id": current_data["image_id"],
                            "true_label": true_label,
                            "ai_label": ai_sug,
                            "initial_top1": initial_top1,
                            "initial_top2": st.session_state.initial_top[1],
                            "initial_top3": st.session_state.initial_top[2],
                            "final_top1": st.session_state.final_top1,
                            "final_decision": st.session_state.final_decision,
                            "initial_conf": st.session_state.initial_conf,
                            "final_conf": conf_final,
                            "time_used": round(time.time() - st.session_state.question_start, 2),
                            "is_correct": (st.session_state.final_top1 == true_label)
                        }
                        st.session_state.user_results.append(result)
                        save_result(result)
                        reset_test_state()
                        st.session_state.current_idx += 1
                        st.rerun()

# === 6. 结果展示 ===
def result_step():
    st.title("🏁 测试完成！结果对账报告")
    results = st.session_state.user_results
    if not results:
        st.warning("暂无答题结果")
        if st.button("🔄 重新开始测试"):
            init_session_state()
            st.rerun()
        return
    correct_initial = sum([r["initial_top1"] == r["true_label"] for r in results])
    correct_final = sum([r["is_correct"] for r in results])
    initial_top1_acc = correct_initial / len(results)
    final_top1_acc = correct_final / len(results)
    
    st.subheader("📊 核心诊断指标")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("初始Top-1准确率", f"{initial_top1_acc:.1%}", f"{correct_initial}/{len(results)}")
    with col2:
        st.metric("最终Top-1准确率", f"{final_top1_acc:.1%}", f"{correct_final}/{len(results)}",
                 delta=f"{(final_top1_acc - initial_top1_acc):.1%}")
    with col3:
        adopt_ai = sum([r["final_decision"] == "采纳AI建议" for r in results])
        st.metric("AI建议采纳率", f"{adopt_ai/len(results):.1%}", f"{adopt_ai}/{len(results)}")
    with col4:
        avg_time = np.mean([r["time_used"] for r in results])
        st.metric("平均答题时间", f"{avg_time:.1f}秒")
    
    st.subheader("🎯 准确率对比分析")
    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        st.markdown("#### Top-1准确率对比")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        categories = ["初始诊断", "最终诊断"]
        accuracies = [initial_top1_acc, final_top1_acc]
        bars = ax1.bar(categories, accuracies, color=["#4285F4", "#34A853"], alpha=0.8)
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel("准确率")
        ax1.set_title("初始 vs 最终诊断准确率")
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{acc:.1%}", ha="center", fontweight="bold")
        st.pyplot(fig1)
    with col_plot2:
        st.markdown("#### Top-3准确率分析")
        top3_correct = sum([r["true_label"] in [r["initial_top1"], r["initial_top2"], r["initial_top3"]] for r in results])
        top3_acc = top3_correct / len(results)
        final_top3_correct = sum([r["true_label"] in [r["final_top1"], r["initial_top2"], r["initial_top3"]] for r in results])
        final_top3_acc = final_top3_correct / len(results)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        categories = ["初始Top-3", "最终Top-3"]
        accuracies = [top3_acc, final_top3_acc]
        bars = ax2.bar(categories, accuracies, color=["#FBBC05", "#EA4335"], alpha=0.8)
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("准确率")
        ax2.set_title("Top-3诊断准确率对比")
        for bar, acc in zip(bars, accuracies):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{acc:.1%}", ha="center", fontweight="bold")
        st.pyplot(fig2)
    
    st.subheader("🔍 诊断路径明细（彩色版）")
    table_data = []
    for idx, r in enumerate(results):
        correct_tag = "✅ 正确" if r["is_correct"] else "❌ 错误"
        decision_tag = {
            "坚持原诊断（与AI一致）": "🟢 一致",
            "坚持原诊断": "🔵 坚持",
            "采纳AI建议": "🟡 采纳AI"
        }.get(r["final_decision"], r["final_decision"])
        table_data.append({
            "序号": idx+1,
            "图片ID": r["image_id"],
            "真实标签": r["true_label"],
            "初始首选": r["initial_top1"],
            "AI建议": r["ai_label"],
            "最终首选": r["final_top1"],
            "决策类型": decision_tag,
            "是否正确": correct_tag,
            "答题时间": f"{r['time_used']}秒",
            "初始信心": f"{r['initial_conf']}分",
            "最终信心": f"{r['final_conf']}分"
        })
    st.dataframe(
        table_data,
        column_config={
            "序号": st.column_config.NumberColumn(width="small"),
            "图片ID": st.column_config.TextColumn(width="medium"),
            "真实标签": st.column_config.TextColumn(width="medium"),
            "初始首选": st.column_config.TextColumn(width="medium"),
            "AI建议": st.column_config.TextColumn(width="medium"),
            "最终首选": st.column_config.TextColumn(width="medium"),
            "决策类型": st.column_config.TextColumn(width="small"),
            "是否正确": st.column_config.TextColumn(width="small"),
            "答题时间": st.column_config.TextColumn(width="small"),
            "初始信心": st.column_config.TextColumn(width="small"),
            "最终信心": st.column_config.TextColumn(width="small")
        },
        use_container_width=True,
        hide_index=True
    )
    
    st.subheader("💾 结果导出")
    col_export1, col_export2 = st.columns(2)
    with col_export1:
        if st.button("导出详细结果CSV"):
            try:
                csv = pd.DataFrame(results).to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📥 下载CSV文件",
                    data=csv,
                    file_name=f"诊断结果_{st.session_state.doctor_id}_{time.strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"导出失败：{str(e)}")
    with col_export2:
        if st.button("🔄 重新开始测试"):
            init_session_state()
            st.rerun()

# === 主函数 ===
def main():
    init_session_state()
    if st.session_state.step == "profile":
        profile_step()
    elif st.session_state.step == "test":
        test_step()
    elif st.session_state.step == "result":
        result_step()

if __name__ == "__main__":
    main()
