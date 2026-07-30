import streamlit as st
import sys
import os
import json
import pandas as pd
import altair as alt  # 引入 Altair 进行图表定制
from pathlib import Path
from PIL import Image

# --- 路径配置 ---
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from modules.dataset_loader.loader import DatasetLoader
except ImportError as e:
    st.error(f"Unable to import DatasetLoader. Please ensure the project structure is correct.\nError: {e}")
    st.stop()

# --- 页面全局配置 ---
st.set_page_config(
    page_title="GSI Dataset Workbench",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 自定义 CSS (美化) ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #4e8cff;
    }
    .metric-card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    /* Adjust JSON display font */
    .stJson {
        font-family: 'Consolas', 'Courier New', monospace;
    }
    /* Adjust Tag style */
    .level-tag {
        display: inline-block;
        background-color: #e8f0fe;
        color: #1967d2;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        margin-right: 4px;
        border: 1px solid #d2e3fc;
    }
</style>
""", unsafe_allow_html=True)

# --- 常量定义 ---
# 协作等级描述
COOR_LEVEL_DESC = {
    "L0": "L0: No Coordination",
    "L1": "L1: Weak Homogeneous Coordination",
    "L2": "L2: Weak Heterogeneous Coordination",
    "L3": "L3: Strong Homogeneous Coordination",
    "L4": "L4: Strong Heterogeneous Coordination"
}

# 语言等级描述
LANG_LEVEL_DESC = {
    "L0": "L0: Explicit Step-by-Step Instructions (Procedural)",
    "L1": "L1: Standard Direct Instructions (Standard)",
    "L2": "L2: Abstract Intent Instructions (Abstract)"
}

# --- 会话状态管理 ---
if "loader" not in st.session_state:
    st.session_state.loader = None
if "df_meta" not in st.session_state:
    st.session_state.df_meta = None
if "filter_opts" not in st.session_state:
    st.session_state.filter_opts = None

# --- 辅助函数 ---
def ensure_list(x):
    """Ensure data is in list format, used for processing plan_level/coor_level"""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [x]
    return []

# --- 核心：预加载与缓存系统 ---

@st.cache_resource(show_spinner=False)
def initialize_loader_resources(repo_id, type_name, token, revision):
    """
    [Core Optimization] Load all resources at once and pre-compute filter options.
    """
    print(f"🚀 [System] Initializing Loader for {repo_id}...")
    
    # 1. 初始化 Loader 并构建索引 (IO Heavy)
    loader = DatasetLoader(repo_id=repo_id, type_name=type_name, token=token, revision=revision)
    loader._build_metadata_index()
    
    # 2. 构建 DataFrame (CPU Heavy)
    df = pd.DataFrame(loader._meta_index) if loader._meta_index else pd.DataFrame()
    
    # 3. 数据清洗 (CPU Heavy)
    if not df.empty:
        for col in ['plan_level', 'coor_level']:
            if col in df.columns:
                df[col] = df[col].apply(ensure_list)
            else:
                df[col] = [[] for _ in range(len(df))]
        
        # 确保 language_level 存在
        if 'language_level' not in df.columns:
            df['language_level'] = "N/A"
    
    # 4. 预计算筛选选项 (避免 UI 渲染时的卡顿)
    filter_options = {}
    if not df.empty:
        filter_options["types"] = sorted(df['goal_type'].unique().tolist())
        # Flatten lists and set for Plan/Coor
        filter_options["plans"] = sorted(list(set([i for s in df['plan_level'] for i in s])))
        filter_options["coors"] = sorted(list(set([i for s in df['coor_level'] for i in s])))
        filter_options["langs"] = sorted(df['language_level'].unique().tolist())
    else:
        filter_options = {"types": [], "plans": [], "coors": [], "langs": []}

    return loader, df, filter_options

# --- 初始化逻辑 (自动启动) ---

# 默认配置
DEFAULT_CONFIG = {
    "repo_id": "WindyLab/GSI",
    "type_name": "cybertown",
    "token": None,
    "revision": "main"
}

# 侧边栏配置区
with st.sidebar:
    st.title("🌌 GSI Workbench")
    
    with st.expander("🛠️ Connection Config", expanded=False):
        repo_id = st.text_input("Repo ID", DEFAULT_CONFIG["repo_id"])
        type_name = st.text_input("Type", DEFAULT_CONFIG["type_name"])
        token = st.text_input("Token", DEFAULT_CONFIG["token"], type="password")
        revision = st.text_input("Revision", DEFAULT_CONFIG["revision"])
        
        if st.button("🔄 Force Reconnect / Refresh Data"):
            initialize_loader_resources.clear()
            st.cache_resource.clear()
            st.rerun()

    st.divider()
    
    selected_page = st.radio("Navigation", ["📊 Dashboard", "🔍 Data Browser"], index=0)
    st.divider()

# --- 自动加载执行 ---
if "loader" not in st.session_state or st.session_state.loader is None:
    try:
        with st.spinner("🚀 System starting: Connecting to dataset and preprocessing index... (First run may be slow)"):
            loader_inst, df_inst, opts_inst = initialize_loader_resources(
                repo_id, type_name, token if token else None, revision
            )
            
            st.session_state.loader = loader_inst
            st.session_state.df_meta = df_inst
            st.session_state.filter_opts = opts_inst
            
            st.toast("System ready!", icon="✅")
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.stop()

loader = st.session_state.loader
df_raw = st.session_state.df_meta
opts = st.session_state.filter_opts

# 底部状态栏
with st.sidebar:
    if not df_raw.empty:
        st.caption(f"✅ Loaded {len(df_raw)} task entries")
        st.caption(f"🔗 `{repo_id}`")

# ==============================================================================
# 页面 1: 📊 仪表盘
# ==============================================================================
if selected_page == "📊 Dashboard":
    st.header("📊 Interactive Dataset Overview")
    
    if df_raw.empty:
        st.warning("Dataset is empty, please check configuration.")
    else:
        # --- 交叉筛选区 ---
        with st.container(border=True):
            st.markdown("##### 🕵️ Cross-Filter")
            f1, f2, f3, f4, f5 = st.columns([1, 1, 1, 1, 0.8])
            
            # 使用缓存的 options
            sel_type = f1.multiselect("Goal Type", opts["types"], placeholder="All")
            sel_plan = f2.multiselect("Plan Level (contains)", opts["plans"], placeholder="All")
            
            # 使用 format_func 显示含义
            sel_coor = f3.multiselect("Coor Level (contains)", opts["coors"], placeholder="All",
                                      format_func=lambda x: COOR_LEVEL_DESC.get(x, x))
            
            # Language Level 使用 format_func
            sel_lang = f4.multiselect("Language Level", opts["langs"], placeholder="All",
                                      format_func=lambda x: LANG_LEVEL_DESC.get(x, x))
            
            # 本地过滤 (联动核心)
            df_view = df_raw.copy()
            if sel_type: 
                df_view = df_view[df_view['goal_type'].isin(sel_type)]
            if sel_plan: 
                # 列表包含逻辑
                df_view = df_view[df_view['plan_level'].apply(lambda x: not set(x).isdisjoint(sel_plan))]
            if sel_coor: 
                df_view = df_view[df_view['coor_level'].apply(lambda x: not set(x).isdisjoint(sel_coor))]
            if sel_lang:
                df_view = df_view[df_view['language_level'].isin(sel_lang)]
            
            f5.metric("Currently Selected Tasks", f"{len(df_view)}", delta=f"{len(df_view)-len(df_raw)}" if len(df_view)!=len(df_raw) else None)

        st.markdown("---")

        # --- 图表区 ---
        r1_1, r1_2 = st.columns(2)
        
        # 1. 任务类型 (Altair 定制)
        with r1_1:
            st.subheader("🎯 Task Type Distribution")
            if not df_view.empty:
                t_counts = df_view['goal_type'].value_counts().reset_index()
                t_counts.columns = ['Goal Type', 'Count']
                
                chart = alt.Chart(t_counts).mark_bar(color="#4e8cff").encode(
                    x=alt.X('Goal Type', 
                            axis=alt.Axis(labelAngle=-30, labelFontWeight='bold', labelOverlap=False, labelLimit=0), 
                            sort='-y'),
                    y='Count', 
                    tooltip=['Goal Type', 'Count']
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No data")

        # 2. Plan Level (Exploded)
        with r1_2:
            st.subheader("🧠 Planning Difficulty (Plan Level)")
            if not df_view.empty:
                df_xp = df_view.explode('plan_level')
                p_counts = df_xp['plan_level'].value_counts().reset_index()
                p_counts.columns = ['Level', 'Count']
                
                chart_p = alt.Chart(p_counts).mark_bar(color="#ff4b4b").encode(
                    x=alt.X('Level', sort='x'), 
                    y='Count', 
                    tooltip=['Level', 'Count']
                ).properties(height=300)
                st.altair_chart(chart_p, use_container_width=True)
            else:
                st.info("No data")

        # 第二行图表
        r2_1, r2_2 = st.columns(2)

        # 3. Coor Level (Exploded)
        with r2_1:
            st.subheader("🤝 Coordination Difficulty (Coor Level)")
            if not df_view.empty:
                df_xc = df_view.explode('coor_level')
                c_counts = df_xc['coor_level'].value_counts().reset_index()
                c_counts.columns = ['Level', 'Count']
                
                chart_c = alt.Chart(c_counts).mark_bar(color="#50c878").encode(
                    x=alt.X('Level', sort='x'), 
                    y='Count', 
                    tooltip=['Level', 'Count']
                ).properties(height=300)
                st.altair_chart(chart_c, use_container_width=True)
            else:
                st.info("No data")

        # 4. Language Level
        with r2_2:
            st.subheader("🗣️ Language Instruction Level (Language Level)")
            if not df_view.empty and 'language_level' in df_view.columns:
                l_counts = df_view['language_level'].value_counts().reset_index()
                l_counts.columns = ['Level', 'Count']
                
                chart_l = alt.Chart(l_counts).mark_bar(color="#ffaa00").encode(
                    x=alt.X('Level', sort='x'),
                    y='Count',
                    tooltip=['Level', 'Count']
                ).properties(height=300)
                st.altair_chart(chart_l, use_container_width=True)
            else:
                st.info("No data or no language level field")

# ==============================================================================
# 页面 2: 🔍 数据浏览
# ==============================================================================
elif selected_page == "🔍 Data Browser":
    st.header("🔍 Data Browser & Details")

    # --- 筛选栏 ---
    with st.expander("🌪️ Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        s_type = c1.multiselect("Goal Type", opts["types"])
        s_plan = c2.multiselect("Plan Level", opts["plans"])
        s_coor = c3.multiselect("Coor Level", opts["coors"], format_func=lambda x: COOR_LEVEL_DESC.get(x, x))
        s_text = c4.text_input("Task ID Search")

    # 执行筛选
    df_f = df_raw.copy()
    if s_type: df_f = df_f[df_f['goal_type'].isin(s_type)]
    if s_plan: df_f = df_f[df_f['plan_level'].apply(lambda x: not set(x).isdisjoint(s_plan))]
    if s_coor: df_f = df_f[df_f['coor_level'].apply(lambda x: not set(x).isdisjoint(s_coor))]
    if s_text: df_f = df_f[df_f['task_id'].str.contains(s_text, case=False)]

    # --- 选择列表 ---
    col_sel, col_cnt = st.columns([3, 1])
    with col_cnt:
        st.metric("Results", len(df_f))
    
    with col_sel:
        # 性能保护
        all_opts = df_f['task_id'].tolist()
        view_opts = all_opts[:200]
        if len(all_opts) > 200:
            st.caption(f"⚠️ Too many matching results ({len(all_opts)}), showing only the first 200. Please use filters to narrow down.")
        
        selected_tid = st.selectbox("Select Task ID", view_opts)

    st.divider()

    # --- 详情视图 ---
    if selected_tid:
        with st.spinner("Loading task details..."):
            data = loader.get_task(selected_tid, lazy=True, include_prompt=True, include_scenario=True)
        
        if data:
            c_main, c_info = st.columns([2.5, 1.0])

            # 左侧：Prompt 信息 + Raw Data
            with c_main:
                # [修改] 增加 Raw Data Tab
                t_view, t_seg, t_raw = st.tabs(["📝 Prompt Preview", "🧩 Prompt Segments", "⚙️ Raw Data"])
                
                with t_view:
                    p_str = data.get("prompt_data", {}).get("prompt", "")
                    if p_str:
                        st.text_area("Full Prompt", p_str, height=850)
                    else:
                        st.warning("Prompt Empty")
                
                with t_seg:
                    st.info("Raw segments from text pool:")
                    st.json(data.get("prompt_data", {}).get("segments", {}), expanded=True)
                
                # Raw JSON View
                with t_raw:
                    st.caption(f"API: loader.get_task('{selected_tid}') full return data")
                    st.json(data)

            # 右侧：场景与元数据
            with c_info:
                # 场景图
                st.subheader("🗺️ Scene")
                sid = data.get("task", {}).get("scenario")
                if sid and loader._local_root:
                    img = loader._local_root / "scenarios" / type_name / sid / "scene.png"
                    if img.exists():
                        st.image(str(img), caption=sid, use_container_width=True)
                    else:
                        st.caption("No Scene Image")
                
                # Meta Tags
                st.subheader("🏷️ Metadata")
                with st.container(border=True):
                    row = df_raw[df_raw['task_id'] == selected_tid].iloc[0]
                    
                    def tag_html(lst):
                        return " ".join([f"<span class='level-tag'>{x}</span>" for x in lst])

                    st.markdown(f"**Plan Level:** {tag_html(row['plan_level'])}", unsafe_allow_html=True)
                    st.markdown(f"**Coor Level:** {tag_html(row['coor_level'])}", unsafe_allow_html=True)
                    st.markdown(f"**Lang Level:** <span class='level-tag'>{row.get('language_level', 'N/A')}</span>", unsafe_allow_html=True)
                    
                    st.divider()
                    st.markdown("**Instruction:**")
                    st.info(data.get("goal_details", {}).get("instruction", "N/A"))
                    
                    with st.expander("Full Goal JSON"):
                        st.json(data.get("goal_details", {}))