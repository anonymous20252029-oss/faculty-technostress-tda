import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import io
import json
import base64
import requests
import urllib.parse
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import qrcode

# --- 1. CẤU HÌNH GIAO DIỆN TINH GỌN ---
st.set_page_config(
    page_title="Khảo Sát & Định Vị Thích Ứng Số",
    page_icon="🌱",
    layout="centered"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        max-width: 780px !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    h3, h4, h5 {
        margin-top: 0.2rem !important;
        margin-bottom: 0.3rem !important;
    }
    p, label {
        font-size: 0.95rem !important;
    }
    .stRadio > div {
        gap: 0.3rem !important;
    }
    .card-box {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
        border: 1px solid #e2e8f0;
    }
    .group-badge {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 5px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

DATA_GLOBAL_PATH = "data/global_aligned_real_dataset.csv"
BASE_URL = "https://faculty-technostress-tda-mmmrqgaetftbvoqpdigqmm.streamlit.app"

# Nhận diện tham số URL
url_group = st.query_params.get("group", None)
if url_group:
    url_group = urllib.parse.unquote(url_group).strip()

url_page = st.query_params.get("view", "survey")

# Khởi tạo mốc thời gian telemetry
if 'dt_start' not in st.session_state:
    st.session_state.dt_start = datetime.now()
    st.session_state.t0 = time.time()
if 't_q1' not in st.session_state:
    st.session_state.t_q1 = None
if 't_q2' not in st.session_state:
    st.session_state.t_q2 = None
if 't_q3' not in st.session_state:
    st.session_state.t_q3 = None
if 'latest_user' not in st.session_state:
    st.session_state.latest_user = None

def on_select_q1():
    st.session_state.t_q1 = time.time()

def on_select_q2():
    st.session_state.t_q2 = time.time()

def on_select_q3():
    st.session_state.t_q3 = time.time()

# --- 2. GITHUB API: ĐỌC VÀ LƯU DỮ LIỆU ---
def get_github_data():
    if "github" not in st.secrets:
        local_path = "data/pilot_survey_cntt_30_responses.csv"
        if os.path.exists(local_path):
            return pd.read_csv(local_path), None
        return pd.DataFrame(), None

    gh = st.secrets["github"]
    url = f"https://api.github.com/repos/{gh['repo']}/contents/{gh['file_path']}?ref={gh['branch']}"
    headers = {
        "Authorization": f"Bearer {gh['token']}",
        "Accept": "application/vnd.github.v3+json"
    }
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            content_json = res.json()
            sha = content_json.get("sha")
            csv_content = base64.b64decode(content_json["content"]).decode("utf-8")
            df = pd.read_csv(io.StringIO(csv_content))
            return df, sha
    except Exception:
        pass
    return pd.DataFrame(), None

def commit_to_github(new_record):
    df, sha = get_github_data()
    df_new = pd.DataFrame([new_record])
    df_combined = df_new if df.empty else pd.concat([df, df_new], ignore_index=True)
    csv_str = df_combined.to_csv(index=False)
    content_b64 = base64.b64encode(csv_str.encode("utf-8")).decode("utf-8")
    
    if "github" in st.secrets:
        gh = st.secrets["github"]
        url = f"https://api.github.com/repos/{gh['repo']}/contents/{gh['file_path']}"
        headers = {
            "Authorization": f"Bearer {gh['token']}",
            "Accept": "application/vnd.github.v3+json"
        }
        payload = {
            "message": f"feat: record response group {new_record['group_id']}",
            "content": content_b64,
            "branch": gh["branch"]
        }
        if sha:
            payload["sha"] = sha
        res = requests.put(url, headers=headers, json=payload)
        return res.status_code in [200, 201], df_combined
    else:
        os.makedirs("data", exist_ok=True)
        df_combined.to_csv("data/pilot_survey_cntt_30_responses.csv", index=False)
        return True, df_combined

# --- 3. LOAD DỮ LIỆU NỀN TOÀN CẦU (N=3.459) & PCA ---
@st.cache_resource
def load_and_fit_pca():
    if os.path.exists(DATA_GLOBAL_PATH):
        df_global = pd.read_csv(DATA_GLOBAL_PATH)
    else:
        df_global = pd.DataFrame(
            np.random.uniform(1.0, 5.0, (3459, 3)),
            columns=['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']
        )
    feature_cols = ['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_global[feature_cols].values)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    df_global['PC1'] = X_2d[:, 0]
    df_global['PC2'] = X_2d[:, 1]
    
    df_bg_sample = df_global.sample(n=min(550, len(df_global)), random_state=42)
    return df_global, df_bg_sample, scaler, pca

df_global, df_bg_sample, scaler, pca = load_and_fit_pca()

def generate_qr_image(link_url):
    qr = qrcode.QRCode(box_size=6, border=2)
    qr.add_data(link_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# --- 4. LUẬN GIẢI CHUYÊN SÂU (CÁ NHÂN, ĐẠI DIỆN NHÓM, VÀ SUGGESTIONS) ---
def synthesize_profiles(u_f1, u_f2, u_f3, df_grp, grp_label):
    # A. Hồ sơ Cá nhân
    if u_f2 >= 3.5 and u_f1 >= 3.5:
        p_zone = "Vùng Quá Tải Nhịp Độ Số & Mệt Mỏi"
        p_meaning = "Bạn đang vận hành nhận thức ở cường độ cao trước sự dồn dập của công nghệ song song với áp lực công việc hàng ngày, khiến tài nguyên phục hồi suy giảm."
        p_tips = ["Thiết lập ranh giới số rõ ràng ngoài giờ làm việc.", "Nghỉ mắt 5 phút sau mỗi 45 phút tập trung vào màn hình.", "Giảm bớt xử lý đa nhiệm, ưu tiên từng việc dứt điểm."]
        p_col = "#e11d48"
    elif u_f2 >= 3.5 and u_f1 < 3.5:
        p_zone = "Vùng Áp Lực Thích Ứng Công Nghệ"
        p_meaning = "Năng lượng nền tảng của bạn còn tốt, nhưng việc phải liên tục cập nhật công cụ/phần mềm mới tạo ra ma sát nhận thức đáng kể."
        p_tips = ["Chỉ chọn lọc dùng 1-2 công cụ AI/số thiết thực nhất.", "Học hỏi mẹo thao tác từ người có kinh nghiệm để giảm thời gian tự mò mẫm.", "Cho phép bản thân có nhịp độ thích ứng tự nhiên."]
        p_col = "#d97706"
    elif u_f1 >= 3.5 and u_f2 < 3.5:
        p_zone = "Vùng Mệt Mỏi Cần Tái Tạo Năng Lượng"
        p_meaning = "Áp lực không đến nhiều từ công cụ số mà chủ yếu do khối lượng công việc và sinh hoạt dồn dập khiến cơ thể mệt mỏi."
        p_tips = ["Ưu tiên chất lượng giấc ngủ và thư giãn cá nhân.", "Chủ động lùi hạn các đầu việc không cấp bách.", "Dành thời gian đi dạo hoặc vận động nhẹ ngoài trời."]
        p_col = "#ea580c"
    else:
        p_zone = "Vùng Cân Bằng Thích Ứng Ổn Định"
        p_meaning = "Bạn duy trì sự điều hòa rất tốt giữa năng lượng cá nhân và nhịp độ công nghệ, làm chủ công cụ mà không bị căng thẳng."
        p_tips = ["Tiếp tục duy trì nhịp làm việc và sinh hoạt khoa học hiện tại.", "Chia sẻ kinh nghiệm làm việc hiệu quả với các thành viên khác.", "Chủ động nhận biết sớm dấu hiệu mệt mỏi vào các tuần cao điểm."]
        p_col = "#16a34a"

    # B. Hồ sơ Đại diện Nhóm
    avg_f1 = df_grp['Burnout_Level'].mean()
    avg_f2 = df_grp['AI_Technostress'].mean()
    n_cnt = len(df_grp)
    
    if avg_f2 >= 3.5 and avg_f1 >= 3.5:
        g_desc = f"Nhóm **{grp_label}** ({n_cnt} thành viên) đang có xu hướng chung rơi vào vùng quá tải nhịp độ số. Áp lực công việc kết hợp tốc độ chuyển đổi số tạo ra căng thẳng diện rộng."
        g_sug = "Đơn vị nên rà soát lại hạn ngạch nhiệm vụ, tinh giản các quy trình số hóa rườm rà và tổ chức tập huấn công cụ bài bản."
    elif avg_f2 >= 3.5:
        g_desc = f"Nhóm **{grp_label}** ({n_cnt} thành viên) có mức chịu tải công việc ổn định nhưng đang gặp điểm nghẽn chính ở việc thích ứng công nghệ mới."
        g_sug = "Nên tạo diễn đàn nội bộ để các thành viên thành thạo hướng dẫn kèm cặp đồng nghiệp, giảm áp lực tự mày mò."
    elif avg_f1 >= 3.5:
        g_desc = f"Nhóm **{grp_label}** ({n_cnt} thành viên) nhìn chung làm chủ công nghệ tốt nhưng khối lượng công việc tổng thể đang ở mức cao, gây mệt mỏi."
        g_sug = "Cần phân bổ lại tiến độ công việc linh hoạt, khuyến khích các hoạt động gắn kết và tái tạo năng lượng tập thể."
    else:
        g_desc = f"Nhóm **{grp_label}** ({n_cnt} thành viên) đang duy trì trạng thái vận hành rất hài hòa và cân bằng so với mặt bằng chung."
        g_sug = "Tiếp tục phát huy mô hình làm việc hiện tại, duy trì môi trường trao đổi cởi mở."

    return p_zone, p_meaning, p_tips, p_col, g_desc, g_sug

# --- 5. SIDEBAR: TẠO QR NHÓM ---
with st.sidebar:
    st.markdown("### 🔗 Tạo Link & Mã QR Nhóm")
    new_group_name = st.text_input("Nhập tên nhóm muốn tạo:", placeholder="Ví dụ: KTPM, KHMT, Nhom_1")
    if new_group_name:
        encoded_grp = urllib.parse.quote(new_group_name.strip())
        full_group_url = f"{BASE_URL}/?group={encoded_grp}"
        st.success(f"Link: **{new_group_name}**")
        st.code(full_group_url, language="text")
        
        qr_bytes = generate_qr_image(full_group_url)
        st.image(qr_bytes, caption=f"QR Nhóm: {new_group_name}", use_container_width=True)
        st.download_button("📥 Tải QR về máy", qr_bytes, file_name=f"QR_{new_group_name}.png", mime="image/png")

# ==============================================================================
# TRANG 1: PHIẾU KHẢO SÁT
# ==============================================================================
if url_page != "result":
    st.markdown("### 🌱 Khảo Sát Nhịp Độ Làm Việc & Thích Ứng")
    st.caption("3 câu hỏi ngắn • Ẩn danh • Tự động định vị trạng thái")
    
    if url_group:
        st.markdown(f'<div class="group-badge">🔒 Nhóm tham gia: <b>{url_group}</b></div>', unsafe_allow_html=True)
        assigned_group = url_group
    else:
        assigned_group = st.text_input("Tên nhóm tham gia (để trống nếu tham gia cá nhân):", value="Chung").strip()
        if not assigned_group:
            assigned_group = "Chung"

    q1_opts = {
        "1. Rất thoải mái, tràn đầy năng lượng": 1.0,
        "2. Hơi mệt mỏi nhưng hồi phục nhanh": 2.0,
        "3. Thỉnh thoảng cạn kiệt sức sau giờ làm": 3.0,
        "4. Thường xuyên mệt mỏi, giảm hứng thú": 4.0,
        "5. Kiệt sức kéo dài, rất khó phục hồi": 5.0
    }
    q1_sel = st.radio("1. Mức độ mệt mỏi / hao mòn sức lực gần đây:", list(q1_opts.keys()), index=1, key="rad_q1", on_change=on_select_q1)
    val_f1 = q1_opts[q1_sel]

    q2_opts = {
        "1. Dễ dàng làm chủ, không thấy áp lực": 1.0,
        "2. Thỉnh thoảng mất chút thời gian làm quen": 2.0,
        "3. Cảm thấy nhịp độ công nghệ khá dồn dập": 3.0,
        "4. Thường xuyên căng thẳng vì phần mềm/AI mới": 4.0,
        "5. Quá tải, cảm giác liên tục bị thúc ép": 5.0
    }
    q2_sel = st.radio("2. Áp lực phải thích nghi với phần mềm / công cụ mới:", list(q2_opts.keys()), index=2, key="rad_q2", on_change=on_select_q2)
    val_f2 = q2_opts[q2_sel]

    q3_opts = {
        "1. Rất chủ động, luôn có cách cân bằng tốt": 1.0,
        "2. Thích ứng ổn định, ít khi bế tắc": 2.0,
        "3. Đôi khi bối rối, cần nhiều thời gian suy nghĩ": 3.0,
        "4. Khó cân bằng, hay đắn đo và trì hoãn việc": 4.0,
        "5. Rất khó khăn trong việc tự điều hòa áp lực": 5.0
    }
    q3_sel = st.radio("3. Khả năng tự điều hòa khi gặp công việc dồn dập:", list(q3_opts.keys()), index=1, key="rad_q3", on_change=on_select_q3)
    val_f3 = q3_opts[q3_sel]

    st.write("")
    btn_submit = st.button("🚀 Gửi & Xem Định Vị Của Bạn", use_container_width=True, type="primary")

    if btn_submit:
        t_end = time.time()
        dt_end = datetime.now()
        
        t0 = st.session_state.t0
        t1 = st.session_state.t_q1 if st.session_state.t_q1 else t0 + (t_end - t0) * 0.33
        t2 = st.session_state.t_q2 if st.session_state.t_q2 else t1 + (t_end - t1) * 0.5
        t3 = st.session_state.t_q3 if st.session_state.t_q3 else t2 + (t_end - t2) * 0.5

        dt_q1 = round(max(0.5, t1 - t0), 2)
        dt_q2 = round(max(0.5, t2 - t1), 2)
        dt_q3 = round(max(0.5, t3 - t2), 2)
        dt_total = round(t_end - t0, 2)

        time_score = 1.0 + 4.0 * min(1.0, max(0.0, (np.log(1 + dt_total) - np.log(6)) / (np.log(45) - np.log(6))))
        final_f3 = round(0.5 * (val_f3 + time_score), 2)

        record = {
            'timestamp': dt_end.strftime("%Y-%m-%d %H:%M:%S"),
            'date': dt_end.strftime("%Y-%m-%d"),
            'year': dt_end.year,
            'month': dt_end.month,
            'day': dt_end.day,
            'start_time': st.session_state.dt_start.strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': dt_end.strftime("%Y-%m-%d %H:%M:%S"),
            'group_id': assigned_group,
            'Burnout_Level': val_f1,
            'AI_Technostress': val_f2,
            'Cognitive_Latency_Proxy': final_f3,
            'Coping_Score': val_f3,
            't1_sec': dt_q1,
            't2_sec': dt_q2,
            't3_sec': dt_q3,
            'total_latency_sec': dt_total
        }

        with st.spinner("Đang ghi nhận kết quả..."):
            commit_to_github(record)

        st.session_state.latest_user = record
        st.query_params["view"] = "result"
        if assigned_group:
            st.query_params["group"] = assigned_group
            
        st.session_state.dt_start = datetime.now()
        st.session_state.t0 = time.time()
        st.session_state.t_q1 = None
        st.session_state.t_q2 = None
        st.session_state.t_q3 = None
        st.rerun()

# ==============================================================================
# TRANG 2: KẾT QUẢ ĐỊNH VỊ (MÔ TẢ CÁ NHÂN, ĐẠI DIỆN NHÓM, VÀ SUGGESTIONS)
# ==============================================================================
else:
    c_btn1, _ = st.columns([1.2, 3])
    with c_btn1:
        if st.button("⬅️ Làm Lại Phiếu", use_container_width=True):
            st.query_params["view"] = "survey"
            st.session_state.dt_start = datetime.now()
            st.session_state.t0 = time.time()
            st.rerun()

    df_resp, _ = get_github_data()

    if not df_resp.empty:
        if st.session_state.latest_user is None:
            st.session_state.latest_user = df_resp.iloc[-1].to_dict()

        u_f1 = float(st.session_state.latest_user.get('Burnout_Level', 2.0))
        u_f2 = float(st.session_state.latest_user.get('AI_Technostress', 2.0))
        u_f3 = float(st.session_state.latest_user.get('Cognitive_Latency_Proxy', 2.0))
        u_grp = str(st.session_state.latest_user.get('group_id', url_group if url_group else "Chung"))

        # Chọn nhóm hiển thị
        available_groups = ["Tất cả nhóm"] + sorted(list(df_resp['group_id'].dropna().astype(str).unique()))
        sel_idx = available_groups.index(u_grp) if u_grp in available_groups else 0
        
        chosen_grp = st.selectbox("Xem dữ liệu nhóm:", available_groups, index=sel_idx)
        
        if chosen_grp == "Tất cả nhóm":
            df_plot = df_resp
            grp_name = "Toàn thể thành viên"
        else:
            df_plot = df_resp[df_resp['group_id'].astype(str) == chosen_grp]
            grp_name = chosen_grp

        # Tổng hợp 3 khối mô tả học thuật
        p_zone, p_meaning, p_tips, p_col, g_desc, g_sug = synthesize_profiles(u_f1, u_f2, u_f3, df_plot, grp_name)

        # -------------------------------------------------------------
        # KHỐI 1: BẢN ĐỒ TÔ-PÔ TỐI GIẢN (ẨN SẠCH NÚT RƯỜM RÀ, ZOOM DỄ DÀNG)
        # -------------------------------------------------------------
        sample_scaled = scaler.transform(df_plot[['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']].values)
        sample_2d = pca.transform(sample_scaled)

        u_scaled = scaler.transform([[u_f1, u_f2, u_f3]])
        u_2d = pca.transform(u_scaled)[0]

        fig = go.Figure()
        # 1. Nền toàn cầu
        fig.add_trace(go.Scatter(
            x=df_bg_sample['PC1'], y=df_bg_sample['PC2'],
            mode='markers', marker=dict(size=5, color='#cbd5e1', opacity=0.4),
            name='Chuẩn cộng đồng (N=3.459)', hoverinfo='skip'
        ))
        # 2. Thành viên nhóm
        fig.add_trace(go.Scatter(
            x=sample_2d[:, 0], y=sample_2d[:, 1],
            mode='markers', marker=dict(size=7, color='#f43f5e', opacity=0.8),
            name=f'Thành viên {grp_name}', hoverinfo='name'
        ))
        # 3. Tâm đại diện nhóm
        if len(sample_2d) > 0:
            c_grp = sample_2d.mean(axis=0)
            fig.add_trace(go.Scatter(
                x=[c_grp[0]], y=[c_grp[1]],
                mode='markers', marker=dict(symbol='star', size=16, color='#fbbf24', line=dict(color='black', width=1.2)),
                name=f'Tâm nhóm {grp_name}', hoverinfo='name'
            ))
        # 4. Điểm cá nhân của bạn
        fig.add_trace(go.Scatter(
            x=[u_2d[0]], y=[u_2d[1]],
            mode='markers+text', marker=dict(size=17, color='#06b6d4', line=dict(color='#083344', width=2.5)),
            text=["📍 Bạn ở đây"], textposition="top center",
            textfont=dict(color="#083344", size=12), name='Vị trí của bạn', hoverinfo='text'
        ))

        fig.update_layout(
            title=dict(text="Bản Đồ Không Gian Trạng Thái Thích Ứng", font=dict(size=12.5)),
            xaxis=dict(title="Trục thích ứng 1", showgrid=True, zeroline=False),
            yaxis=dict(title="Trục thích ứng 2", showgrid=True, zeroline=False),
            margin=dict(l=10, r=10, t=35, b=25),
            height=340,
            dragmode='pan',
            legend=dict(orientation="h", yanchor="bottom", y=-0.36, xanchor="center", x=0.5, font=dict(size=9)),
            template="plotly_white"
        )

        # CẤU HÌNH GỌN NHẤT: ẨN TOÀN BỘ NÚT BẤM, CHỈ CHO ZOOM CẢM ỨNG & DOUBLE-CLICK ĐỂ RESET
        plotly_clean_config = {
            'displayModeBar': False,  # ẨN HOÀN TOÀN THANH CÔNG CỤ NHIỀU NÚT
            'scrollZoom': True,       # Chụm/mở 2 ngón tay hoặc lăn chuột để phóng to/thu nhỏ
            'doubleClick': 'reset'    # Chạm đúp (double-click) là thu nhỏ lại góc nhìn gốc
        }
        st.plotly_chart(fig, use_container_width=True, config=plotly_clean_config)
        st.caption("🔍 *Mẹo xem hình:* Chụm 2 ngón tay (hoặc lăn chuột) để phóng to • Chạm đúp (hoặc nhấp đúp) để thu nhỏ về ban đầu.")

        # -------------------------------------------------------------
        # KHỐI 2: MÔ TẢ ĐỊNH VỊ CÁ NHÂN
        # -------------------------------------------------------------
        st.markdown(f"""
        <div class="card-box" style="border-left: 5px solid {p_col};">
            <h4 style="color: {p_col}; margin: 0 0 6px 0;">👤 1. Định Vị Cá Nhân: {p_zone}</h4>
            <p style="color: #334155; margin-bottom: 0; line-height: 1.5;">{p_meaning}</p>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------------------------------------------
        # KHỐI 3: MÔ TẢ TRẠNG THÁI ĐẠI DIỆN NHÓM
        # -------------------------------------------------------------
        st.markdown(f"""
        <div class="card-box" style="border-left: 5px solid #6366f1;">
            <h4 style="color: #4f46e5; margin: 0 0 6px 0;">👥 2. Trạng Thái Đại Diện Nhóm: {grp_name}</h4>
            <p style="color: #334155; margin-bottom: 6px; line-height: 1.5;">{g_desc}</p>
            <p style="color: #475569; font-size: 0.9rem; margin-bottom: 0;">
                • Điểm mệt mỏi trung bình: <b>{df_plot['Burnout_Level'].mean():.2f}/5.0</b> | 
                Áp lực công nghệ: <b>{df_plot['AI_Technostress'].mean():.2f}/5.0</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------------------------------------------
        # KHỐI 4: GỢI Ý ĐIỀU HÒA & KHUYẾN NGHỊ (SUGGESTIONS)
        # -------------------------------------------------------------
        st.markdown(f"""
        <div class="card-box" style="border-left: 5px solid #10b981;">
            <h4 style="color: #059669; margin: 0 0 6px 0;">💡 3. Gợi Ý Điều Hòa & Khuyến Nghị (Suggestions)</h4>
            <p style="color: #1e293b; font-weight: 600; margin-bottom: 4px;">Dành cho bản thân bạn:</p>
            <ul style="margin: 0 0 8px 0; padding-left: 18px; color: #334155; font-size: 0.9rem;">
                {''.join([f"<li>{t}</li>" for t in p_tips])}
            </ul>
            <p style="color: #1e293b; font-weight: 600; margin-bottom: 4px;">Dành cho đơn vị / quản lý nhóm:</p>
            <p style="color: #334155; font-size: 0.9rem; margin-bottom: 0; padding-left: 6px;">{g_sug}</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Chưa có dữ liệu nào trên GitHub. Vui lòng quay lại gửi phiếu khảo sát đầu tiên.")
