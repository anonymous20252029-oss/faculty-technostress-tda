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

# --- 1. CẤU HÌNH GIAO DIỆN & RESPONSIVE CSS ---
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
    .result-box {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 12px 14px;
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

# Nhận diện nhóm & view từ URL
url_group = st.query_params.get("group", None)
if url_group:
    url_group = urllib.parse.unquote(url_group).strip()

url_page = st.query_params.get("view", "survey")

# Khởi tạo trạng thái mốc thời gian bắt đầu
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

# --- 2. HÀM TƯƠNG TÁC GITHUB API: ĐỌC VÀ LƯU DỮ LIỆU VĨNH VIỄN ---
def get_github_data():
    """Lấy dữ liệu CSV trực tiếp từ GitHub repository"""
    if "github" not in st.secrets:
        # Dự phòng đọc local nếu đang chạy offline
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
        elif res.status_code == 404:
            return pd.DataFrame(), None
    except Exception as e:
        st.error(f"Lỗi kết nối GitHub: {e}")
    return pd.DataFrame(), None

def commit_to_github(new_record):
    """Thêm dòng dữ liệu mới và commit trực tiếp lên GitHub"""
    df, sha = get_github_data()
    df_new = pd.DataFrame([new_record])
    
    if df.empty:
        df_combined = df_new
    else:
        df_combined = pd.concat([df, df_new], ignore_index=True)
        
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
            "message": f"feat: add survey response for group {new_record['group_id']} at {new_record['end_time']}",
            "content": content_b64,
            "branch": gh["branch"]
        }
        if sha:
            payload["sha"] = sha
            
        res = requests.put(url, headers=headers, json=payload)
        return res.status_code in [200, 201], df_combined
    else:
        # Chạy offline dự phòng
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
    
    # 600 điểm nền đại diện hiển thị nhẹ mượt
    df_bg_sample = df_global.sample(n=min(600, len(df_global)), random_state=42)
    return df_global, df_bg_sample, scaler, pca

df_global, df_bg_sample, scaler, pca = load_and_fit_pca()

# --- 4. HÀM TẠO ẢNH QR CODE ---
def generate_qr_image(link_url):
    qr = qrcode.QRCode(box_size=6, border=2)
    qr.add_data(link_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# --- 5. LUẬN GIẢI TRẠNG THÁI & KHUYẾN NGHỊ ---
def get_status_feedback(f1, f2, f3, dt1, dt2, dt3):
    if f2 >= 3.5 and f1 >= 3.5:
        zone = "Vùng Quá Tải Nhịp Độ Số"
        desc = "Bạn đang phải xử lý nhiều luồng công việc số với cường độ cao, khiến năng lượng phục hồi bị suy giảm."
        tips = [
            "Tập thói quen ngắt thông báo công việc/ứng dụng sau giờ làm việc.",
            "Nghỉ ngắn 5 phút sau mỗi 45 phút tập trung vào màn hình thiết bị.",
            "Ưu tiên hoàn thành từng việc một, giảm bớt thói quen xử lý đa nhiệm."
        ]
        color = "#e11d48"
    elif f2 >= 3.5 and f1 < 3.5:
        zone = "Vùng Áp Lực Thích Ứng Công Nghệ"
        desc = "Nền tảng năng lượng còn tốt, nhưng việc thích nghi liên tục với công cụ/quy trình mới tạo ra ma sát nhận thức."
        tips = [
            "Chỉ chọn lọc 1–2 công cụ thiết thực phục vụ mục tiêu chính.",
            "Trao đổi kinh nghiệm với đồng nghiệp để rút ngắn thời gian làm quen.",
            "Cho bản thân thời gian thích ứng tự nhiên, không nóng vội."
        ]
        color = "#d97706"
    elif f1 >= 3.5 and f2 < 3.5:
        zone = "Vùng Mệt Mỏi Cần Tái Tạo"
        desc = "Áp lực chủ yếu đến từ khối lượng công việc và sinh hoạt dồn dập khiến cơ thể mệt mỏi."
        tips = [
            "Ưu tiên chất lượng giấc ngủ và thời gian thư giãn cá nhân.",
            "Lùi hạn các đầu việc không thật sự cấp bách.",
            "Dành thời gian vận động nhẹ hoặc ra ngoài hít thở không khí tự nhiên."
        ]
        color = "#ea580c"
    else:
        zone = "Vùng Cân Bằng Ổn Định"
        desc = "Bạn đang điều tiết nhịp độ rất tốt, làm chủ công cụ và duy trì năng lượng tinh thần thoải mái."
        tips = [
            "Tiếp tục duy trì nhịp độ làm việc và sinh hoạt khoa học hiện tại.",
            "Sẵn sàng chia sẻ mẹo làm việc hiệu quả với các thành viên trong nhóm.",
            "Lắng nghe cơ thể để chủ động điều chỉnh khi vào các tuần cao điểm."
        ]
        color = "#16a34a"

    max_t = max(dt1, dt2, dt3)
    if max_t == dt2 and dt2 > 6.0:
        time_insight = "⏱️ Bạn dừng lại lâu nhất ở câu **Áp lực công nghệ**, cho thấy đây là yếu tố gây trăn trở nhận thức đáng chú ý."
    elif max_t == dt1 and dt1 > 6.0:
        time_insight = "⏱️ Bạn suy xét nhiều nhất ở câu **Mức độ mệt mỏi**, phản ánh trạng thái hao mòn năng lượng đang được nội tâm quan sát kỹ."
    elif max_t == dt3 and dt3 > 6.0:
        time_insight = "⏱️ Bạn ngập ngừng nhiều nhất ở câu **Khả năng tự điều hòa**, cho thấy chiến lược thích nghi hiện tại đang có sự do dự."
    else:
        time_insight = "⏱️ Tốc độ phản hồi của bạn tương đối đồng đều và dứt khoát giữa các câu hỏi."

    return zone, desc, tips, color, time_insight

# --- 6. SIDEBAR: TẠO QR & QUẢN TRỊ VIÊN ---
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
# TRANG 1: PHIẾU KHẢO SÁT & BỘ THU THẬP THỜI GIAN
# ==============================================================================
if url_page != "result":
    st.markdown("### 🌱 Khảo Sát Nhịp Độ Làm Việc & Thích Ứng")
    st.caption("3 câu hỏi trắc nghiệm nhanh • Ẩn danh • Tự động định vị")
    
    if url_group:
        st.markdown(f'<div class="group-badge">🔒 Nhóm tham gia: <b>{url_group}</b></div>', unsafe_allow_html=True)
        assigned_group = url_group
    else:
        assigned_group = st.text_input("Tên nhóm tham gia (để trống nếu tham gia cá nhân):", value="Chung").strip()
        if not assigned_group:
            assigned_group = "Chung"

    # Câu 1
    q1_opts = {
        "1. Rất thoải mái, tràn đầy năng lượng": 1.0,
        "2. Hơi mệt mỏi nhưng hồi phục nhanh": 2.0,
        "3. Thỉnh thoảng cạn kiệt sức sau giờ làm": 3.0,
        "4. Thường xuyên mệt mỏi, giảm hứng thú": 4.0,
        "5. Kiệt sức kéo dài, rất khó phục hồi": 5.0
    }
    q1_sel = st.radio("1. Mức độ mệt mỏi / hao mòn sức lực gần đây:", list(q1_opts.keys()), index=1, key="rad_q1", on_change=on_select_q1)
    val_f1 = q1_opts[q1_sel]

    # Câu 2
    q2_opts = {
        "1. Dễ dàng làm chủ, không thấy áp lực": 1.0,
        "2. Thỉnh thoảng mất chút thời gian làm quen": 2.0,
        "3. Cảm thấy nhịp độ công nghệ khá dồn dập": 3.0,
        "4. Thường xuyên căng thẳng vì phần mềm/AI mới": 4.0,
        "5. Quá tải, cảm giác liên tục bị thúc ép": 5.0
    }
    q2_sel = st.radio("2. Áp lực phải thích nghi với phần mềm / công cụ mới:", list(q2_opts.keys()), index=2, key="rad_q2", on_change=on_select_q2)
    val_f2 = q2_opts[q2_sel]

    # Câu 3
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

        # Tính toán thời gian từng câu (giây)
        dt_q1 = round(max(0.5, t1 - t0), 2)
        dt_q2 = round(max(0.5, t2 - t1), 2)
        dt_q3 = round(max(0.5, t3 - t2), 2)
        dt_total = round(t_end - t0, 2)

        # Chuẩn hóa thời gian vào chiều F3
        time_score = 1.0 + 4.0 * min(1.0, max(0.0, (np.log(1 + dt_total) - np.log(6)) / (np.log(45) - np.log(6))))
        final_f3 = round(0.5 * (val_f3 + time_score), 2)

        # Bản ghi đầy đủ Metadata thời gian nghiên cứu khoa học
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

        # Lưu trực tiếp vào GitHub
        with st.spinner("Đang lưu dữ liệu vào hệ thống..."):
            success, _ = commit_to_github(record)

        st.session_state.latest_user = record
        st.query_params["view"] = "result"
        if assigned_group:
            st.query_params["group"] = assigned_group
            
        # Reset mốc thời gian
        st.session_state.dt_start = datetime.now()
        st.session_state.t0 = time.time()
        st.session_state.t_q1 = None
        st.session_state.t_q2 = None
        st.session_state.t_q3 = None
        st.rerun()

# ==============================================================================
# TRANG 2: ĐỊNH VỊ TÔ-PÔ & PHÂN TÍCH THỜI GIAN NHẬN THỨC
# ==============================================================================
else:
    c_btn1, _ = st.columns([1.2, 3])
    with c_btn1:
        if st.button("⬅️ Làm Lại Phiếu", use_container_width=True):
            st.query_params["view"] = "survey"
            st.session_state.dt_start = datetime.now()
            st.session_state.t0 = time.time()
            st.rerun()

    # Lấy dữ liệu mới nhất từ GitHub
    df_resp, _ = get_github_data()

    if not df_resp.empty:
        if st.session_state.latest_user is None:
            st.session_state.latest_user = df_resp.iloc[-1].to_dict()

        if st.session_state.latest_user:
            u_f1 = float(st.session_state.latest_user['Burnout_Level'])
            u_f2 = float(st.session_state.latest_user['AI_Technostress'])
            u_f3 = float(st.session_state.latest_user['Cognitive_Latency_Proxy'])
            u_grp = str(st.session_state.latest_user['group_id'])
            u_t1 = float(st.session_state.latest_user.get('t1_sec', 3.0))
            u_t2 = float(st.session_state.latest_user.get('t2_sec', 3.0))
            u_t3 = float(st.session_state.latest_user.get('t3_sec', 3.0))
            u_total_t = float(st.session_state.latest_user.get('total_latency_sec', 9.0))
            
            zone_title, zone_desc, tips, zone_col, time_insight = get_status_feedback(u_f1, u_f2, u_f3, u_t1, u_t2, u_t3)
            
            st.markdown(f"""
            <div class="result-box" style="border-left: 5px solid {zone_col};">
                <h4 style="color: {zone_col}; margin: 0 0 4px 0;">🎯 Vị trí của bạn: {zone_title}</h4>
                <p style="color: #334155; margin-bottom: 6px;">{zone_desc}</p>
                <p style="color: #475569; font-size: 0.9rem; margin-bottom: 8px;">{time_insight}</p>
                <strong>🌱 Gợi ý điều hòa:</strong>
                <ul style="margin: 4px 0 0 0; padding-left: 18px; color: #475569; font-size: 0.9rem;">
                    {''.join([f"<li>{t}</li>" for t in tips])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            u_scaled = scaler.transform([[u_f1, u_f2, u_f3]])
            u_2d = pca.transform(u_scaled)[0]
        else:
            u_2d = None
            u_grp = url_group if url_group else "Chung"

        # Lọc nhóm
        available_groups = ["Tất cả nhóm"] + sorted(list(df_resp['group_id'].dropna().astype(str).unique()))
        sel_idx = available_groups.index(u_grp) if u_grp in available_groups else 0
        
        chosen_grp = st.selectbox("Xem dữ liệu nhóm:", available_groups, index=sel_idx)
        
        if chosen_grp == "Tất cả nhóm":
            df_plot = df_resp
            grp_name = "Tất cả"
        else:
            df_plot = df_resp[df_resp['group_id'].astype(str) == chosen_grp]
            grp_name = chosen_grp

        sample_scaled = scaler.transform(df_plot[['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']].values)
        sample_2d = pca.transform(sample_scaled)

        # 1. BIỂU ĐỒ BẢN ĐỒ TÔ-PÔ (PLOTLY)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_bg_sample['PC1'], y=df_bg_sample['PC2'],
            mode='markers', marker=dict(size=5, color='#94a3b8', opacity=0.35),
            name='Chuẩn cộng đồng (N=3.459)', hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=sample_2d[:, 0], y=sample_2d[:, 1],
            mode='markers', marker=dict(size=8, color='#f43f5e', opacity=0.85),
            name=f'Nhóm {grp_name} ({len(df_plot)} mẫu)', hoverinfo='name'
        ))
        if len(sample_2d) > 0:
            c_grp = sample_2d.mean(axis=0)
            fig.add_trace(go.Scatter(
                x=[c_grp[0]], y=[c_grp[1]],
                mode='markers', marker=dict(symbol='star', size=16, color='#fbbf24', line=dict(color='black', width=1.2)),
                name=f'Tâm nhóm {grp_name}', hoverinfo='name'
            ))
        if u_2d is not None:
            fig.add_trace(go.Scatter(
                x=[u_2d[0]], y=[u_2d[1]],
                mode='markers+text', marker=dict(size=18, color='#06b6d4', line=dict(color='#083344', width=2.5)),
                text=["📍 Bạn ở đây"], textposition="top center",
                textfont=dict(color="#083344", size=12), name='Vị trí của bạn', hoverinfo='text'
            ))

        fig.update_layout(
            title=dict(text="Bản Đồ Không Gian Trạng Thái Thích Ứng", font=dict(size=12)),
            xaxis=dict(title="Trục thích ứng 1", showgrid=True, zeroline=False),
            yaxis=dict(title="Trục thích ứng 2", showgrid=True, zeroline=False),
            margin=dict(l=10, r=10, t=35, b=25),
            height=360,
            dragmode='pan',
            legend=dict(orientation="h", yanchor="bottom", y=-0.38, xanchor="center", x=0.5, font=dict(size=9)),
            template="plotly_white"
        )
        plotly_config = {
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {'format': 'png', 'filename': f'dinh_vi_thich_ung_{grp_name}', 'height': 600, 'width': 800, 'scale': 2}
        }
        st.plotly_chart(fig, use_container_width=True, config=plotly_config)

        # 2. KHỐI PHÂN TÍCH THỜI GIAN
        st.markdown("##### ⏱️ Phân Tích Thời Gian Suy Xét Từng Câu (Giây)")
        if st.session_state.latest_user and 't1_sec' in st.session_state.latest_user:
            fig_time = go.Figure()
            categories = ['Câu 1: Mệt mỏi', 'Câu 2: Áp lực AI', 'Câu 3: Điều hòa']
            user_times = [u_t1, u_t2, u_t3]
            
            avg_t1 = df_plot['t1_sec'].mean() if 't1_sec' in df_plot.columns else 3.5
            avg_t2 = df_plot['t2_sec'].mean() if 't2_sec' in df_plot.columns else 4.2
            avg_t3 = df_plot['t3_sec'].mean() if 't3_sec' in df_plot.columns else 3.8
            group_times = [avg_t1, avg_t2, avg_t3]

            fig_time.add_trace(go.Bar(
                x=categories, y=user_times,
                name='Thời gian của bạn', marker_color='#06b6d4', text=[f"{t:.1f}s" for t in user_times], textposition='auto'
            ))
            fig_time.add_trace(go.Bar(
                x=categories, y=group_times,
                name=f'Trung bình nhóm {grp_name}', marker_color='#cbd5e1', text=[f"{t:.1f}s" for t in group_times], textposition='auto'
            ))
            fig_time.update_layout(
                barmode='group', height=240, margin=dict(l=10, r=10, t=25, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5, font=dict(size=9)),
                template="plotly_white", yaxis=dict(title="Thời gian (giây)")
            )
            st.plotly_chart(fig_time, use_container_width=True, config={'displayModeBar': False})
            st.caption(f"Tổng thời gian hoàn thành khảo sát của bạn: **{u_total_t:.1f} giây** (Nhóm TB: **{df_plot['total_latency_sec'].mean() if 'total_latency_sec' in df_plot.columns else 12.0:.1f} giây**).")

        # 3. CHỈ SỐ SO SÁNH
        m_b = df_plot['Burnout_Level'].mean()
        m_t = df_plot['AI_Technostress'].mean()
        col1, col2 = st.columns(2)
        col1.metric("Mệt mỏi TB Nhóm", f"{m_b:.2f}/5.0")
        col2.metric("Áp lực công nghệ TB Nhóm", f"{m_t:.2f}/5.0")
    else:
        st.info("Chưa có dữ liệu nào trên GitHub. Vui lòng quay lại gửi phiếu khảo sát đầu tiên.")
