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

# --- 1. CẤU HÌNH GIAO DIỆN SIÊU GỌN ---
st.set_page_config(
    page_title="Khảo Sát & Định Vị Thích Ứng Số",
    page_icon="🌱",
    layout="wide"  # Dùng layout wide để chia 3 cột dàn ngang vừa vặn, không bị cuộn dọc
)

st.markdown("""
<style>
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 1.0rem !important;
        padding-left: 1.0rem !important;
        padding-right: 1.0rem !important;
        max-width: 1100px !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    h3, h4, h5 {
        margin-top: 0.1rem !important;
        margin-bottom: 0.2rem !important;
    }
    p, label {
        font-size: 0.9rem !important;
    }
    .stRadio > div {
        gap: 0.25rem !important;
    }
    .compact-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 10px 12px;
        height: 100%;
        border: 1px solid #e2e8f0;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .group-badge {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

DATA_GLOBAL_PATH = "data/global_aligned_real_dataset.csv"
BASE_URL = "https://faculty-technostress-tda-mmmrqgaetftbvoqpdigqmm.streamlit.app"

url_group = st.query_params.get("group", None)
if url_group:
    url_group = urllib.parse.unquote(url_group).strip()

url_page = st.query_params.get("view", "survey")

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

# --- 3. LOAD NỀN TOÀN CẦU (N=3.459) & PCA ---
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
    
    df_bg_sample = df_global.sample(n=min(500, len(df_global)), random_state=42)
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

# --- 4. SIDEBAR: TẠO QR NHÓM ---
with st.sidebar:
    st.markdown("### 🔗 Tạo Link & Mã QR Nhóm")
    new_group_name = st.text_input("Tên nhóm muốn tạo:", placeholder="Ví dụ: CNTT_TDTU, Pilot_30")
    if new_group_name:
        encoded_grp = urllib.parse.quote(new_group_name.strip())
        full_group_url = f"{BASE_URL}/?group={encoded_grp}"
        st.success(f"Link: {new_group_name}")
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

        dt_q1 = round(max(0.5, t1 - t0), 1)
        dt_q2 = round(max(0.5, t2 - t1), 1)
        dt_q3 = round(max(0.5, t3 - t2), 1)
        dt_total = round(t_end - t0, 1)

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

        with st.spinner("Đang lưu kết quả..."):
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
# TRANG 2: KẾT QUẢ ĐỊNH VỊ (BẢN ĐỒ + 3 CỘT TRONG 1 HÀNG DÀN NGANG)
# ==============================================================================
else:
    c_head1, c_head2 = st.columns([1.5, 4.5])
    with c_head1:
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
        u_t1 = float(st.session_state.latest_user.get('t1_sec', 2.5))
        u_t2 = float(st.session_state.latest_user.get('t2_sec', 3.0))
        u_t3 = float(st.session_state.latest_user.get('t3_sec', 2.0))
        u_total = float(st.session_state.latest_user.get('total_latency_sec', 7.5))

        # Chọn nhóm
        available_groups = ["Tất cả nhóm"] + sorted(list(df_resp['group_id'].dropna().astype(str).unique()))
        sel_idx = available_groups.index(u_grp) if u_grp in available_groups else 0
        
        with c_head2:
            chosen_grp = st.selectbox("Xem dữ liệu nhóm:", available_groups, index=sel_idx)

        if chosen_grp == "Tất cả nhóm":
            df_plot = df_resp
            grp_name = "Tất cả thành viên"
        else:
            df_plot = df_resp[df_resp['group_id'].astype(str) == chosen_grp]
            grp_name = chosen_grp

        # --- BẢN ĐỒ TÔ-PÔ THU GỌN ---
        sample_scaled = scaler.transform(df_plot[['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']].values)
        sample_2d = pca.transform(sample_scaled)

        u_scaled = scaler.transform([[u_f1, u_f2, u_f3]])
        u_2d = pca.transform(u_scaled)[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_bg_sample['PC1'], y=df_bg_sample['PC2'],
            mode='markers', marker=dict(size=4.5, color='#cbd5e1', opacity=0.4),
            name='Chuẩn cộng đồng (N=3.459)', hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=sample_2d[:, 0], y=sample_2d[:, 1],
            mode='markers', marker=dict(size=7, color='#f43f5e', opacity=0.8),
            name=f'Thành viên {grp_name}', hoverinfo='name'
        ))
        if len(sample_2d) > 0:
            c_grp = sample_2d.mean(axis=0)
            fig.add_trace(go.Scatter(
                x=[c_grp[0]], y=[c_grp[1]],
                mode='markers', marker=dict(symbol='star', size=15, color='#fbbf24', line=dict(color='black', width=1.2)),
                name=f'Tâm nhóm {grp_name}', hoverinfo='name'
            ))
        fig.add_trace(go.Scatter(
            x=[u_2d[0]], y=[u_2d[1]],
            mode='markers+text', marker=dict(size=16, color='#06b6d4', line=dict(color='#083344', width=2.2)),
            text=["📍 Bạn ở đây"], textposition="top center",
            textfont=dict(color="#083344", size=11), name='Vị trí của bạn', hoverinfo='text'
        ))

        fig.update_layout(
            margin=dict(l=5, r=5, t=10, b=20),
            height=280,  # Chiều cao thấp, siêu gọn
            dragmode='pan',
            legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="center", x=0.5, font=dict(size=8.5)),
            template="plotly_white",
            xaxis=dict(showgrid=True, zeroline=False),
            yaxis=dict(showgrid=True, zeroline=False)
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': True, 'doubleClick': 'reset'})
        st.caption("🔍 *Mẹo:* Chụm ngón tay hoặc cuộn chuột để phóng to • Chạm đúp để thu nhỏ lại.")

        # --- TỔNG HỢP NỘI DUNG (HOÀN TOÀN KHÔNG DÙNG DẤU SAO **) ---
        if u_f2 >= 3.5 and u_f1 >= 3.5:
            p_title = "Quá Tải Nhịp Độ Số & Mệt Mỏi"
            p_desc = "Cường độ làm việc cao song song với nhịp độ công nghệ mới làm giảm tốc độ hồi phục năng lượng."
            p_color = "#e11d48"
            tips_html = "<li>Đặt ranh giới số ngoài giờ làm.</li><li>Nghỉ 5 phút sau 45 phút nhìn màn hình.</li><li>Ưu tiên xử lý từng việc dứt điểm.</li>"
        elif u_f2 >= 3.5 and u_f1 < 3.5:
            p_title = "Áp Lực Thích Ứng Công Nghệ"
            p_desc = "Nền tảng sức khỏe tốt nhưng việc làm quen công cụ số mới đang tạo ra ma sát nhận thức thời điểm."
            p_color = "#d97706"
            tips_html = "<li>Chỉ chọn 1-2 công cụ cần thiết nhất.</li><li>Hỏi mẹo nhanh từ người quen thao tác.</li><li>Cho bản thân thời gian thích ứng tự nhiên.</li>"
        elif u_f1 >= 3.5 and u_f2 < 3.5:
            p_title = "Mệt Mỏi Cần Tái Tạo"
            p_desc = "Khối lượng công việc nói chung dồn dập khiến cơ thể mệt mỏi, không bắt nguồn từ công nghệ."
            p_color = "#ea580c"
            tips_html = "<li>Ưu tiên chất lượng giấc ngủ.</li><li>Lùi hạn các đầu việc không cấp bách.</li><li>Vận động nhẹ hoặc đi dạo ngoài trời.</li>"
        else:
            p_title = "Cân Bằng Thích Ứng Ổn Định"
            p_desc = "Bạn đang điều hòa rất tốt giữa năng lượng làm việc và nhịp độ thích nghi công cụ mới."
            p_color = "#16a34a"
            tips_html = "<li>Duy trì nhịp sinh hoạt khoa học hiện tại.</li><li>Sẵn sàng chia sẻ kinh nghiệm cho nhóm.</li><li>Nhận biết sớm khi vào tuần cao điểm.</li>"

        # Đánh giá độ trễ thời gian từng câu
        max_t = max(u_t1, u_t2, u_t3)
        if max_t == u_t2 and u_t2 > 5.0:
            time_remark = "Bạn suy xét lâu nhất ở câu Áp lực công nghệ, cho thấy đây là mối bận tâm nhận thức lớn nhất."
        elif max_t == u_t1 and u_t1 > 5.0:
            time_remark = "Bạn dừng lại lâu nhất ở câu Mức độ mệt mỏi, phản ánh trạng thái hao mòn năng lượng đang được lưu tâm."
        elif max_t == u_t3 and u_t3 > 5.0:
            time_remark = "Bạn đắn đo nhiều nhất ở câu Điều hòa, cho thấy chiến lược thích nghi hiện tại đang có sự do dự."
        else:
            time_remark = "Thời gian phản hồi các câu khá nhanh và dứt khoát."

        avg_f1 = df_plot['Burnout_Level'].mean()
        avg_f2 = df_plot['AI_Technostress'].mean()
        grp_summary = f"Mức mệt mỏi TB: <b>{avg_f1:.2f}/5.0</b> | Áp lực công nghệ: <b>{avg_f2:.2f}/5.0</b>"

        # --- HIỂN THỊ 3 CỘT TRONG CÙNG 1 HÀNG DÀN NGANG (KHÔNG CUỘN DỌC) ---
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            st.markdown(f"""
            <div class="compact-card" style="border-left: 4px solid {p_color};">
                <b style="color: {p_color}; font-size: 0.95rem;">👤 1. Định Vị Cá Nhân</b><br>
                <b>Trạng thái:</b> {p_title}<br>
                <span style="color: #475569;">{p_desc}</span><br><br>
                <b>Dữ liệu lựa chọn & thời gian:</b><br>
                • Câu 1 (Mệt mỏi): <b>{u_f1:.1f} điểm</b> ({u_t1}s)<br>
                • Câu 2 (Áp lực AI): <b>{u_f2:.1f} điểm</b> ({u_t2}s)<br>
                • Câu 3 (Điều hòa): <b>{u_f3:.1f} điểm</b> ({u_t3}s)<br>
                • Tổng thời gian: <b>{u_total}s</b><br>
                <span style="color: #0369a1; font-size: 0.82rem;">{time_remark}</span>
            </div>
            """, unsafe_allow_html=True)

        with col_c2:
            st.markdown(f"""
            <div class="compact-card" style="border-left: 4px solid #6366f1;">
                <b style="color: #4f46e5; font-size: 0.95rem;">👥 2. Đại Diện Nhóm</b><br>
                <b>Nhóm:</b> {grp_name} ({len(df_plot)} mẫu)<br>
                {grp_summary}<br><br>
                <b>Xu hướng tập thể:</b><br>
                Tâm trạng thái nhóm tập trung ở mức <b>{'Quá tải' if avg_f2 >= 3.5 else 'Cân bằng'}</b> so với mặt bằng chung toàn cầu.
            </div>
            """, unsafe_allow_html=True)

        with col_c3:
            st.markdown(f"""
            <div class="compact-card" style="border-left: 4px solid #10b981;">
                <b style="color: #059669; font-size: 0.95rem;">💡 3. Gợi Ý Điều Hòa</b><br>
                <b>Cho bạn:</b>
                <ul style="margin: 2px 0 6px 0; padding-left: 16px; color: #334155;">
                    {tips_html}
                </ul>
                <b>Cho quản lý nhóm:</b><br>
                <span style="color: #334155;">Linh hoạt tiến độ, tổ chức chia sẻ kinh nghiệm công cụ để giảm bớt tự mày mò.</span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Chưa có dữ liệu nào trên GitHub. Vui lòng quay lại gửi phiếu khảo sát đầu tiên.")
