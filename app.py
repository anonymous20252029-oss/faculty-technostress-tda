import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import io
import urllib.parse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import qrcode

# --- 1. CẤU HÌNH TRANG VÀ RESPONSIVE CSS ---
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
    /* Thanh công cụ zoom của Plotly luôn hiển thị rõ ràng trên mobile */
    .modebar-container {
        opacity: 0.85 !important;
    }
</style>
""", unsafe_allow_html=True)

DATA_GLOBAL_PATH = "data/global_aligned_real_dataset.csv"
DATA_RESPONSES_PATH = "data/pilot_survey_cntt_30_responses.csv"

# --- 2. NHẬN DIỆN THAM SỐ URL (CHỐNG MẤT DỮ LIỆU KHI MOBILE RELOAD) ---
url_group = st.query_params.get("group", None)
if url_group:
    url_group = urllib.parse.unquote(url_group).strip()

url_page = st.query_params.get("view", "survey")

if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'latest_user' not in st.session_state:
    st.session_state.latest_user = None

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
    
    # Lấy 600 điểm đại diện hiển thị nền nhẹ mượt trên di động
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
def get_status_feedback(f1, f2, f3):
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
    return zone, desc, tips, color

# --- 6. SIDEBAR: TẠO LINK & QR NHÓM ---
with st.sidebar:
    st.markdown("### 🔗 Tạo Link & Mã QR Nhóm")
    new_group_name = st.text_input("Nhập tên nhóm muốn tạo:", placeholder="Ví dụ: KTPM, KHMT, Nhom_1")
    if new_group_name:
        encoded_grp = urllib.parse.quote(new_group_name.strip())
        generated_link = f"?group={encoded_grp}"
        st.success(f"Link nhóm: **{new_group_name}**")
        st.code(generated_link, language="text")
        
        qr_bytes = generate_qr_image(generated_link)
        st.image(qr_bytes, caption=f"Mã QR: {new_group_name}", use_container_width=True)
        st.download_button("📥 Tải QR về máy", qr_bytes, file_name=f"QR_{new_group_name}.png", mime="image/png")

# ==============================================================================
# TRANG 1: PHIẾU KHẢO SÁT
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

    with st.form("quick_survey_form"):
        # Câu 1
        q1_opts = {
            "1. Rất thoải mái, tràn đầy năng lượng": 1.0,
            "2. Hơi mệt mỏi nhưng hồi phục nhanh": 2.0,
            "3. Thỉnh thoảng cạn kiệt sức sau giờ làm": 3.0,
            "4. Thường xuyên mệt mỏi, giảm hứng thú": 4.0,
            "5. Kiệt sức kéo dài, rất khó phục hồi": 5.0
        }
        q1_sel = st.radio("1. Mức độ mệt mỏi / hao mòn sức lực gần đây:", list(q1_opts.keys()), index=1)
        val_f1 = q1_opts[q1_sel]

        # Câu 2
        q2_opts = {
            "1. Dễ dàng làm chủ, không thấy áp lực": 1.0,
            "2. Thỉnh thoảng mất chút thời gian làm quen": 2.0,
            "3. Cảm thấy nhịp độ công nghệ khá dồn dập": 3.0,
            "4. Thường xuyên căng thẳng vì phần mềm/AI mới": 4.0,
            "5. Quá tải, cảm giác liên tục bị thúc ép": 5.0
        }
        q2_sel = st.radio("2. Áp lực phải thích nghi với phần mềm / công cụ mới:", list(q2_opts.keys()), index=2)
        val_f2 = q2_opts[q2_sel]

        # Câu 3
        q3_opts = {
            "1. Rất chủ động, luôn có cách cân bằng tốt": 1.0,
            "2. Thích ứng ổn định, ít khi bế tắc": 2.0,
            "3. Đôi khi bối rối, cần nhiều thời gian suy nghĩ": 3.0,
            "4. Khó cân bằng, hay đắn đo và trì hoãn việc": 4.0,
            "5. Rất khó khăn trong việc tự điều hòa áp lực": 5.0
        }
        q3_sel = st.radio("3. Khả năng tự điều hòa khi gặp công việc dồn dập:", list(q3_opts.keys()), index=1)
        val_f3 = q3_opts[q3_sel]

        st.write("")
        btn_submit = st.form_submit_button("🚀 Gửi & Xem Định Vị Của Bạn", use_container_width=True, type="primary")

        if btn_submit:
            elapsed = time.time() - st.session_state.start_time
            latency_bonus = min(1.0, elapsed / 25.0)
            final_f3 = min(5.0, val_f3 + latency_bonus)

            record = {
                'Burnout_Level': val_f1,
                'AI_Technostress': val_f2,
                'Cognitive_Latency_Proxy': final_f3,
                'group_id': assigned_group,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }

            if os.path.exists(DATA_RESPONSES_PATH):
                df_curr = pd.read_csv(DATA_RESPONSES_PATH)
                df_curr = pd.concat([df_curr, pd.DataFrame([record])], ignore_index=True)
            else:
                df_curr = pd.DataFrame([record])
            df_curr.to_csv(DATA_RESPONSES_PATH, index=False)

            st.session_state.latest_user = record
            
            # Cập nhật query params điều hướng
            st.query_params["view"] = "result"
            if assigned_group:
                st.query_params["group"] = assigned_group
            st.session_state.start_time = time.time()
            st.rerun()

# ==============================================================================
# TRANG 2: KẾT QUẢ ĐỊNH VỊ (TƯƠNG TÁC ZOOM & LƯU ẢNH CHUẨN)
# ==============================================================================
else:
    c_btn1, _ = st.columns([1.2, 3])
    with c_btn1:
        if st.button("⬅️ Làm Lại Phiếu", use_container_width=True):
            st.query_params["view"] = "survey"
            st.session_state.start_time = time.time()
            st.rerun()

    if os.path.exists(DATA_RESPONSES_PATH):
        df_resp = pd.read_csv(DATA_RESPONSES_PATH)
        
        # Dự phòng reload
        if st.session_state.latest_user is None and len(df_resp) > 0:
            st.session_state.latest_user = df_resp.iloc[-1].to_dict()

        if st.session_state.latest_user:
            u_f1 = float(st.session_state.latest_user['Burnout_Level'])
            u_f2 = float(st.session_state.latest_user['AI_Technostress'])
            u_f3 = float(st.session_state.latest_user['Cognitive_Latency_Proxy'])
            u_grp = str(st.session_state.latest_user['group_id'])
            
            zone_title, zone_desc, tips, zone_col = get_status_feedback(u_f1, u_f2, u_f3)
            
            st.markdown(f"""
            <div class="result-box" style="border-left: 5px solid {zone_col};">
                <h4 style="color: {zone_col}; margin: 0 0 4px 0;">🎯 Vị trí của bạn: {zone_title}</h4>
                <p style="color: #334155; margin-bottom: 6px;">{zone_desc}</p>
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

        # Lựa chọn nhóm đối sánh
        available_groups = ["Tất cả nhóm"] + sorted(list(df_resp['group_id'].dropna().astype(str).unique()))
        sel_idx = available_groups.index(u_grp) if u_grp in available_groups else 0
        
        chosen_grp = st.selectbox("Xem dữ liệu nhóm:", available_groups, index=sel_idx)
        
        if chosen_grp == "Tất cả nhóm":
            df_plot = df_resp
            grp_name = "Tất cả"
        else:
            df_plot = df_resp[df_resp['group_id'].astype(str) == chosen_grp]
            grp_name = chosen_grp

        # Tọa độ nhóm
        sample_scaled = scaler.transform(df_plot[['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']].values)
        sample_2d = pca.transform(sample_scaled)

        # --- VẼ BẰNG PLOTLY HỖ TRỢ ZOOM CẢM ỨNG & DOWNLOAD ---
        fig = go.Figure()

        # 1. Điểm nền toàn cầu
        fig.add_trace(go.Scatter(
            x=df_bg_sample['PC1'], y=df_bg_sample['PC2'],
            mode='markers',
            marker=dict(size=5, color='#94a3b8', opacity=0.35),
            name='Chuẩn cộng đồng (N=3.459)',
            hoverinfo='skip'
        ))

        # 2. Thành viên trong nhóm
        fig.add_trace(go.Scatter(
            x=sample_2d[:, 0], y=sample_2d[:, 1],
            mode='markers',
            marker=dict(size=8, color='#f43f5e', opacity=0.85),
            name=f'Nhóm {grp_name} ({len(df_plot)} mẫu)',
            hoverinfo='name'
        ))

        # 3. Tâm nhóm (Ngôi sao vàng)
        if len(sample_2d) > 0:
            c_grp = sample_2d.mean(axis=0)
            fig.add_trace(go.Scatter(
                x=[c_grp[0]], y=[c_grp[1]],
                mode='markers',
                marker=dict(symbol='star', size=16, color='#fbbf24', line=dict(color='black', width=1.2)),
                name=f'Tâm nhóm {grp_name}',
                hoverinfo='name'
            ))

        # 4. Vị trí của bạn (Xanh Neon nổi bật)
        if u_2d is not None:
            fig.add_trace(go.Scatter(
                x=[u_2d[0]], y=[u_2d[1]],
                mode='markers+text',
                marker=dict(size=18, color='#06b6d4', line=dict(color='#083344', width=2.5)),
                text=["📍 Bạn ở đây"],
                textposition="top center",
                textfont=dict(color="#083344", size=12),
                name='Vị trí của bạn',
                hoverinfo='text'
            ))

        fig.update_layout(
            title=dict(text="Bản Đồ Không Gian Trạng Thái (Chụm ngón tay để Phóng to / Thu nhỏ)", font=dict(size=12)),
            xaxis=dict(title="Trục thích ứng 1", showgrid=True, zeroline=False),
            yaxis=dict(title="Trục thích ứng 2", showgrid=True, zeroline=False),
            margin=dict(l=10, r=10, t=35, b=25),
            height=370,
            dragmode='pan',  # Chế độ mặc định trên điện thoại là chạm ngón tay để trượt xem
            legend=dict(orientation="h", yanchor="bottom", y=-0.38, xanchor="center", x=0.5, font=dict(size=9)),
            template="plotly_white"
        )

        # CẤU HÌNH TƯƠNG TÁC ĐẦY ĐỦ: CHO PHÉP ZOOM BẰNG TAY, CUỘN CHUỘT, LƯU ẢNH
        plotly_config = {
            'scrollZoom': True,        # Bật zoom bằng con lăn chuột hoặc 2 ngón tay trên điện thoại
            'displayModeBar': True,    # Luôn hiển thị thanh công cụ
            'displaylogo': False,      # Ẩn logo plotly cho gọn
            'modeBarButtonsToAdd': ['zoom2d', 'pan2d', 'resetScale2d', 'toImage'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'dinh_vi_thich_ung_{grp_name}',
                'height': 600,
                'width': 800,
                'scale': 2              # Tải về ảnh độ nét cao x2
            }
        }

        st.plotly_chart(fig, use_container_width=True, config=plotly_config)

        # Hướng dẫn thao tác nhanh cho người dùng di động
        st.caption("💡 *Mẹo:* Dùng 2 ngón tay chụm/mở để phóng to/thu nhỏ, trượt ngón tay để di chuyển bản đồ.")

        # CHỈ SỐ SO SÁNH GỌN GÀNG
        m_b = df_plot['Burnout_Level'].mean()
        m_t = df_plot['AI_Technostress'].mean()
        col1, col2 = st.columns(2)
        col1.metric("Mệt mỏi TB", f"{m_b:.2f}/5.0")
        col2.metric("Áp lực công nghệ TB", f"{m_t:.2f}/5.0")
    else:
        st.info("Chưa có dữ liệu nào. Vui lòng quay lại điền phiếu.")
