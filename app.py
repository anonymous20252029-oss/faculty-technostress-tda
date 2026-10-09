import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. CẤU HÌNH TRANG VÀ RESPONSIVE CSS ---
st.set_page_config(
    page_title="Khảo Sát & Định Vị Thích Ứng Số",
    page_icon="🌱",
    layout="centered"  # Giữ giao diện tập trung ở giữa màn hình, không bị loãng trên màn hình lớn
)

# Nhúng CSS tùy biến để loại bỏ khoảng trắng thừa, tối ưu cho cả mobile & desktop
st.markdown("""
<style>
    /* Giảm padding mặc định của Streamlit */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 1.0rem !important;
        padding-right: 1.0rem !important;
        max-width: 860px !important;
    }
    /* Ẩn bớt footer và menu mặc định */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Canh chỉnh chữ gọn gàng */
    h3, h4, h5 {
        margin-top: 0.2rem !important;
        margin-bottom: 0.4rem !important;
    }
    p, label {
        font-size: 0.95rem !important;
        margin-bottom: 0.2rem !important;
    }
    .stRadio > div {
        gap: 0.35rem !important;
    }
    /* Khung kết quả bo tròn gọn nhẹ */
    .result-box {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

DATA_GLOBAL_PATH = "data/global_aligned_real_dataset.csv"
DATA_RESPONSES_PATH = "data/pilot_survey_cntt_30_responses.csv"

# Quản lý trạng thái
if 'page' not in st.session_state:
    st.session_state.page = "survey"
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'latest_user' not in st.session_state:
    st.session_state.latest_user = None

# --- 2. TẢI DỮ LIỆU NỀN TOÀN CẦU (N=3.459) & PCA ---
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
    return df_global, scaler, pca

df_global, scaler, pca = load_and_fit_pca()

# --- 3. ĐÁNH GIÁ TRẠNG THÁI & KHUYẾN NGHỊ THIẾT THỰC ---
def get_status_feedback(f1, f2, f3):
    if f2 >= 3.5 and f1 >= 3.5:
        zone = "Vùng Quá Tải Nhịp Độ Số"
        desc = "Bạn đang phải xử lý nhiều luồng công việc số với cường độ cao, khiến năng lượng phục hồi bị suy giảm rõ rệt."
        tips = [
            "Tập thói quen ngắt thông báo công việc/ứng dụng sau giờ làm việc.",
            "Nghỉ ngắn 5 phút sau mỗi 45 phút tập trung vào màn hình thiết bị.",
            "Ưu tiên hoàn thành từng việc một, giảm bớt thói quen xử lý đa nhiệm (multitasking)."
        ]
        color = "#e11d48"
    elif f2 >= 3.5 and f1 < 3.5:
        zone = "Vùng Áp Lực Thích Ứng Công Nghệ"
        desc = "Nền tảng năng lượng của bạn còn tốt, nhưng việc làm quen liên tục với các công cụ/quy trình số mới đang tạo ra áp lực thời điểm."
        tips = [
            "Chỉ chọn lọc dùng 1–2 công cụ thật sự cần thiết phục vụ mục tiêu chính.",
            "Tham khảo kinh nghiệm hoặc cách làm tắt từ đồng nghiệp để đỡ mất thời gian tự mò mẫm.",
            "Cho bản thân thời gian thích ứng tự nhiên, không nóng vội."
        ]
        color = "#d97706"
    elif f1 >= 3.5 and f2 < 3.5:
        zone = "Vùng Mệt Mỏi Cần Tái Tạo"
        desc = "Áp lực không đến nhiều từ công nghệ mà chủ yếu do khối lượng công việc và sinh hoạt dồn dập khiến cơ thể mệt mỏi."
        tips = [
            "Ưu tiên chất lượng giấc ngủ và thời gian thư giãn cá nhân.",
            "Giảm bớt hoặc lùi hạn các đầu việc không cấp bách.",
            "Dành thời gian vận động nhẹ hoặc ra ngoài hít thở không khí tự nhiên."
        ]
        color = "#ea580c"
    else:
        zone = "Vùng Cân Bằng Ổn Định"
        desc = "Bạn đang điều tiết nhịp độ rất tốt, làm chủ công cụ và duy trì năng lượng tinh thần thoải mái."
        tips = [
            "Tiếp tục duy trì nhịp độ làm việc và sinh hoạt khoa học hiện tại.",
            "Sẵn sàng chia sẻ mẹo làm việc hiệu quả với các thành viên khác trong nhóm.",
            "Lắng nghe cơ thể để chủ động điều chỉnh khi bước vào các tuần cao điểm."
        ]
        color = "#16a34a"
    return zone, desc, tips, color

# ==============================================================================
# TRANG 1: PHIẾU KHẢO SÁT TINH GỌN (CHUNG CHO MỌI NGƯỜI)
# ==============================================================================
if st.session_state.page == "survey":
    st.markdown("### 🌱 Khảo Sát Nhịp Độ Làm Việc & Thích Ứng Số")
    st.caption("3 câu hỏi trắc nghiệm nhanh • Ẩn danh • Tự động định vị vị trí")
    
    with st.form("quick_survey_form"):
        # Gom nhóm chung chung (mặc định 'Chung', người dùng chỉ đổi nếu có nhóm riêng)
        group_input = st.text_input(
            "Tên nhóm hoặc Đơn vị tham gia (để trống nếu tham gia cá nhân):", 
            value="Chung",
            help="Dùng để gom nhóm các thành viên cùng nhóm/phòng ban với nhau"
        ).strip()
        if not group_input:
            group_input = "Chung"

        # Câu 1
        q1_opts = {
            "1. Rất thoải mái, tràn đầy năng lượng": 1.0,
            "2. Hơi mệt mỏi nhưng hồi phục nhanh": 2.0,
            "3. Thỉnh thoảng cạn kiệt sức sau giờ làm": 3.0,
            "4. Thường xuyên mệt mỏi, giảm hứng thú": 4.0,
            "5. Kiệt sức kéo dài, rất khó phục hồi": 5.0
        }
        q1_sel = st.radio(
            "1. Mức độ mệt mỏi / hao mòn sức lực gần đây:",
            options=list(q1_opts.keys()),
            index=1
        )
        val_f1 = q1_opts[q1_sel]

        # Câu 2
        q2_opts = {
            "1. Dễ dàng làm chủ, không thấy áp lực": 1.0,
            "2. Thỉnh thoảng mất chút thời gian làm quen": 2.0,
            "3. Cảm thấy nhịp độ thay đổi công nghệ khá dồn dập": 3.0,
            "4. Thường xuyên căng thẳng vì phải chạy theo phần mềm/AI mới": 4.0,
            "5. Quá tải, cảm giác liên tục bị công nghệ thúc ép": 5.0
        }
        q2_sel = st.radio(
            "2. Áp lực phải liên tục thích nghi với công cụ số / phần mềm mới:",
            options=list(q2_opts.keys()),
            index=2
        )
        val_f2 = q2_opts[q2_sel]

        # Câu 3
        q3_opts = {
            "1. Rất chủ động, luôn có cách cân bằng tốt": 1.0,
            "2. Thích ứng ổn định, ít khi bế tắc": 2.0,
            "3. Đôi khi bối rối, cần nhiều thời gian suy nghĩ": 3.0,
            "4. Khó cân bằng, hay đắn đo và trì hoãn giải quyết việc": 4.0,
            "5. Rất khó khăn trong việc tự điều hòa áp lực": 5.0
        }
        q3_sel = st.radio(
            "3. Khả năng tự điều hòa và giải tỏa khi gặp công việc dồn dập:",
            options=list(q3_opts.keys()),
            index=1
        )
        val_f3 = q3_opts[q3_sel]

        st.write("")
        btn_submit = st.form_submit_button("🚀 Gửi & Xem Định Vị Của Bạn", use_container_width=True, type="primary")

        if btn_submit:
            # Ghi nhận độ trễ phản hồi thực tế
            elapsed = time.time() - st.session_state.start_time
            latency_bonus = min(1.0, elapsed / 25.0)
            final_f3 = min(5.0, val_f3 + latency_bonus)

            record = {
                'Burnout_Level': val_f1,
                'AI_Technostress': val_f2,
                'Cognitive_Latency_Proxy': final_f3,
                'group_id': group_input,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # Ghi dữ liệu
            if os.path.exists(DATA_RESPONSES_PATH):
                df_curr = pd.read_csv(DATA_RESPONSES_PATH)
                df_curr = pd.concat([df_curr, pd.DataFrame([record])], ignore_index=True)
            else:
                df_curr = pd.DataFrame([record])
            df_curr.to_csv(DATA_RESPONSES_PATH, index=False)

            st.session_state.latest_user = record
            # TỰ ĐỘNG CHUYỂN TRANG
            st.session_state.page = "result"
            st.session_state.start_time = time.time()
            st.rerun()

# ==============================================================================
# TRANG 2: KẾT QUẢ ĐỊNH VỊ TRỰC QUAN & LỜI KHUYÊN
# ==============================================================================
elif st.session_state.page == "result":
    # Nút quay lại gọn nhẹ ở góc trên
    c_btn1, c_btn2 = st.columns([1, 3])
    with c_btn1:
        if st.button("⬅️ Làm Lại", use_container_width=True):
            st.session_state.page = "survey"
            st.session_state.start_time = time.time()
            st.rerun()

    if os.path.exists(DATA_RESPONSES_PATH):
        df_resp = pd.read_csv(DATA_RESPONSES_PATH)
        
        # 1. HỘP GIẢI THÍCH & LỜI KHUYÊN CHO CÁ NHÂN
        if st.session_state.latest_user:
            u_f1 = st.session_state.latest_user['Burnout_Level']
            u_f2 = st.session_state.latest_user['AI_Technostress']
            u_f3 = st.session_state.latest_user['Cognitive_Latency_Proxy']
            u_grp = st.session_state.latest_user['group_id']
            
            zone_title, zone_desc, tips, zone_col = get_status_feedback(u_f1, u_f2, u_f3)
            
            st.markdown(f"""
            <div class="result-box" style="border-left: 5px solid {zone_col};">
                <h4 style="color: {zone_col}; margin: 0 0 6px 0;">🎯 Trạng thái của bạn: {zone_title}</h4>
                <p style="color: #334155; margin-bottom: 8px;">{zone_desc}</p>
                <strong>🌱 Gợi ý điều hòa phù hợp:</strong>
                <ul style="margin: 4px 0 0 0; padding-left: 18px; color: #475569; font-size: 0.9rem;">
                    {''.join([f"<li>{t}</li>" for t in tips])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            u_scaled = scaler.transform([[u_f1, u_f2, u_f3]])
            u_2d = pca.transform(u_scaled)[0]
        else:
            u_2d = None
            u_grp = "Chung"

        # 2. XỬ LÝ DỮ LIỆU NHÓM
        available_groups = ["Tất cả nhóm"] + sorted(list(df_resp['group_id'].dropna().unique()))
        sel_idx = available_groups.index(u_grp) if u_grp in available_groups else 0
        
        col_g_sel, col_cnt = st.columns([2.5, 1.5])
        with col_g_sel:
            chosen_grp = st.selectbox("Xem dữ liệu nhóm:", available_groups, index=sel_idx)
        
        if chosen_grp == "Tất cả nhóm":
            df_plot = df_resp
            grp_name = "Tất cả thành viên"
        else:
            df_plot = df_resp[df_resp['group_id'] == chosen_grp]
            grp_name = chosen_grp
            
        with col_cnt:
            st.metric("Số mẫu nhóm", f"{len(df_plot)} người")

        # 3. VẼ BẢN ĐỒ TÔ-PÔ THU GỌN (VỪA VẶN MÀN HÌNH ĐIỆN THOẠI & LAPTOP)
        sample_scaled = scaler.transform(df_plot[['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']].values)
        sample_2d = pca.transform(sample_scaled)

        plt.rcParams['font.family'] = 'sans-serif'
        fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=180)  # Kích thước chuẩn, không chiếm quá nhiều diện tích dọc

        # Dữ liệu nền đa miền
        slices = [
            ('Chuẩn Doanh nghiệp Công nghệ (N=1259)', slice(0, 1259), '#38bdf8'),
            ('Chuẩn Học thuật & Đào tạo (N=1100)', slice(1259, 2359), '#4ade80'),
            ('Chuẩn Giáo dục Bậc cao (N=1100)', slice(2359, 3459), '#c084fc')
        ]
        for s_title, s_idx, s_c in slices:
            sub = df_global.iloc[s_idx]
            ax.scatter(sub['PC1'], sub['PC2'], c=s_c, alpha=0.18, s=16, label=s_title)

        # Các thành viên trong nhóm
        ax.scatter(sample_2d[:, 0], sample_2d[:, 1], c='#f43f5e', edgecolors='black', linewidth=0.6,
                   s=55, alpha=0.9, zorder=5, label=f'Thành viên {grp_name}')

        # Tâm trung bình của nhóm
        if len(sample_2d) > 0:
            c_grp = sample_2d.mean(axis=0)
            ax.scatter(c_grp[0], c_grp[1], c='#fbbf24', marker='*', s=260, edgecolors='black', linewidth=1.2,
                       zorder=6, label=f'Tâm nhóm: {grp_name}')
            circle = plt.Circle((c_grp[0], c_grp[1]), 0.95, color='#f43f5e', fill=False, linestyle='--', linewidth=1.5, zorder=5)
            ax.add_patch(circle)

        # Vị trí cá nhân (Xanh Neon nổi bật)
        if u_2d is not None:
            ax.scatter(u_2d[0], u_2d[1], c='#06b6d4', marker='o', s=160, edgecolors='#083344', linewidth=2.0,
                       zorder=7, label='📍 Vị trí của bạn')
            ax.annotate("Bạn ở đây", xy=(u_2d[0], u_2d[1]), xytext=(u_2d[0] + 0.35, u_2d[1] + 0.35),
                        arrowprops=dict(facecolor='#06b6d4', edgecolor='black', arrowstyle="->", lw=1.2),
                        fontsize=8.5, fontweight='bold', color="#083344",
                        bbox=dict(boxstyle="round,pad=0.15", fc="#cffafe", ec="#06b6d4", lw=0.8), zorder=8)

        ax.set_title("Bản Đồ Không Gian Trạng Thái Thích Ứng", fontsize=10.5, fontweight='bold', pad=8)
        ax.set_xlabel(f"Trục thích ứng 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=8.5)
        ax.set_ylabel(f"Trục thích ứng 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=7.2)
        ax.grid(True, linestyle=':', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)

        # 4. CHỈ SỐ SO SÁNH GỌN GÀNG (3 CỘT)
        m_b = df_plot['Burnout_Level'].mean()
        m_t = df_plot['AI_Technostress'].mean()
        g_b = df_global['Burnout_Level'].mean()
        g_t = df_global['AI_Technostress'].mean()

        col1, col2 = st.columns(2)
        col1.metric("Mức mệt mỏi trung bình", f"{m_b:.2f} / 5.0", delta=f"{m_b - g_b:+.2f} so với chuẩn chung")
        col2.metric("Áp lực công nghệ trung bình", f"{m_t:.2f} / 5.0", delta=f"{m_t - g_t:+.2f} so với chuẩn chung")
    else:
        st.info("Chưa có dữ liệu nào. Vui lòng quay lại điền phiếu.")
