import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. CẤU HÌNH TRANG STREAMLIT
st.set_page_config(
    page_title="TDA Faculty Wellbeing Diagnostic",
    page_icon="🧠",
    layout="wide"
)

DATA_GLOBAL_PATH = "data/global_aligned_real_dataset.csv"
DATA_RESPONSES_PATH = "data/pilot_survey_cntt_30_responses.csv"

# Khởi tạo session state lưu thời gian bắt đầu trả lời
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()

# 2. HÀM TẢI DỮ LIỆU VÀ HUẤN LUYỆN PCA
@st.cache_resource
def load_and_fit_pca():
    df_global = pd.read_csv(DATA_GLOBAL_PATH)
    feature_cols = ['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']
    X_global = df_global[feature_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_global)
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    
    df_global['PC1'] = X_2d[:, 0]
    df_global['PC2'] = X_2d[:, 1]
    return df_global, scaler, pca

df_global, scaler, pca = load_and_fit_pca()

# 3. GIAO DIỆN HEADER
st.title("🧠 Khảo sát Nhanh: Tải Nhận thức & Áp lực Công nghệ (TDA)")
st.caption("Dự án nghiên cứu chẩn đoán trạng thái thích ứng công nghệ dựa trên Phân tích Dữ liệu Tô-pô (TDA)")

tab1, tab2 = st.tabs(["📋 Phiếu Khảo Sát Tại Chỗ", "📊 Bản Đồ Tô-pô Trạng Thái Khoa"])

# --- TAB 1: PHIẾU KHẢO SÁT ---
with tab1:
    st.subheader("Phiếu đánh giá nhanh (3 câu hỏi)")
    st.write("Vui lòng chọn mức độ phù hợp nhất với trải nghiệm hiện tại của Thầy/Cô:")
    
    with st.form("survey_form"):
        col_q1, col_q2, col_q3 = st.columns(3)
        
        with col_q1:
            q1 = st.slider(
                "1. Mức độ kiệt sức / mệt mỏi nhận thức gần đây (Burnout):",
                min_value=1.0, max_value=5.0, value=3.0, step=0.5,
                help="1: Hoàn toàn thư thái -> 5: Kiệt quệ năng lượng"
            )
            
        with col_q2:
            q2 = st.slider(
                "2. Áp lực phải liên tục thích ứng công cụ số / AI mới (Technostress):",
                min_value=1.0, max_value=5.0, value=4.0, step=0.5,
                help="1: Hoàn toàn làm chủ -> 5: Quá tải nhịp độ số"
            )
            
        with col_q3:
            q3_confidence = st.select_slider(
                "3. Khả năng phục hồi & quản lý tải nhận thức cá nhân (Coping):",
                options=[1.0, 2.0, 3.0, 4.0, 5.0],
                value=3.0,
                help="1: Rất yếu (bế tắc) -> 5: Rất chủ động và cân bằng"
            )
            
        role = st.selectbox("Vai trò chuyên môn:", ["Giảng viên cơ hữu", "Nghiên cứu viên", "Cán bộ quản lý chuyên môn"])
        submitted = st.form_submit_button("🚀 Gửi Kết Quả Khảo Sát", use_container_width=True)
        
        if submitted:
            # Đo độ trễ phản hồi (Silent Timer)
            elapsed_sec = time.time() - st.session_state.start_time
            # Ánh xạ độ trễ sang thang đo chuẩn 1.0 - 5.0
            latency_proxy = min(5.0, max(1.0, 1.0 + (elapsed_sec / 15.0) * 4.0))
            
            # Đảo nghịch thang coping (Coping thấp = Trễ/Rào cản cao)
            cognitive_deficit = (6.0 - q3_confidence + latency_proxy) / 2.0
            cognitive_deficit = min(5.0, max(1.0, cognitive_deficit))
            
            new_row = {
                'Burnout_Level': q1,
                'AI_Technostress': q2,
                'Cognitive_Latency_Proxy': cognitive_deficit,
                'group_id': role
            }
            
            # Ghi vào file local
            if os.path.exists(DATA_RESPONSES_PATH):
                df_pilot = pd.read_csv(DATA_RESPONSES_PATH)
                df_pilot = pd.concat([df_pilot, pd.DataFrame([new_row])], ignore_index=True)
            else:
                df_pilot = pd.DataFrame([new_row])
            df_pilot.to_csv(DATA_RESPONSES_PATH, index=False)
            
            st.success(f"✅ Đã ghi nhận phản hồi thành công! (Thời gian suy xét: {elapsed_sec:.1f}s)")
            st.info("Vui lòng chuyển qua Tab **'Bản Đồ Tô-pô Trạng Thái Khoa'** để xem vị trí cập nhật trực tiếp.")
            # Reset lại timer cho người tiếp theo
            st.session_state.start_time = time.time()

# --- TAB 2: BIỂU ĐỒ BẢN ĐỒ TÔ-PÔ ĐỐI SÁNH TRỰC TIẾP ---
with tab2:
    st.subheader("Không Gian Trạng Thái Đối Sánh Đa Miền (Empirical State-Space)")
    
    if os.path.exists(DATA_RESPONSES_PATH):
        df_responses = pd.read_csv(DATA_RESPONSES_PATH)
        n_samples = len(df_responses)
        
        st.markdown(f"**Tổng số phản hồi ghi nhận thực tế:** `{n_samples}` giảng viên.")
        
        # Chiếu các điểm khảo sát qua PCA
        sample_scaled = scaler.transform(df_responses[['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']].values)
        sample_2d = pca.transform(sample_scaled)
        
        # Vẽ biểu đồ Matplotlib
        fig, ax = plt.subplots(figsize=(11, 6.8), dpi=200)
        
        # 1. Vẽ 3 datasets nền
        slices = [
            ('OSMI Tech (N=1259)', slice(0, 1259), '#0084FF', '#0056b3'),
            ('Academic Stress (N=1100)', slice(1259, 2359), '#00C853', '#007E33'),
            ('Higher Ed (N=1100)', slice(2359, 3459), '#AA00FF', '#6A0080')
        ]
        for name, s, col, edge in slices:
            sub = df_global.iloc[s]
            ax.scatter(sub['PC1'], sub['PC2'], c=col, edgecolors=edge, alpha=0.35, s=25, label=name)
            
        # 2. Vẽ các điểm khảo sát thực tế (Đỏ)
        ax.scatter(sample_2d[:, 0], sample_2d[:, 1], c='#FF1744', edgecolors='black', s=90, alpha=0.95, 
                   zorder=5, label=f'Faculty Cohort Live ({n_samples} responses)')
        
        # 3. Vẽ Tâm trạng thái tập thể (Ngôi sao vàng)
        center = sample_2d.mean(axis=0)
        ax.scatter(center[0], center[1], c='#FFD600', marker='*', s=380, edgecolors='black', zorder=6,
                   label='Faculty Centroid (Department State)')
        
        # 4. Vẽ vòng bao dao động
        circle = plt.Circle((center[0], center[1]), 1.0, color='#D50000', fill=False, linestyle='--', linewidth=2.0, zorder=5)
        ax.add_patch(circle)
        
        ax.set_title("Live Empirical State-Space Positioning", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10)
        ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
        ax.grid(True, linestyle=':', alpha=0.5)
        
        st.pyplot(fig)
        
        # Chỉ số so sánh
        c1, c2, c3 = st.columns(3)
        c1.metric("Burnout TB Khoa", f"{df_responses['Burnout_Level'].mean():.2f} / 5.0", delta=f"{df_responses['Burnout_Level'].mean() - df_global['Burnout_Level'].mean():.2f} vs Global")
        c2.metric("Technostress TB Khoa", f"{df_responses['AI_Technostress'].mean():.2f} / 5.0", delta=f"{df_responses['AI_Technostress'].mean() - df_global['AI_Technostress'].mean():.2f} vs Global")
        c3.metric("Độ trễ Nhận thức TB", f"{df_responses['Cognitive_Latency_Proxy'].mean():.2f} / 5.0")
    else:
        st.warning("Chưa có phản hồi nào được ghi nhận. Vui lòng gửi khảo sát tại Tab 1.")
