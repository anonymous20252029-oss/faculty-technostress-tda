import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. CẤU HÌNH GIAO DIỆN HỆ THỐNG ---
st.set_page_config(
    page_title="Lắng Nghe Nhịp Độ Nhận Thức & Thích Ứng Công Nghệ",
    page_icon="🌱",
    layout="wide"
)

DATA_GLOBAL_PATH = "data/global_aligned_real_dataset.csv"
DATA_RESPONSES_PATH = "data/pilot_survey_cntt_30_responses.csv"

# Quản lý trạng thái chuyển trang và lưu kết quả cá nhân
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "survey"
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'latest_submission' not in st.session_state:
    st.session_state.latest_submission = None

# --- 2. TẢI DỮ LIỆU CHUẨN TOÀN CẦU VÀ HUẤN LUYỆN PCA ---
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
    X_global = df_global[feature_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_global)
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    
    df_global['PC1'] = X_2d[:, 0]
    df_global['PC2'] = X_2d[:, 1]
    return df_global, scaler, pca

df_global, scaler, pca = load_and_fit_pca()

# --- 3. HÀM LUẬN GIẢI TÂM LÝ & KHUYẾN NGHỊ THÍCH ỨNG ---
def get_psychological_insight(f1, f2, f3):
    """
    f1: Mức độ hao mòn năng lượng (Burnout)
    f2: Áp lực nhịp độ công nghệ (Technostress)
    f3: Tải nhận thức & rào cản phục hồi (Cognitive Latency & Deficit)
    """
    if f2 >= 3.5 and f1 >= 3.5:
        zone = "Vùng Quá Tải Nhịp Độ Số & Hao Mòn Năng Lượng (High Technostress & Burnout Trap)"
        meaning = (
            "Thầy/Cô đang phải liên tục vận hành nhận thức ở cường độ cao trước sự dồn dập của các công cụ mới "
            "song song với áp lực công việc thường nhật. Điều này làm cạn kiệt tài nguyên phục hồi tự nhiên, "
            "dễ tạo nên cảm giác 'bị công nghệ thúc ép' và mệt mỏi tinh thần kéo dài."
        )
        recommendations = [
            "**Thiết lập ranh giới số cá nhân (Digital Boundaries):** Dành khung giờ cố định ngắt kết nối với thông báo công việc/AI sau giờ làm việc.",
            "**Kỹ thuật Vi nghỉ ngơi (Micro-breaks):** Áp dụng quy tắc 50-10 (sau 50 phút làm việc với màn hình, dành 10 phút thả lỏng mắt, hít thở sâu hoặc vận động nhẹ).",
            "**Giảm tải kỳ vọng tức thời:** Cho phép bản thân có lộ trình thích nghi vừa sức với công nghệ mới, tránh tâm lý FOMO (sợ tụt hậu)."
        ]
        color = "#DC2626"
    elif f2 >= 3.5 and f1 < 3.5:
        zone = "Vùng Căng Thẳng Thích Nghi Kỹ Thuật Số (Techno-Adaptation Friction)"
        meaning = (
            "Nguồn năng lượng nền tảng của Thầy/Cô vẫn khá vững vàng, tuy nhiên nhịp độ cập nhật và vận hành công cụ số "
            "đang tạo ra những 'ma sát nhận thức' đáng kể. Thầy/Cô có xu hướng dành nhiều nỗ lực tập trung để làm chủ hệ thống mới, "
            "dẫn đến cảm giác căng thẳng thời điểm."
        )
        recommendations = [
            "**Tối ưu hóa quy trình tiếp cận:** Chọn lọc 1-2 công cụ AI/số thiết thực nhất phục vụ trực tiếp bài giảng/nghiên cứu, tạm gác các công nghệ ngoại vi.",
            "**Chia sẻ gánh nặng chuyên môn:** Trao đổi cùng đồng nghiệp trong bộ môn về các mẫu bài giảng/tài nguyên có sẵn để tránh phải tự tìm tòi lại từ đầu.",
            "**Bài tập thư giãn nhận thức:** Thực hành 3-5 phút thiền buông thư hoặc hít thở điều hòa trước mỗi giờ chuyển tiếp công việc."
        ]
        color = "#D97706"
    elif f1 >= 3.5 and f2 < 3.5:
        zone = "Vùng Mỏi Nhận Thức Chuyên Môn & Cần Phục Hồi (Emotional & Professional Fatigue)"
        meaning = (
            "Áp lực từ công nghệ không phải là nguyên nhân chính, mà trạng thái mỏi mệt chủ yếu đến từ khối lượng công việc, "
            "trách nhiệm giảng dạy, nghiên cứu tích tụ lâu ngày làm hao tổn năng lượng thần kinh."
        )
        recommendations = [
            "**Tái tạo tài nguyên tâm lý:** Dành sự ưu tiên hàng đầu cho chất lượng giấc ngủ và thời gian thư giãn cá nhân.",
            "**Sắp xếp thứ tự ưu tiên:** Phân loại nhiệm vụ theo ma trận khẩn cấp/quan trọng; chủ động ủy thác hoặc giãn tiến độ cho các đầu việc không cấp bách.",
            "**Tìm kiếm sự đồng cảm:** Chia sẻ trạng thái hiện tại với người thân hoặc đồng nghiệp thân thiết để giải tỏa tải cảm xúc."
        ]
        color = "#EA580C"
    else:
        zone = "Vùng Thích Ứng Cân Bằng & Năng Lượng Ổn Định (Resilient Equilibrium)"
        meaning = (
            "Thầy/Cô đang duy trì được sự điều hòa rất tốt giữa nhịp độ công nghệ và nội lực tinh thần. "
            "Khả năng thích ứng linh hoạt giúp Thầy/Cô làm chủ công cụ mà không bị cuốn vào vòng xoáy áp lực số."
        )
        recommendations = [
            "**Duy trì nhịp sinh học hiện tại:** Tiếp tục giữ vững các thói quen quản lý thời gian và chăm sóc bản thân đang phát huy hiệu quả.",
            "**Chia sẻ kinh nghiệm:** Lan tỏa cách thức cân bằng và sử dụng công nghệ hiệu quả đến các đồng nghiệp trong đơn vị.",
            "**Tiếp tục lắng nghe bản thân:** Định kỳ tự quan sát cảm xúc để chủ động điều chỉnh khi bước vào các giai đoạn cao điểm thi cử/nghiên cứu."
        ]
        color = "#16A34A"
        
    return zone, meaning, recommendations, color

# --- 4. GIAO DIỆN HEADER ---
st.title("🌱 Lắng Nghe Nhịp Độ Nhận Thức & Thích Ứng Công Nghệ")
st.markdown(
    "Chào mừng Quý Thầy/Cô. Không gian này được thiết kế để cùng Thầy/Cô lắng nghe nhịp độ tâm lý, "
    "định vị mức độ dung nạp công nghệ và chia sẻ những gợi ý điều hòa năng lượng khoa học, an lành."
)
st.write("")

# --- NÚT ĐIỀU HƯỚNG TRANG (TỰ ĐỘNG HOẶC CHỦ ĐỘNG) ---
nav_col1, nav_col2, _ = st.columns([1.5, 2.2, 3])
with nav_col1:
    if st.button("📝 Điền Phiếu Khảo Sát", use_container_width=True, 
                 type="primary" if st.session_state.current_tab == "survey" else "secondary"):
        st.session_state.current_tab = "survey"
        st.rerun()
with nav_col2:
    if st.button("📊 Xem Kết Quả Định Vị & Lời Khuyên", use_container_width=True, 
                 type="primary" if st.session_state.current_tab == "result" else "secondary"):
        st.session_state.current_tab = "result"
        st.rerun()

st.divider()

# ==============================================================================
# TRANG 1: PHIẾU KHẢO SÁT VỚI NGÔN NGỮ THÂN THIỆN
# ==============================================================================
if st.session_state.current_tab == "survey":
    st.subheader("🌿 Phiếu Chia Sẻ Cảm Nhận Trạng Thái Công Việc")
    st.markdown(
        "Khảo sát hoàn toàn ẩn danh, gồm **3 câu hỏi ngắn gọn**. Thầy/Cô chỉ cần lựa chọn phương án "
        "phù hợp nhất với trải nghiệm thực tế gần đây của mình."
    )
    
    with st.form("survey_form_vietnamese"):
        st.markdown("##### 🏢 Thông tin Đơn vị & Vai trò Chuyên môn")
        c_g1, c_g2 = st.columns(2)
        with c_g1:
            group_options = [
                "Bộ môn Khoa học Máy tính",
                "Bộ môn Kỹ thuật Phần mềm",
                "Bộ môn Hệ thống Thông tin",
                "Bộ môn Mạng & An ninh mạng",
                "Tổ Văn phòng / Đảm bảo Chất lượng",
                "Nhóm Nghiên cứu Trọng điểm",
                "Khác (Tự điền tên)"
            ]
            sel_group = st.selectbox("Đơn vị / Bộ môn Thầy/Cô đang công tác:", group_options)
            if sel_group == "Khác (Tự điền tên)":
                custom_g = st.text_input("Vui lòng ghi tên đơn vị:", value="Bộ môn Khác")
                final_group = custom_g.strip()
            else:
                final_group = sel_group
                
        with c_g2:
            role_options = [
                "Giảng viên cơ hữu",
                "Giảng viên kiêm nhiệm / Nghiên cứu viên",
                "Cán bộ Quản lý chuyên môn",
                "Nghiên cứu sinh / Trợ giảng"
            ]
            sel_role = st.selectbox("Vai trò công tác:", role_options)

        st.write("")
        st.markdown("##### 💬 Cảm nhận của Thầy/Cô trong thời gian gần đây:")
        
        # Câu 1
        burnout_dict = {
            "1. Hoàn toàn thư thái, tràn đầy năng lượng và hào hứng với bài giảng / đề tài": 1.0,
            "2. Đôi lúc thấm mệt nhưng dễ dàng hồi phục sau giấc ngủ hoặc ngày nghỉ": 2.0,
            "3. Thường xuyên thấy hao hụt năng lượng sau giờ làm việc, cần nhiều nỗ lực để bắt đầu": 3.0,
            "4. Thường xuyên kiệt sức tinh thần, giảm sút niềm vui và sự kiên nhẫn chuyên môn": 4.0,
            "5. Kiệt quệ trầm trọng, cảm giác quá tải kéo dài và rất khó tái tạo năng lượng": 5.0
        }
        q1_ans = st.radio(
            "1. Thầy/Cô cảm nhận thế nào về mức độ hồi phục và năng lượng tinh thần của mình? (Burnout Severity)",
            options=list(burnout_dict.keys()),
            index=2
        )
        f1_val = burnout_dict[q1_ans]
        
        # Câu 2
        techno_dict = {
            "1. Rất thoải mái và làm chủ tốt các công cụ số / nền tảng AI mới": 1.0,
            "2. Thỉnh thoảng cần chút thời gian làm quen nhưng nhịp độ tiếp thu rất dễ chịu": 2.0,
            "3. Cảm thấy nhịp độ chuyển đổi số và công cụ AI diễn ra khá dồn dập, đôi lúc thấy áp lực": 3.0,
            "4. Thường xuyên lo âu và căng thẳng vì phải liên tục chạy theo các chuẩn công nghệ mới": 4.0,
            "5. Rất áp lực và quá tải, có cảm giác bị công nghệ và yêu cầu số hóa 'đuổi theo' liên tục": 5.0
        }
        q2_ans = st.radio(
            "2. Nhịp độ thích nghi với các công cụ công nghệ / AI mới đang tác động đến Thầy/Cô như thế nào? (Technostress)",
            options=list(techno_dict.keys()),
            index=2
        )
        f2_val = techno_dict[q2_ans]
        
        # Câu 3
        coping_dict = {
            "1. Thầy/Cô luôn có chiến lược tự cân bằng tốt, chủ động điều tiết và giải tỏa áp lực": 1.0,
            "2. Khả năng thích ứng khá ổn định, ít khi bị rơi vào trạng thái bế tắc hay đắn đo kéo dài": 2.0,
            "3. Thỉnh thoảng có chút bối rối, cần nhiều thời gian suy xét và trì hoãn giải quyết công việc": 3.0,
            "4. Thường gặp khó khăn trong việc cân bằng cảm xúc, hay trăn trở và lo âu khi xử lý việc": 4.0,
            "5. Cảm thấy bất an, rất khó buông bỏ lo âu và gặp trở ngại lớn trong việc tự hồi phục": 5.0
        }
        q3_ans = st.radio(
            "3. Khi đối mặt với nhiều áp lực dồn dập, khả năng tự điều hòa và vượt qua của Thầy/Cô ra sao? (Coping & Latency)",
            options=list(coping_dict.keys()),
            index=2
        )
        f3_val = coping_dict[q3_ans]

        st.write("")
        submit_btn = st.form_submit_button("🌱 Gửi Chia Sẻ & Xem Kết Quả Định Vị Cá Nhân", use_container_width=True, type="primary")
        
        if submit_btn:
            # Bấm giờ ngầm đo thời gian suy xét (Silent Latency Timer)
            elapsed_sec = time.time() - st.session_state.start_time
            latency_bonus = min(1.0, (elapsed_sec / 20.0))
            final_f3 = min(5.0, f3_val + latency_bonus)
            
            # Ghi nhận kết quả
            submission_record = {
                'Burnout_Level': f1_val,
                'AI_Technostress': f2_val,
                'Cognitive_Latency_Proxy': final_f3,
                'group_id': final_group,
                'role': sel_role,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Lưu vào file CSV
            if os.path.exists(DATA_RESPONSES_PATH):
                df_pilot = pd.read_csv(DATA_RESPONSES_PATH)
                df_pilot = pd.concat([df_pilot, pd.DataFrame([submission_record])], ignore_index=True)
            else:
                df_pilot = pd.DataFrame([submission_record])
            df_pilot.to_csv(DATA_RESPONSES_PATH, index=False)
            
            # Lưu lại trạng thái cá nhân vừa nộp để hiển thị
            st.session_state.latest_submission = submission_record
            
            # TỰ ĐỘNG CHUYỂN TRANG QUA TRANG KẾT QUẢ
            st.session_state.current_tab = "result"
            st.session_state.start_time = time.time()
            st.rerun()

# ==============================================================================
# TRANG 2: KẾT QUẢ ĐỊNH VỊ ĐÔI (CÁ NHÂN & NHÓM) KÈM LUẬN GIẢI TÂM LÝ
# ==============================================================================
elif st.session_state.current_tab == "result":
    st.subheader("📍 Không Gian Định Vị Trạng Thái Nhận Thức & Thấu Cảm")
    
    if os.path.exists(DATA_RESPONSES_PATH):
        df_responses = pd.read_csv(DATA_RESPONSES_PATH)
        n_total = len(df_responses)
        
        # 1. PHẦN LUẬN GIẢI DÀNH RIÊNG CHO CÁ NHÂN (NẾU CÓ DỮ LIỆU VỪA NỘP)
        if st.session_state.latest_submission is not None:
            user_data = st.session_state.latest_submission
            u_f1 = user_data['Burnout_Level']
            u_f2 = user_data['AI_Technostress']
            u_f3 = user_data['Cognitive_Latency_Proxy']
            u_group = user_data['group_id']
            
            # Phân tích tâm lý
            zone_name, meaning_text, recomms, zone_color = get_psychological_insight(u_f1, u_f2, u_f3)
            
            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; border-left: 6px solid {zone_color}; padding: 18px; border-radius: 8px; margin-bottom: 20px;">
                    <h4 style="color: {zone_color}; margin-top: 0;">🎯 Định Vị Cá Nhân Của Thầy/Cô: {zone_name}</h4>
                    <p style="font-size: 15px; color: #1E293B; line-height: 1.6;"><strong>Ý nghĩa trạng thái:</strong> {meaning_text}</p>
                    <p style="font-size: 14.5px; color: #334155; margin-bottom: 6px;"><strong>🌱 Gợi ý điều hòa năng lượng dành riêng cho Thầy/Cô:</strong></p>
                    <ul style="font-size: 14px; color: #334155; line-height: 1.6; margin-bottom: 0;">
                        {''.join([f"<li>{r}</li>" for r in recomms])}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Chiếu tọa độ của riêng cá nhân
            user_scaled = scaler.transform([[u_f1, u_f2, u_f3]])
            user_2d = pca.transform(user_scaled)[0]
        else:
            user_2d = None
            u_group = "Toàn thể Khoa"
            
        # 2. BỘ LỌC ĐỐI SÁNH THEO NHÓM
        st.markdown("##### 👥 Không Gian Đối Sánh Tương Quan (Cá Nhân - Nhóm - Chuẩn Toàn Cầu)")
        col_sel, col_stat = st.columns([2, 1])
        
        groups_list = ["Toàn thể Khoa (Tất cả phản hồi)"] + sorted(list(df_responses['group_id'].dropna().unique()))
        
        # Mặc định chọn nhóm của người vừa nộp nếu có
        default_index = 0
        if st.session_state.latest_submission is not None and u_group in groups_list:
            default_index = groups_list.index(u_group)
            
        with col_sel:
            selected_group = st.selectbox("Chọn Nhóm / Bộ môn để xem định vị tương quan:", groups_list, index=default_index)
            
        if selected_group == "Toàn thể Khoa (Tất cả phản hồi)":
            df_curr_group = df_responses
            display_group_name = "Toàn thể Khoa"
        else:
            df_curr_group = df_responses[df_responses['group_id'] == selected_group]
            display_group_name = selected_group
            
        n_group = len(df_curr_group)
        with col_stat:
            st.metric(f"Số lượng thành viên ({display_group_name})", f"{n_group} Thầy/Cô")
            
        # Chiếu tọa độ các thành viên trong nhóm
        group_scaled = scaler.transform(df_curr_group[['Burnout_Level', 'AI_Technostress', 'Cognitive_Latency_Proxy']].values)
        group_2d = pca.transform(group_scaled)
        
        # 3. VẼ BIỂU ĐỒ TRỰC QUAN HỌC THUẬT (2-LEVEL POSITIONING)
        plt.rcParams['font.family'] = 'serif'
        fig, ax = plt.subplots(figsize=(12, 7.5), dpi=250)
        
        # Lớp nền: 3 bộ dữ liệu chuẩn toàn cầu (N=3.459)
        slices = [
            ('Chuẩn Doanh nghiệp CNTT Toàn cầu (OSMI Tech, N=1,259)', slice(0, 1259), '#0084FF', '#0056b3'),
            ('Chuẩn Áp lực Học thuật Đại học (Academic Stress, N=1,100)', slice(1259, 2359), '#00C853', '#007E33'),
            ('Chuẩn Giáo dục Bậc cao Quốc tế (Higher Ed Wellbeing, N=1,100)', slice(2359, 3459), '#AA00FF', '#6A0080')
        ]
        for s_name, s_idx, col, edge in slices:
            sub = df_global.iloc[s_idx]
            ax.scatter(sub['PC1'], sub['PC2'], c=col, edgecolors=edge, alpha=0.25, s=24, label=s_name)
            
        # Lớp 2: Các thành viên trong nhóm (Màu đỏ cam)
        ax.scatter(
            group_2d[:, 0], group_2d[:, 1],
            c='#FF1744', edgecolors='black', linewidth=0.8,
            s=85, alpha=0.90, zorder=5,
            label=f'Thành viên {display_group_name} (n={n_group})'
        )
        
        # Lớp 3: Tâm trạng thái của nhóm (Ngôi sao vàng)
        if n_group > 0:
            center_group = group_2d.mean(axis=0)
            ax.scatter(
                center_group[0], center_group[1],
                c='#FFD600', marker='*', s=450, edgecolors='black', linewidth=1.5,
                zorder=6, label=f'Tâm trạng thái Nhóm: {display_group_name}'
            )
            # Vòng tròn bao phương sai nhóm
            circle = plt.Circle(
                (center_group[0], center_group[1]), 0.95,
                color='#D50000', fill=False, linestyle='--', linewidth=2.0,
                zorder=5, label='Vùng dao động tập trung của Nhóm'
            )
            ax.add_patch(circle)
            
        # Lớp 4: ĐỊNH VỊ CÁ NHÂN (Vòng tròn Xanh Neon rực rỡ)
        if user_2d is not None:
            ax.scatter(
                user_2d[0], user_2d[1],
                c='#00E5FF', marker='o', s=260, edgecolors='#004D40', linewidth=2.2,
                zorder=7, label='📍 VỊ TRÍ CỦA THẦY/CÔ (Cá Nhân Bản Thân)'
            )
            # Chú thích mũi tên trỏ vào cá nhân
            ax.annotate(
                "Bạn ở đây!", xy=(user_2d[0], user_2d[1]), xytext=(user_2d[0] + 0.45, user_2d[1] + 0.45),
                arrowprops=dict(facecolor='#00E5FF', edgecolor='black', arrowstyle="->", lw=1.5),
                fontsize=10.5, fontweight='bold', color="#004D40",
                bbox=dict(boxstyle="round,pad=0.2", fc="#E0F7FA", ec="#00ACC1", lw=1.0),
                zorder=8
            )
            
        ax.set_title("Bản Đồ Không Gian Trạng Thái Tô-pô Đa Miền: Cá Nhân vs. Nhóm vs. Chuẩn Toàn Cầu", 
                     fontsize=12.5, fontweight='bold', pad=14)
        ax.set_xlabel(f"Trục Tọa độ Nhận thức 1 (PC1 - {pca.explained_variance_ratio_[0]*100:.1f}% Phương sai)", fontsize=10.5)
        ax.set_ylabel(f"Trục Tọa độ Nhận thức 2 (PC2 - {pca.explained_variance_ratio_[1]*100:.1f}% Phương sai)", fontsize=10.5)
        ax.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor='white', fontsize=8.8)
        ax.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # 4. CHỈ SỐ SO SÁNH THỰC TẾ
        st.write("")
        st.markdown("##### 📊 Chỉ Số So Sánh Trung Bình Giữa Các Mức Độ")
        c1, c2, c3 = st.columns(3)
        
        m_b = df_curr_group['Burnout_Level'].mean()
        m_t = df_curr_group['AI_Technostress'].mean()
        m_c = df_curr_group['Cognitive_Latency_Proxy'].mean()
        
        g_b = df_global['Burnout_Level'].mean()
        g_t = df_global['AI_Technostress'].mean()
        g_c = df_global['Cognitive_Latency_Proxy'].mean()
        
        c1.metric("Mức Hao Mòn Năng Lượng (Burnout)", f"{m_b:.2f} / 5.0", delta=f"{m_b - g_b:+.2f} so với Toàn cầu")
        c2.metric("Áp Lực Thích Ứng Công Nghệ (Technostress)", f"{m_t:.2f} / 5.0", delta=f"{m_t - g_t:+.2f} so với Toàn cầu")
        c3.metric("Độ Trễ Nhận Thức (Cognitive Latency)", f"{m_c:.2f} / 5.0", delta=f"{m_c - g_c:+.2f} so với Toàn cầu")
        
        # 5. KHUYẾN NGHỊ DÀNH CHO CẤP ĐỘ NHÓM / BỘ MÔN (POLICY RECOMMENDATIONS)
        with st.expander(f"🏛️ Khuyến nghị điều hòa nhịp độ công việc cho cấp Đơn vị / {display_group_name}"):
            st.markdown(
                f"""
                - **Chỉ số Technostress trung bình của nhóm ({m_t:.2f}/5.0):** 
                  Nếu cao hơn chuẩn toàn cầu, đơn vị nên xem xét tổ chức các buổi chia sẻ 'Best Practice' nội bộ 
                  để các giảng viên có thế mạnh công nghệ hỗ trợ kèm cặp đồng nghiệp, giảm áp lực tự mày mò đơn độc.
                - **Chỉ số Burnout trung bình ({m_b:.2f}/5.0):** 
                  Nếu ở mức trên 3.5, đơn vị cần linh hoạt trong phân bổ hạn ngạch nhiệm vụ, ưu tiên giảm bớt các thủ tục hành chính số hóa rườm rà.
                - **Bảo toàn tính đa dạng nhận thức:** 
                  Vòng dao động của nhóm phản ánh sự gắn kết nhưng cũng có sự phân hóa giữa các độ tuổi và thâm niên; 
                  cần lắng nghe và tôn trọng nhịp độ thích nghi riêng của từng cá nhân.
                """
            )
    else:
        st.warning("🌱 Hiện tại chưa có dữ liệu phản hồi nào. Mời Thầy/Cô bấm nút 'Điền Phiếu Khảo Sát' ở phía trên để bắt đầu trải nghiệm.")
