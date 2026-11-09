import streamlit as st
import os
import glob
import zipfile
import parselmouth
import pandas as pd
import plotly.express as px
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. 기본 설정 및 데이터 전처리 (캐싱으로 로딩 속도 최적화) ---
st.set_page_config(page_title="ㅐ/ㅔ 발음 분류 AI", layout="wide")

@st.cache_data
def prepare_data():
    extract_dir = "temp_dataset"
    os.makedirs(extract_dir, exist_ok=True)
    
    # [수정됨] gtt_dataset.zip 파일만 압축 해제 및 학습 데이터로 사용
    zip_files = ["gtt_dataset.zip"]
    for zf in zip_files:
        if os.path.exists(zf):
            with zipfile.ZipFile(zf, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
    wav_files = glob.glob(f"{extract_dir}/**/*.wav", recursive=True)
    data_list = []
    
    for wav in wav_files:
        filename = os.path.basename(wav)
        
        # 라벨링 로직 (gtt_dataset 명명 규칙에 맞춤)
        if "ae.wav" in filename: label = "ae (ㅐ)"
        elif "e.wav" in filename: label = "e (ㅔ)"
        else: continue
            
        try:
            sound = parselmouth.Sound(wav)
            formant = sound.to_formant_burg(max_number_of_formants=5.0, maximum_formant=5500.0)
            mid_point = sound.get_total_duration() / 2
            f1 = formant.get_value_at_time(1, mid_point)
            f2 = formant.get_value_at_time(2, mid_point)
            
            if pd.notna(f1) and pd.notna(f2):
                data_list.append({"라벨": label, "F1": f1, "F2": f2})
        except:
            pass
            
    df = pd.DataFrame(data_list)
    
    if not df.empty:
        # 결측치 및 이상치 처리 적용
        df = df.dropna(subset=['F1', 'F2'])
        df = df[(df['F1'] > 300) & (df['F1'] < 1200)] 
        df = df[(df['F2'] > 1200) & (df['F2'] < 3000)]
        
    return df

@st.cache_resource
def train_model(df):
    X = df[['F1', 'F2']]
    y = df['라벨']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 80% 학습, 20% 테스트 데이터로 분할
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 모델 학습
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # 분할된 20% 테스트 데이터로 정확도 계산
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    return model, scaler, accuracy

# 데이터 로드 및 모델 학습
df = prepare_data()
if not df.empty:
    model, scaler, accuracy = train_model(df)

# --- 2. 웹앱 화면 구성 ---
st.title("🎙️ 한국어 'ㅐ' / 'ㅔ' 발음 분류 인공지능")

# 이론적 배경 탭 구성
tab1, tab2 = st.tabs(["📊 분류기 테스트", "📖 과학적 원리와 사회적 의의"])

with tab2:
    st.header("음향음성학적 원리: 포먼트(Formant)와 혀의 위치")
    st.markdown("""
    * **'ㅐ' (전설 평순 중저모음):** 입을 더 크게 벌려 혀의 높이가 낮아지므로, 첫 번째 주파수(F1)가 상대적으로 높게 측정됩니다.
    * **'ㅔ' (전설 평순 중고모음):** 입을 덜 벌려 혀의 위치가 높아지므로, F1 주파수가 낮게 측정됩니다.
    * 이 모델은 자음의 영향을 받아 주파수가 미세하게 변하는 '조음 동화' 현상 속에서도, 기계학습(SVM)을 통해 두 모음의 주파수 경계를 객관적으로 찾아냅니다.
    """)
    
    st.header("사회적 의의: 발음의 병합(Merger) 현상")
    st.markdown("""
    현대 한국어, 특히 청년층 화자들 사이에서는 'ㅐ'와 'ㅔ'의 입 벌림 크기 차이가 줄어들어 사실상 동일한 소리로 발음되는 **음운 병합(Phonemic merger)** 현상이 진행 중입니다. 이로 인해 '결제/결재', '베개/배개' 등의 맞춤법 오류가 빈번하게 발생합니다.
    본 웹앱은 사라져가는 두 모음의 차이를 정량적으로 시각화하며, 아나운서 지망생의 발음 교정이나 외국인을 위한 한국어 교육 도구로 활용될 수 있습니다.
    """)

with tab1:
    if df.empty:
        st.error("데이터를 불러오지 못했습니다. GitHub 저장소에 'gtt_dataset.zip' 파일이 있는지 확인해 주세요.")
    else:
        st.write(f"✅ 총 **{len(df)}개**의 통제된 발음 데이터(TTS) 학습 완료")
        st.write(f"🎯 테스트 데이터(20%) 기준 모델 분류 정확도: **{accuracy:.1f}%**")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("1. 음성 입력")
            
            # 다중 음성 입력을 위한 세션 상태 초기화
            if 'audio_history' not in st.session_state:
                st.session_state.audio_history = []
                
            audio_value = st.audio_input("직접 '개' 또는 '게'를 녹음해 보세요!")
            uploaded_file = st.file_uploader("또는 wav 파일을 업로드하세요.", type=["wav"])
            
            new_audio = audio_value if audio_value else uploaded_file
            
            # 새로운 입력이 들어오면 리스트에 추가
            if new_audio and new_audio not in st.session_state.audio_history:
                st.session_state.audio_history.append(new_audio)
                
            # 삭제 기능 UI 구성
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("⏪ 최근 녹음 삭제") and st.session_state.audio_history:
                    st.session_state.audio_history.pop()
                    st.rerun()
            with col_btn2:
                if st.button("🗑️ 전체 녹음 삭제") and st.session_state.audio_history:
                    st.session_state.audio_history.clear()
                    st.rerun()
            
            # 산점도에 그릴 마커 데이터를 모아둘 리스트
            analyzed_points = []
            
            if st.session_state.audio_history:
                st.subheader("2. 분석 결과")
                
                # 누적된 음성 기록을 모두 분석하여 출력
                for idx, audio_data in enumerate(st.session_state.audio_history):
                    st.markdown(f"**🎤 {idx + 1}번째 녹음**")
                    st.audio(audio_data)
                    
                    with open(f"temp_input_{idx}.wav", "wb") as f:
                        f.write(audio_data.read())
                        audio_data.seek(0) # 다음 읽기를 위해 파일 포인터 초기화
                    
                    try:
                        sound_test = parselmouth.Sound(f"temp_input_{idx}.wav")
                        formant_test = sound_test.to_formant_burg(max_number_of_formants=5.0, maximum_formant=5500.0)
                        mid_point = sound_test.get_total_duration() / 2
                        test_f1 = formant_test.get_value_at_time(1, mid_point)
                        test_f2 = formant_test.get_value_at_time(2, mid_point)
                        
                        if pd.notna(test_f1):
                            X_test = scaler.transform([[test_f1, test_f2]])
                            prediction = model.predict(X_test)[0]
                            prob = model.predict_proba(X_test)[0]
                            max_prob = max(prob) * 100
                            
                            st.success(f"결과: **{prediction}** (확률: {max_prob:.1f}%)")
                            st.caption(f"📍 F1(혀 높이): {test_f1:.1f}Hz, F2(혀 앞뒤): {test_f2:.1f}Hz")
                            
                            # 산점도 마커용 데이터 저장
                            analyzed_points.append({
                                "idx": idx + 1, 
                                "F1": test_f1, 
                                "F2": test_f2, 
                                "prediction": prediction
                            })
                        else:
                            st.error("모음 구간을 찾을 수 없습니다.")
                    except Exception as e:
                        st.error("음성 분석 중 오류가 발생했습니다.")
                    st.divider()
                    
        with col2:
            st.subheader("3. 포먼트 공간 분포도")
            fig = px.scatter(df, x="F2", y="F1", color="라벨", 
                             title="학습 데이터 분포 (gTTS 통제 데이터) 및 내 발음 위치",
                             labels={"F2": "F2 (혀의 전후 위치 - 낮을수록 뒤)", "F1": "F1 (혀의 높낮이 - 높을수록 낮음)"},
                             color_discrete_map={"ae (ㅐ)": "blue", "e (ㅔ)": "green"})
            fig.update_yaxes(autorange="reversed") # 음성학 관례에 따라 F1 역순 배치
            fig.update_xaxes(autorange="reversed") # 음성학 관례에 따라 F2 역순 배치
            
            # 누적된 사용자 발음 위치 마커 추가
            for pt in analyzed_points:
                marker_color = 'red' if 'ae' in pt['prediction'] else 'orange'
                fig.add_scatter(x=[pt['F2']], y=[pt['F1']], mode='markers+text',
                                text=[f"#{pt['idx']}"], textposition="top center",
                                marker=dict(size=15, color=marker_color, symbol='star', line=dict(width=1, color='black')),
                                name=f"내 발음 #{pt['idx']}")
                
            st.plotly_chart(fig, use_container_width=True)
