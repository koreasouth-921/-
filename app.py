import streamlit as st
import os
import glob
import zipfile
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# --- 1. 기본 설정 및 전체 음성 파형(MFCC) 데이터 전처리 ---
st.set_page_config(page_title="ㅐ/ㅔ 전체 음성 학습 AI", layout="wide")

@st.cache_data
def prepare_data():
    extract_dir = "temp_dataset"
    os.makedirs(extract_dir, exist_ok=True)
    
    zip_files = ["gtt_dataset.zip", "ae_data.zip", "e_data.zip"]
    for zf in zip_files:
        if os.path.exists(zf):
            with zipfile.ZipFile(zf, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
    wav_files = glob.glob(f"{extract_dir}/**/*.wav", recursive=True)
    data_list = []
    
    for wav in wav_files:
        filename = os.path.basename(wav)
        parent_dir = os.path.basename(os.path.dirname(wav))
        
        if "ae.wav" in filename: label = "ae (ㅐ)"
        elif "e.wav" in filename: label = "e (ㅔ)"
        elif "ㅐ" in parent_dir or "ae" in parent_dir: label = "ae (ㅐ)"
        elif "ㅔ" in parent_dir or "e" in parent_dir: label = "e (ㅔ)"
        else: continue
            
        try:
            # 1. 오디오 파일 전체 로드
            y, sr = librosa.load(wav, sr=22050)
            
            # 2. 파일 전체 길이에 대한 음향 특성(MFCC) 13개 추출
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # 3. 전체 시간 축에 대한 평균값을 구하여 파일 1개의 전체 특징 벡터로 압축
            mfcc_mean = np.mean(mfcc.T, axis=0)
            
            # 추출된 13개의 변수를 데이터프레임에 담기
            feature_dict = {"라벨": label}
            for i in range(13):
                feature_dict[f"MFCC_{i+1}"] = mfcc_mean[i]
            data_list.append(feature_dict)
        except:
            pass
            
    return pd.DataFrame(data_list)

@st.cache_resource
def train_model(df):
    # 라벨을 제외한 13차원의 MFCC 전체 특성을 X 데이터로 사용
    X = df.drop(columns=['라벨', 'PCA1', 'PCA2'], errors='ignore')
    y = df['라벨']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 80/20 분할 및 모델 학습
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
    
    # 13차원의 데이터를 화면에 시각화하기 위해 PCA(주성분 분석)로 2차원 압축
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df_pca = df.copy()
    df_pca['PCA1'] = pca_result[:, 0]
    df_pca['PCA2'] = pca_result[:, 1]
    
    return model, scaler, pca, df_pca, accuracy

# 데이터 로드 및 학습
df = prepare_data()
if not df.empty:
    model, scaler, pca, df_pca, accuracy = train_model(df)

# --- 2. 웹앱 화면 구성 ---
st.title("🎙️ 한국어 'ㅐ' / 'ㅔ' 음성 전체 파형 학습 AI")

tab1, tab2 = st.tabs(["📊 분류기 테스트", "📖 기술적 원리 (MFCC)"])

with tab2:
    st.header("음성 파일 전체를 학습하는 MFCC 기술")
    st.markdown("""
    기존의 F1, F2 포먼트 추출 방식이 음성 중간 지점의 '혀의 위치'만 단편적으로 측정했다면, 이 모델은 **MFCC(Mel-Frequency Cepstral Coefficients)** 알고리즘을 도입하여 발음이 시작되고 끝날 때까지의 전체 파형 특성을 학습합니다.
    * 13개의 다차원 변수를 사용하여 '개'나 '게'를 발음할 때 모음 앞뒤로 섞이는 호흡과 자음의 변화 과정 전체를 인공지능이 분석합니다.
    * 고차원의 수치 데이터를 인간의 눈으로 확인할 수 있도록 PCA(주성분 분석) 기법을 사용해 2차원 산점도로 시각화했습니다.
    """)

with tab1:
    if not df.empty:
        st.write(f"✅ 총 **{len(df)}개**의 전체 음성 파형 데이터를 성공적으로 학습했습니다.")
        st.write(f"🎯 테스트 데이터(20%) 기준 모델 정확도: **{accuracy:.1f}%**")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("1. 음성 입력")
            if 'audio_history' not in st.session_state:
                st.session_state.audio_history = []
                
            audio_value = st.audio_input("직접 '개' 또는 '게'를 녹음해 보세요!")
            uploaded_file = st.file_uploader("또는 wav 파일을 업로드하세요.", type=["wav"])
            new_audio = audio_value if audio_value else uploaded_file
            
            if new_audio and new_audio not in st.session_state.audio_history:
                st.session_state.audio_history.append(new_audio)
                
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("⏪ 최근 녹음 삭제") and st.session_state.audio_history:
                    st.session_state.audio_history.pop()
                    st.rerun()
            with col_btn2:
                if st.button("🗑️ 전체 삭제") and st.session_state.audio_history:
                    st.session_state.audio_history.clear()
                    st.rerun()
            
            analyzed_points = []
            
            if st.session_state.audio_history:
                st.subheader("2. 분석 결과")
                for idx, audio_data in enumerate(st.session_state.audio_history):
                    st.markdown(f"**🎤 {idx + 1}번째 녹음**")
                    st.audio(audio_data)
                    
                    with open(f"temp_input_{idx}.wav", "wb") as f:
                        f.write(audio_data.read())
                        audio_data.seek(0)
                    
                    try:
                        y_test, sr_test = librosa.load(f"temp_input_{idx}.wav", sr=22050)
                        mfcc_test = librosa.feature.mfcc(y=y_test, sr=sr_test, n_mfcc=13)
                        mfcc_mean_test = np.mean(mfcc_test.T, axis=0)
                        
                        X_test_scaled = scaler.transform([mfcc_mean_test])
                        prediction = model.predict(X_test_scaled)[0]
                        max_prob = max(model.predict_proba(X_test_scaled)[0]) * 100
                        
                        st.success(f"결과: **{prediction}** (확률: {max_prob:.1f}%)")
                        
                        # 사용자 발음의 13차원 배열을 산점도용 2차원(PCA) 좌표로 변환
                        pca_test = pca.transform(X_test_scaled)[0]
                        analyzed_points.append({
                            "idx": idx + 1, "PCA1": pca_test[0], "PCA2": pca_test[1], "prediction": prediction
                        })
                    except Exception as e:
                        st.error("음성 분석 오류")
                    st.divider()
                    
        with col2:
            st.subheader("3. 전체 음성 파형 분포도 (PCA 시각화)")
            fig = px.scatter(df_pca, x="PCA1", y="PCA2", color="라벨", 
                             title="13차원 MFCC 음향 데이터 공간 분포 (2차원 압축)",
                             color_discrete_map={"ae (ㅐ)": "blue", "e (ㅔ)": "green"})
            
            for pt in analyzed_points:
                marker_color = 'red' if 'ae' in pt['prediction'] else 'orange'
                fig.add_scatter(x=[pt['PCA1']], y=[pt['PCA2']], mode='markers+text',
                                text=[f"#{pt['idx']}"], textposition="top center",
                                marker=dict(size=15, color=marker_color, symbol='star', line=dict(width=1, color='black')),
                                name=f"내 발음 #{pt['idx']}")
                
            st.plotly_chart(fig, use_container_width=True)
