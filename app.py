## 1. 상단 임포트 및 페이지 설정 ##
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt



st.set_page_config(page_title="AI 영화 비평가", layout="centered")

## 2. 모델 및 토크나이저 로드 함수 ##
@st.cache_resource
def load_resources():
    # 저장된 모델과 피클 파일을 불러옵니다.
    model = load_model('best_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_resources()
okt = Okt()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','영화','최고','정말','진짜']

# 화면 상단
# <> 페이지 제목
st.title("🎬 AI 영화 리뷰 감성 분석기")
st.info("궁금한 영화의 리뷰를 아래에 입력해보세요.")


# 화면 사이드 바
# <> 정확도 표시
st.sidebar.header("📊 Model Info")
st.sidebar.write("Algorithm: LSTM")
st.sidebar.write("Test Accuracy: **85.20%**")

# 화면 중단
# <> 영화 제목 입력
movie_title = st.text_input("1. 영화 제목을 입력하세요", placeholder="예: 어벤져스")

# <> 영화 리뷰 내용 작성 칸
review_text = st.text_area("2. 영화 리뷰 내용을 입력하세요", placeholder="이 영화 정말 감동적이네요!")

# 화면 하단

# <> 분석 버튼
if st.button("분석 시작"):
    if review_text:
# 실시간 전처리
        new_sentence = okt.morphs(review_text, stem=True)
        new_sentence = [word for word in new_sentence if not word in stopwords]
        encoded = tokenizer.texts_to_sequences([new_sentence])
        pad_new = pad_sequences(encoded, maxlen=80) 
        
        # 예측
        score = float(model.predict(pad_new))
        
        st.divider() 
        # <> 결과
        # <> 영화 리뷰 긍정인지 부정인지        
        
        if score > 0.5:
            st.subheader(f"✅ '{movie_title}' 리뷰 분석 결과: [긍정]")
            st.write(f"인공지능이 판단한 긍정 확률은 **{score*100:.2f}%**입니다.")
        else:
            st.subheader(f"❌ '{movie_title}' 리뷰 분석 결과: [부정]")
            st.write(f"인공지능이 판단한 부정 확률은 **{(1-score)*100:.2f}%**입니다.")

        # <> 해당 영화의 AI 평점            
        # 평점 표시
        ai_score = round(score * 10, 1)
        st.metric(label=f"이 영화의 AI 평점은?", value=f"{ai_score} / 10점")
        
        if score > 0.8: st.balloons()
    else:
        st.warning("리뷰 내용을 입력해주세요!")