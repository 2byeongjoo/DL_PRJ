## 1. 상단 임포트 및 페이지 설정 ##
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt
from utils import preprocess_review

st.set_page_config(page_title="AI 영화 비평가", layout="centered")  # html의 title 태그 같은 역할

## 2. 모델 및 토크나이저 로드 함수 ##
@st.cache_resource   # 모델을 메모리에 고정하여 버튼을 누를 때마다 새로 읽지 않도록 속도를 최적화합니다.
def load_resources():
    # 학습이 완료된 딥러닝 모델 파일(.h5)을 불러옵니다.
    model = load_model('best_model.h5')
    # 단어를 숫자로 변환해주는 사전 파일(pickle)을 불러옵니다.
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer


# 함수를 실행하여 모델과 토크나이저를 변수에 저장합니다.
model, tokenizer = load_resources()
# 한국어 형태소 분석기(가위 역할)를 준비합니다.
okt = Okt()
# 분석 시 제외할 의미 없는 단어(불용어) 명단을 정의합니다.
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
        '''
        # 실시간 전처리
        # 문장을 단어 단위로 쪼개고, '재밌어요'를 '재밌다'처럼 기본형으로 변환합니다.
        new_sentence = okt.morphs(review_text, stem=True)
        
        # 의미 없는 단어(은, 는, 이, 가 등)를 제거하여 핵심 단어만 남깁니다.
        new_sentence = [word for word in new_sentence if not word in stopwords]
        
        # 단어들을 미리 정해진 숫자 번호로 바꿉니다. (예: '최고' -> 15번)
        encoded = tokenizer.texts_to_sequences([new_sentence])
        
        # 모든 입력 길이를 30자로 맞춥니다. (짧으면 0을 채움)
        pad_new = pad_sequences(encoded, maxlen=30) 
        '''
        pad_new = preprocess_review(review_text, tokenizer)
        
        # 예측
        # 학습된 모델이 데이터를 읽고 0(부정) ~ 1(긍정) 사이의 확률 점수를 내놓습니다.
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