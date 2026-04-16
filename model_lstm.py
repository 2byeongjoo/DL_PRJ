import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import okt, stopwords  # 병주님의 utils.py 연동

class ReviewLSTM:
    def __init__(self):
        # 1. 병주님의 파일명(best_model.h5, tokenizer.pickle)에 맞춤
        self.model = load_model('best_model.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.max_len = 30 # utils.py의 기준

    def predict_sentiment(self, text):
        # 2. utils.py의 객체를 사용한 전처리
        # 한글 외 문자 제거 등 추가 전처리가 필요하면 여기에 넣을 수 있습니다.
        tokenized = okt.morphs(text, stem=True)
        sw_removed = [word for word in tokenized if not word in stopwords]
        
        # 3. 숫자로 변환 및 패딩
        encoded = self.tokenizer.texts_to_sequences([sw_removed])
        padded = pad_sequences(encoded, maxlen=self.max_len)
        
        # 4. 예측 (0~1 사이의 확률값 반환)
        score = float(self.model.predict(padded))
        return score