import numpy as np
# 토큰화(Tokenization) 및 불용어 제거(Stopwords Removal)
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences


# 1. 형태소 분석기 객체 생성
okt = Okt()

# 2. 불용어 리스트 정의 (감성 분석에 도움이 안 되는 단어들)
# 분석을 해보면서 나중에 여기에 단어를 추가하면 정확도가 올라갑니다!
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','영화','최고','정말','진짜']
max_len = 30  # 최대 길이 30
# 3. 테스트 (한 번 쪼개볼까요?)
print(okt.morphs("이 영화 진짜 재밌는데 왜 다들 욕하지?", stem=True))

# --- [함수부] 실시간 서비스용 변환 함수 ---
def preprocess_review(new_sentence, tokenizer):
    """
    사용자가 입력한 텍스트를 모델이 읽을 수 있는 숫자 배열로 변환합니다.
    """
    # 1. 형태소 분석 및 어간 추출
    new_sentence = okt.morphs(new_sentence, stem=True) 
    
    # 2. 불용어 제거
    new_sentence = [word for word in new_sentence if not word in stopwords] 
    
    # 3. 정수 인코딩 (전달받은 tokenizer 사전 활용)
    encoded = tokenizer.texts_to_sequences([new_sentence]) 
    
    # 4. 패딩 (길이 맞추기)
    pad_new = pad_sequences(encoded, maxlen=max_len) 
    
    return pad_new

'''

# 훈련 데이터 토큰화 (리스트 형식으로 저장)
X_train = []
for sentence in train_data['document']:
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_train.append(stopwords_removed_sentence)

# 상위 3개만 확인
print(X_train[:3])

# (위의 X_train 코드 아래에 이어서 붙이세요)

# 테스트 데이터도 똑같이 토큰화 진행
X_test = []
for sentence in test_data['document']:
    tokenized_sentence = okt.morphs(sentence, stem=True)
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
    X_test.append(stopwords_removed_sentence)

print("전체 데이터 토큰화 완료!")

# 정수 인코딩(Integer Encoding)
# 1. 단어 집합(Vocabulary) 만들기 
# 훈련 데이터를 기반으로 어떤 단어에 몇 번 번호를 줄지 결정합니다.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train) # 훈련 데이터에 단어별 고유번호 할당

# 2. 텍스트를 정수 인덱스로 변환
# 예: ['영화', '재밌다'] -> [15, 203]
X_train = tokenizer.texts_to_sequences(X_train) # 문자를 숫자리스트로 변환 ex) In ['영화', '최고'] Out[0, 0, 0, ... , 12, 5 ]
X_test = tokenizer.texts_to_sequences(X_test)

# 3. 결과 확인
print(X_train[:3])

# 1. 모든 리뷰의 길이를 동일하게 맞춤 (가장 긴 리뷰 기준 또는 일정 길이)
# 여기서는 가장 긴 리뷰의 길이를 확인하거나 보통 30~50 정도로 잡습니다.
''''''
pad_sequence(변수명, maxlen=길이지정)
ex) maxlen 30가정
In [12, 5]              길이 : 2
Out [0, 0, ... , 1, 5]  길이 : 30'''
'''


X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# 2. 결과 확인 (모든 데이터가 30개의 숫자로 채워졌는지 확인)
print(f"pad_sequence 반영 후, 훈련데이터의 모양\n, {X_train.shape}")
print(f"pad_sequence 반영 후, 훈련데이터 0번째 인덱스 값 추출\n{X_train[0]}")

# 훈련 데이터의 정답(0: 부정, 1: 긍정) 추출
y_train = np.array(train_data['label']) # 훈련 데이터의 label 컬럼을 y_훈련 데이터로 둚.

# 테스트 데이터의 정답 추출
y_test = np.array(test_data['label']) # 테스트 데이터의 label 컬럼을 y_테스트 데이터로 둚.'''