# model_llm.py
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser # 추가

class ReviewLLM:
    def __init__(self):
        # 1. Ollama 모델 설정
        self.llm = Ollama(
            base_url="https://intermetacarpal-concepcion-fondlingly.ngrok-free.dev/", # ◀ 방금 뜬 따끈따끈한 주소!
            model="llama3:latest"
        )
        
        # 2. 비평을 위한 프롬프트 템플릿 설계
        self.template = """
        당신은 전문 영화 및 상품 비평가입니다. 
        사용자의 리뷰 내용과 인공지능이 분석한 감성 점수를 바탕으로 심층 비평을 작성해주세요.
        
        [리뷰 내용]: {review}
        [감성 분석 점수]: {score}% (100%에 가까울수록 긍정, 0%에 가까울수록 부정)
        
        작성 가이드:
        1. 이 점수가 나온 이유를 리뷰 내용에서 찾아 분석하세요.
        2. 전문 용어를 섞어 설득력 있게 비평하세요.
        3. 마지막에는 소비자나 제작자에게 주는 한 줄 조언을 포함하세요.
        4. (매우 중요!!!) 반드시!!! `한국어`로 답하세요
        
        비평 결과(한국어로 작성):
        """
        self.prompt = PromptTemplate(template=self.template, input_variables=["review", "score"]) # input_variables=["review", "score"]
        '''
        self.template 부분
        설명: LLM에게 "넌 비평가야"라는 **역할(Persona)**을 주고, 
        LSTM에서 나온 숫자를 해석하는 논리적 가이드라인을 준 것입니다. 
        이게 없으면 LLM은 그냥 평범한 대답만 내놓게 됩니다.
        '''
        # 3. 최신 방식(Chain) 구성: | 기호를 사용해 연결합니다.
        self.chain = self.prompt | self.llm | StrOutputParser()

    def analyze_review(self, review_text, sentiment_score):
        # 점수를 퍼센트로 변환해서 전달
        display_score = round(sentiment_score * 100, 2)
        # run 대신 invoke를 사용합니다 (최신 규격)
        return self.chain.invoke({"review": review_text, "score": display_score})