# model_llm.py
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class ReviewLLM:
    def __init__(self):
        # 1. Ollama 모델 설정 (설치하신 모델명에 맞게 수정 가능: llama3, mistral 등)
        self.llm = Ollama(model="llama3") 
        
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
        
        비평 결과:
        """
        self.prompt = PromptTemplate(template=self.template, input_variables=["review", "score"])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def analyze_review(self, review_text, sentiment_score):
        # 점수를 퍼센트로 변환해서 전달
        display_score = round(sentiment_score * 100, 2)
        return self.chain.run(review=review_text, score=display_score)