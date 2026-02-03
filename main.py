import os
import json
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# 사용자 정의 모듈 임포트
from matcher import RoommateMatcher
from feedback import FeedbackEngine

# 환경 변수 로드 (.env 파일에 GEMINI_API_KEY 저장 필요)
load_dotenv()

app = FastAPI(title="AI PM Personalization Engine v2")

# --- [1. API 설정 및 엔진 초기화] ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # 로컬 테스트용 직접 입력 (배포 시에는 .env 사용 권장)
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

feedback_engine = FeedbackEngine(api_key=GEMINI_API_KEY)

# --- [2. DTO 정의: 백엔드 규격 준수] ---

class UserProfile(BaseModel):
    id: str
    nickname: str
    gender: str
    smoke: bool
    sleep_habit: str
    sleep_time_val: float
    wake_time_val: float
    clean_cycle_val: float
    hvac_val: float
    alarm_val: float
    outing_val: float
    # dW DB에서 가져온 현재 유저의 개인화 가중치
    w_sleep: float 
    w_smoke: float 
    w_sleep_habit: float 
    w_hvac: float 
    w_clean_cycle: float 
    w_noise: float 
    w_outing: float 
    # 필터 조건
    block_smoke: bool = False
    block_sleep_habit: bool = False

class MatchRequest(BaseModel):
    user_profile: UserProfile
    candidates: List[UserProfile]

class MatchResultItem(BaseModel):
    nickname: str
    score: float
    risks: List[str]

class WeightUpdate(BaseModel):
    """백엔드 dW DB 업데이트용 포맷"""
    w_sleep: float
    w_smoke: float
    w_sleep_habit: float
    w_hvac: float
    w_clean_cycle: float
    w_noise: float
    w_outing: float

class AIResponse(BaseModel):
    results: List[MatchResultItem]
    updated_weights: Optional[WeightUpdate] = None

class FeedbackRequest(BaseModel):
    user_profile: UserProfile
    target_profile: UserProfile
    score: float # 매칭 당시 점수
    label: int   # 만족도 (1:만족, 0:불만족)
    review_text: str = ""
    eta: float = 0.05 # 학습률

# --- [3. API 엔드포인트] ---

@app.post("/api/v1/match", response_model=AIResponse)
async def match_endpoint(data: MatchRequest):
    """
    [추천 단계] 
    - W0(고정) + dW(현재값)를 사용하여 점수 산출.
    - W0는 갱신되지 않음.
    """
    try:
        # 1. 매칭 엔진 초기화 (user_profile 내의 dW 값들이 포함됨)
        matcher = RoommateMatcher(data.user_profile.dict())
        
        results = []
        for cand in data.candidates:
            # 2. 독립적인 점수 계산 수행
            score, risks = matcher.get_score(cand.dict())
            if score > 0:
                results.append({
                    "nickname": cand.nickname,
                    "score": round(score * 100, 1),
                    "risks": risks
                })
        
        # 3. 현재의 dW 상태를 그대로 반환 (백엔드 확인용)
        current_dw = WeightUpdate(**{k: getattr(data.user_profile, k) for k in [
            'w_sleep', 'w_smoke', 'w_sleep_habit', 'w_hvac', 'w_clean_cycle', 'w_noise', 'w_outing'
        ]})

        return {
            "results": sorted(results, key=lambda x: x['score'], reverse=True),
            "updated_weights": current_dw
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/feedback", response_model=WeightUpdate)
async def feedback_endpoint(data: FeedbackRequest):
    """
    [학습 단계] 
    - 전달받은 score와 label을 비교하여 새로운 dW_new를 계산.
    - dW_new = dW_old + d(BCE) + d(LLM)
    """
    try:
        # 1. 피드백 엔진을 통해 새로운 dW 벡터 계산
        # 이 과정에서 내부적으로 BCE Gradient와 Gemini API 호출이 이루어짐
        new_dw_dict = await feedback_engine.get_new_dw_dict(
            user=data.user_profile.dict(),
            target=data.target_profile.dict(),
            score=data.score,
            label=data.label,
            review=data.review_text,
            eta=data.eta
        )
        
        # 2. 백엔드 dW DB에 즉시 덮어씌울 수 있는 dict 반환
        return new_dw_dict

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)