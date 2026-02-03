import numpy as np
import google.generativeai as genai
import json

class FeedbackEngine:
    """
    사용자 피드백 기반 dW 업데이트 엔진
    - BCE Gradient: 통계적 오차 보정
    - Gemini API: 텍스트 기반 정성적 보정
    """
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel('gemini-1.5-flash')
        self.categories = [
            'w_sleep', 'w_smoke', 'w_sleep_habit', 
            'w_hvac', 'w_clean_cycle', 'w_noise', 'w_outing'
        ]

    def map_to_7_vec(self, p_dict: dict):
        """8개 피처 데이터를 7개 가중치 차원 벡터로 변환"""
        return np.array([
            (p_dict.get('sleep_time_val', 0.5) + p_dict.get('wake_time_val', 0.5)) / 2,
            float(p_dict.get('smoke', False)),
            1.0 if p_dict.get('sleep_habit') != 'none' else 0.0,
            p_dict.get('hvac_val', 0.5),
            p_dict.get('clean_cycle_val', 0.5),
            p_dict.get('alarm_val', 0.5),
            p_dict.get('outing_val', 0.5)
        ])

    async def get_new_dw_dict(self, user: dict, target: dict, score: float, label: int, review: str, eta: float = 0.05):
        """최종 갱신된 dW를 Dictionary 형태로 반환 (백엔드 UPDATE용)"""
        
        # 1. 벡터 준비
        u_vec = self.map_to_7_vec(user)
        t_vec = self.map_to_7_vec(target)
        old_dw = np.array([user.get(cat, 0.0) for cat in self.categories])

        # 2. 정량적 업데이트 (BCE Gradient)
        # score가 100점 만점으로 들어올 경우 0~1로 스케일링
        normalized_score = score / 100.0 if score > 1.0 else score
        error = normalized_score - label
        delta_bce = -eta * error * np.abs(u_vec - t_vec)

        # 3. 정성적 업데이트 (Gemini LLM)
        delta_llm = np.zeros(7)
        if review and len(review.strip()) > 5:
            prompt = f"""
            당신은 룸메이트 만족도 분석 전문가입니다. 리뷰를 분석해 7개 항목의 가중치 변화량(dW)을 JSON으로 반환하세요.
            항목: {self.categories}
            규칙: 특정 항목에 대한 강한 불만 시 +0.05, 미온적 불만 +0.02, 만족 혹은 언급 없으면 0.
            리뷰: "{review}"
            반드시 다음 JSON 포맷으로만 답변하세요: {{"w_sleep": 0.0, ...}}
            """
            try:
                response = await self.llm.generate_content_async(prompt)
                res_json = json.loads(response.text.replace('```json', '').replace('```', '').strip())
                delta_llm = np.array([float(res_json.get(cat, 0.0)) for cat in self.categories])
            except Exception as e:
                print(f"LLM 분석 실패: {e}")

        # 4. 합산 및 가드레일 (dW_new = dW_old + d_bce + d_llm)
        new_dw_vec = np.clip(old_dw + delta_bce + delta_llm, 0.0, 0.5)

        # 5. 백엔드 DB 컬럼명 매핑 반환
        return {cat: float(new_dw_vec[i]) for i, cat in enumerate(self.categories)}