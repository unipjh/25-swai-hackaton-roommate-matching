import numpy as np

class RoommateMatcher:
    """
    지능형 매칭 엔진 v2.0
    - W0 (고정 베이스라인) + dW (개인화 보정치) 결합 구조
    - 8개 생활 특징 데이터를 7개 가중치 카테고리로 매핑하여 연산
    """
    def __init__(self, user_profile: dict):
        self.me = user_profile
        # 1. 설문 데이터 기반 확정된 W0 (7개 카테고리)
        # [수면, 흡연, 잠버릇, 냉난방, 청결, 소음, 외출/음주]
        self.w0 = np.array([0.233, 0.167, 0.113, 0.133, 0.147, 0.173, 0.033])
        
        # 2. 피처 벡터 구성을 위한 키 순서 정의 (8개 피처)
        self.feature_keys = [
            "sleep_time_val", "wake_time_val", "clean_cycle_val", 
            "hvac_val", "alarm_val", "outing_val", "smoke", "sleep_habit_val"
        ]
        
        # 3. 최종 가중치 벡터 생성
        self.weights = self._generate_final_weights()

    def _generate_final_weights(self):
        """W0와 dW를 합산하고 8차원 피처 벡터에 맞게 확장"""
        # DB에서 넘어온 현재 유저의 dW (없으면 0.0)
        dw = np.array([
            self.me.get('w_sleep', 0.0),       # 0. 수면
            self.me.get('w_smoke', 0.0),       # 1. 흡연
            self.me.get('w_sleep_habit', 0.0), # 2. 잠버릇
            self.me.get('w_hvac', 0.0),        # 3. 냉난방
            self.me.get('w_clean_cycle', 0.0), # 4. 청결
            self.me.get('w_noise', 0.0),       # 5. 소음
            self.me.get('w_outing', 0.0)        # 6. 외출
        ])
        
        # 합산 가중치 (7차원)
        combined_w = self.w0 + dw
        
        # 8차원 피처 벡터에 매핑 (수면 가중치는 취침/기상에 각각 적용)
        return np.array([
            combined_w[0], combined_w[0], # sleep_time, wake_time
            combined_w[4],                # clean_cycle
            combined_w[3],                # hvac
            combined_w[5],                # alarm
            combined_w[6],                # outing
            combined_w[1],                # smoke
            combined_w[2]                 # sleep_habit
        ])

    def check_filter(self, target: dict):
        """매칭 불가 조건 체크 (Hard Constraints)"""
        # 성별 체크
        if self.me.get('gender') != target.get('gender'):
            return False, "성별 불일치"
        # 흡연 차단 조건
        if self.me.get('block_smoke') and target.get('smoke'):
            return False, "흡연자 기피"
        # 잠버릇 차단 조건
        if self.me.get('block_sleep_habit') and target.get('sleep_habit') != 'none':
            return False, "잠버릇 기피"
        return True, None

    def get_score(self, target: dict):
        """가중 코사인 유사도 기반 최종 점수 산출"""
        is_pass, reason = self.check_filter(target)
        if not is_pass:
            return 0.0, [reason]

        # 8차원 벡터 생성 (잠버릇 등 문자열은 사전에 numeric으로 변환되어 있어야 함)
        vec_me = np.array([float(self.me.get(k, 0.5)) for k in self.feature_keys])
        vec_target = np.array([float(target.get(k, 0.5)) for k in self.feature_keys])

        # 가중치 적용 (W * X)
        wa = self.weights * vec_me
        wb = self.weights * vec_target

        # 코사인 유사도 연산
        norm_a = np.linalg.norm(wa)
        norm_b = np.linalg.norm(wb)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0, ["데이터 부족"]

        score = np.dot(wa, wb) / (norm_a * norm_b)
        
        # 리스크 분석 (차이가 큰 항목 추출)
        risks = []
        diff = np.abs(vec_me - vec_target) * self.weights
        if diff[2] > 0.15: risks.append("청소 주기 불일치")
        if diff[0] > 0.15 or diff[1] > 0.15: risks.append("생활 시간대 차이")

        return float(score), risks