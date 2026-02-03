


# 🏠 Roommate Matching & Personalization AI Engine

> **룸메이트 매칭 및 관리 솔루션**
> (Hackathon Project  🥈)

본 프로젝트는 **생활 패턴 기반 룸메이트 매칭**과  
**사용자 피드백을 통한 개인화 학습**을 목표로 설계된 AI 엔진입니다.

해커톤 실사용 시나리오를 가정하여,  
**AI 로직을 독립 서버로 분리하고 백엔드와 연동 가능한 구조**로 구현되었습니다.

---

## 🔍 Problem Definition

기존 룸메이트 매칭 서비스의 한계는 다음과 같습니다.

- 단순 조건 필터링 (흡연 여부, 기상 시간 등)
- 개인 성향 반영 부족
- 사용 후 만족도 피드백이 매칭에 반영되지 않음

👉 본 프로젝트는  
**“정적 조건 + 동적 개인화 가중치(dW)”** 구조를 통해  
사용자 경험이 누적될수록 매칭 품질이 개선되는 시스템을 지향합니다.

---

## 🧠 Core AI Logic

### 1. Matching Engine (Inference)

- **Weighted Cosine Similarity**
- 가중치 구조:
  - `W0` : 설문 기반 고정 베이스라인
  - `dW` : 사용자별 개인화 보정치
- 최종 가중치:  
```

W = W0 + dW

```

- 생활 특징 8개 → 7개 가중치 카테고리로 매핑
  - 수면
  - 흡연
  - 잠버릇
  - 냉/난방
  - 청결
  - 소음
  - 외출/음주

- Hard Constraint 필터
  - 성별 불일치
  - 흡연/잠버릇 차단 조건

---

### 2. Feedback-based Personalization (Learning)

매칭 이후 사용자 만족도를 기반으로 **dW를 업데이트**합니다.

#### (1) 정량적 업데이트
- Binary Cross Entropy 기반 Gradient
- 매칭 점수 vs 실제 만족도(label)

#### (2) 정성적 업데이트
- Gemini API 활용
- 사용자 리뷰 텍스트 분석
- 특정 항목에 대한 불만/만족을 가중치 변화량으로 변환

```

dW_new = dW_old + dW_BCE + dW_LLM

```

- 가중치 안정성을 위해 클리핑 적용 (0.0 ~ 0.5)

---

## 🏗️ System Architecture

```

[ Backend Server ]
│
│ (HTTP / REST)
▼
[ AI Personalization Server ]
│
├─ Matching API
└─ Feedback API (Gemini LLM)

```

- AI 서버 독립 운영
- 백엔드에서는 **ngrok을 통해 AI 서버 접근**
- 실서비스 연동을 고려한 구조 설계

---

## 🛠 Tech Stack

- **Python**
- **FastAPI**
- **NumPy**
- **Pydantic**
- **Google Gemini API**
- **ngrok (development)**

---

## 📁 Project Structure

```

.
├── main.py        # FastAPI entrypoint
├── matcher.py    # Matching Engine (W0 + dW)
├── feedback.py   # Feedback & Learning Engine
├── .env          # Gemini API Key (not included)
├── requirement.txt          # Gemini API Key (not included)
└── README.md

````

---

## ⚙️ Environment Setup

```bash
pip install -r requirements.txt
````

`.env` 파일 생성:

```
GEMINI_API_KEY=YOUR_API_KEY
```

서버 실행:

```bash
python main.py
```

---

## 🧑‍💻 My Role

* **AI 로직 전체 설계 및 구현**
* 매칭 알고리즘 설계 (W0 + dW 구조)
* 피드백 기반 개인화 학습 로직 설계
* Gemini API 기반 정성 분석 파이프라인 구축
* AI 서버 분리 아키텍처 설계
* 백엔드 연동을 고려한 API 명세 및 구조 정의

---

## 🏆 Result

* Hackathon **Silver Award (은상)**
* 실사용 시나리오 기반 AI 매칭/학습 구조 검증
* 서비스 확장 가능한 아키텍처 설계 경험

---

## 📌 Notes

본 프로젝트는 **해커톤 당시 구현된 실험적 버전**이며,
향후 기회가 되면 확장 예정입니다.


