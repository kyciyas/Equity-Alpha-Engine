# Equity Alpha Engine (교차 분석 퀀트 프레임워크)
> **언어(Language)**: [English](./README.md) | [한국어]

## 개요 (Overview)
본 프로젝트는 주식 시장의 교차 분석(Cross-sectional) 트레이딩 프레임워크를 구현하며, 전체 파이프라인의 완성도에 집중합니다:

> **신호 생성(Signal) → 포트폴리오 구축(Portfolio Construction) → 실행(Execution) → 성과 평가(Evaluation)**

목표는 알파 신호가 회전율(Turnover) 및 거래 비용(Transaction Costs)과 같은 **현실적인 포트폴리오 제약 조건** 하에서 어떻게 변화하고 작동하는지 연구하는 것입니다.

---

## 핵심 구성 요소 (Key Components)

### 1. 데이터 파이프라인 (Data Pipeline)
- 한국 및 미국 주식 데이터 지원
- **유동성 필터(거래대금 상위 N개 종목)**를 통한 유니버스 선정
- 마스킹 및 제어된 전방 채우기(Forward-fill)를 통한 결측치 처리
- 정렬된 **가격, 거래량, 수익률 행렬(Matrix)** 출력

---

### 2. 알파 신호 (Alpha Signals)
구현된 교차 분석 신호:
- 모멘텀 (중기)
- 변동성 조정 모멘텀
- 평균 회귀 (단기)
- 잔차 모멘텀 (시장 중립화)

모든 신호는 다양한 타임 호라이즌(Horizon)에 걸쳐 **정보 계수(IC, Information Coefficient)**를 통해 평가됩니다.

---

### 3. 포트폴리오 구축 (Portfolio Construction)

세 가지 포트폴리오 구축 접근 방식을 채택합니다:

#### (1) 고속 휴리스틱 최적화 (Fast Heuristic Optimizer)
- 변동성 스케일링 (Volatility scaling)
- 교차 분석 정규화 및 직접 신호-가중치 매핑 (비최적화 방식)
- 회전율 인식 조정 (Soft constraint)
- **연산 속도와 견고함(Robustness)**에 최적화된 설계

#### (2) 팩터 투영 최적화 (Factor Projection Optimizer / Long-Short)
- 팩터 잔차화(Factor residualization)를 통한 부분적 시장 중립화
- 잔차 기반의 롱-숏 구축 (완전한 베타 중립 제약은 아님)
- 공분산 추정이 필요 없는 구조
- 잔차 기반 알파 추출 및 순위(Rank) 중심의 포트폴리오 형성
- 시장 전체의 충격에 견고함

#### (3) 볼록 최적화 (Convex Optimization / Mean-Variance)
- 현재 미사용 (향후 연구 과제)
- 목적 함수: \(\alpha^T w - \lambda w^T \Sigma w\) 극대화
- 롤링 공분산 추정 (Lookahead-free)
- 외부 추정된 공분산 행렬 사용
- 포함 제약 조건: 시장 중립, 총 레버리지, 종목별 보유 한도, 회전율 제약 등

---

### 4. 실행 및 비용 모델링 (Execution & Cost Modeling)
- **포트폴리오 드리프트(Drift)**를 반영한 주기적 리밸런싱
- 명시적인 회전율(Turnover) 추적
- 손익(PnL)에서 거래 비용 차감 반영

---

### 5. 평가 지표 (Evaluation Metrics)
- 연평균 성장률 (CAGR)
- 샤프 지수 (Sharpe Ratio)
- 최대 낙폭 (MDD)
- 정보 계수 (IC)
- 전이 계수 (TC, Transfer Coefficient)

> - **IC**는 신호의 예측력을 측정하며, **TC**는 생성된 신호와 실제 구현된 포트폴리오 가중치 사이의 상관관계를 측정합니다 (신호 구현 효율성 지표).

---

### 6. 전략 선택 및 앙상블 (Strategy Selection & Ensemble)
- IC 기반 필터링 (StrategyPruner)
- 상관관계 기반 가지치기 (StrategyPruner)
- 전략별 PnL의 소프트맥스(Softmax) 가중치 앙상블 (SoftmaxEnsembleEngine)

---

## 주요 발견 사항 (Key Findings)

- **IC가 반드시 PnL로 직결되지는 않음**  
  포트폴리오 구축 과정에서 신호의 유효성이 크게 왜곡될 수 있습니다.

- **실행(Execution)의 결정적 역할**  
  회전율과 거래 비용은 실현 수익률에 실질적인 영향을 미칩니다.

- **최적화의 트레이드오프**  
  평균-분산 최적화는 약한 신호를 희석시킬 수 있는 반면, 휴리스틱 방법은 알파를 더 효과적으로 보존하는 경우가 많습니다.

- **알파 희석 문제**  
  순위(Rank) 기반 필터링이 신호 대 잡음비(SNR)를 개선하는 데 효과적이었습니다.

---

## 프로젝트 구조 (Project Structure)

```text
project/
├── datahandler/ # 데이터 로딩 및 전처리
├── strategy/    # 신호 생성 및 백테스팅 엔진
├── optimizer/   # 포트폴리오 구축 (Heuristic / CVX)
├── pruner/      # 신호 필터링 (IC, 상관관계)
├── softmax/     # 전략 앙상블
├── main.py      # 통합 실행부
└── README.md
```

## 사용법 (Usage)

```python
# 데이터 핸들러 초기화 (한국 시장)
data = Datahandler.DataHandler_KR(start="2016-06-01", end="2026-03-31", universe="KOSPI")

price = data.get_price()
returns = data.get_returns()
volume = data.get_volume()

# 전략 엔진 설정
engine = Strategy.StrategyEngine(price, returns, volume)

# 알파 신호 추가 (예: 60일 모멘텀, 20일 지연)
engine.add_signal("momentum", lambda p: p.shift(20).pct_change(60))

# 신호 계산
engine.compute_signals()

# 백테스트 실행 및 결과 필터링
results = engine.run_all()
pruner = Pruner.StrategyPruner(results, ic_threshold=0.3, corr_threshold=0.7)

filtered_results = pruner.run()

# 소프트맥스 기반 전략 앙상블
soft_engine = Softmax.SoftmaxEnsembleEngine(filtered_results)

output = soft_engine.run()
```
