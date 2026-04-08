[ PRD / TRD 및 기획안 (Real-time Sync) ]

# 1. PRD (Product Requirements Document)
* 목적: 기존의 GUI(PySide6) 의존성을 완전히 걷어내고, TensorFlow/Keras 기반 물리 데이터 학습을 독립적으로 검증할 수 있는 교육용 시뮬레이션 환경 구축.
* 핵심 기능:
  - 1D 함수 근사, 포물선 궤적, 모델 과적합/과소적합, 진자 운동 예측에 대한 4가지 독립 Lab 스크립트 실행.
  - 실행 시 사용자의 추가 조작 없이 `outputs/` 디렉토리에 고품질 PNG 분석 그래프 자동 생성.
* 동적 업데이트 목표: 
  - 신규 물리 모델 추가나 테스트 결과 피드백 발생 시, 요구사항을 즉각적으로 기획과 코드에 연동하여 실시간 반영.

# 2. TRD (Technical Requirements Document)
* 핵심 기술: Python 3.x, TensorFlow/Keras, NumPy, Matplotlib.
* 아키텍처: 프론트엔드를 배제한 단독 실행형 스크립트(Standalone) 구조. 코어 수식 및 신경망 학습 로직에만 리소스 집중.
* 데이터 파이프라인:
  - 순차 데이터에서 발생하는 Validation Split 오류를 방지하기 위해 `np.random.permutation`을 통한 데이터 셔플링 필수 적용.
  - RK4(Runge-Kutta 4차) 수치 적분을 활용한 진자 운동 시뮬레이션 데이터 동적 생성.
* 동적 업데이트 목표:
  - 하이퍼파라미터(예: Learning rate, Batch size, Dropout) 튜닝 지표를 실시간으로 추적하여 네트워크 레이어 아키텍처 지속 개선.

# 3. Brainstorming & Action Plan (Superpowers)
* 현황 분석: 복잡한 시각적 인터페이스가 제거되면서, "AI가 물리 법칙을 어떻게 모사하는가"에 대한 코어 기능이 훨씬 명확해졌습니다.
* 다음 단계 계획 (Make-Plan):
  - [Phase 1] 03overfitting.py에 구현된 성능 비교 테이블 생성 기능을 다른 Lab 스크립트(포물선, 진자 운동 등)로 확장하여 일관된 평가 지표 마련.
  - [Phase 2] 다음 주차에 예정된 PINNs(Physics-Informed Neural Networks) 도입을 대비해, 데이터 기반(Data-driven) Loss 함수를 수식 기반(Equation-driven)으로 전환할 수 있는 커스텀 템플릿 초안 작성.
  - [Phase 3] 심화 과제(스프링-질량 시스템, 2D Ising Model 등)로 즉각 확장할 수 있는 모듈형 데이터 제너레이터 설계.
