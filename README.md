# Smart Parking Recommendation System

> 주차장 영상을 기반으로 **비어있는 주차 공간을 탐지하고**,  
> 사용자가 지정한 위치를 기준으로 **가장 가까운 빈 공간을 추천**하는 AI 시스템입니다.

---

## 🧠 프로젝트 개요

- YOLO + DeepSORT를 활용한 실시간 차량 감지 및 추적  
- 사전 학습된 머신러닝 모델을 통한 빈자리/점유자리 판단  
- 사용자가 클릭한 기준 좌표에서 가까운 빈자리 추천 가능  
- FastAPI 기반 웹 인터페이스를 통해 영상 업로드, 분석 결과 스트리밍/다운로드 제공  

---

## 👤 담당 역할 (원정환)

### ✅ 1. YOLO + DeepSORT 기반 차량 추적 시스템 구현

- **YOLOv8 모델로 프레임 단위 차량 탐지**
- DeepSORT를 연동해 차량의 ID를 부여하고 프레임 간 **지속적인 객체 추적**
- 추적된 ID를 프레임에 시각화하여 결과 영상으로 저장

### ✅ 2. 영상 분석 API 서버 구현 (FastAPI)

- FastAPI를 기반으로 영상 업로드, 분석, 결과 반환까지 처리하는 API 설계
- `/upload_video/`, `/stream_video/`, `/download_video/` 등 엔드포인트 구성
- HTML + JS 기반 UI를 통해 영상 업로드 후 자동 스트리밍 및 다운로드 제공

### ✅ 3. 주차 공간 분석 결과 시각화

- OpenCV를 활용해 분석된 주차 영역에 **초록(빈 공간) / 빨강(점유 공간)** 바운딩 박스 시각화
- Plotly + ipywidgets를 활용한 **인터랙티브 주차장 시각화** 구현
- 단일 이미지 및 전체 영상 단위로 결과 영상 및 이미지 저장

---

## 🛠 사용 기술

| 구분        | 기술/라이브러리 |
|-------------|----------------|
| Language    | Python         |
| Vision      | OpenCV, YOLOv8, DeepSORT, scikit-image |
| ML Model    | Scikit-learn, pickle 기반 분류기 |
| Framework   | FastAPI        |
