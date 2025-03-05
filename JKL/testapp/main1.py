import os
import cv2
import asyncio
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from starlette.websockets import WebSocketDisconnect
from ultralytics import YOLO  # YOLOv8 모델 사용
from deep_sort_realtime.deepsort_tracker import DeepSort

app = FastAPI()

# ✅ 정적 파일 제공 (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ HTML 페이지 반환 (index.html)
@app.get("/")
async def serve_index():
    return FileResponse("static/index2.html")

# ✅ 저장할 디렉토리
UPLOAD_DIR = "uploads"
OUTPUT_VIDEO = "static/output.mp4"  # 결과 비디오 저장 경로
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ YOLOv8 + DeepSORT 초기화
yolo_model = YOLO("static/best_3000_xl.pt")  # YOLO 모델
tracker = DeepSort(max_age=30)

# ✅ 웹소켓 연결 관리
class ConnectionManager:
    def __init__(self):
        self.active_connections = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_frame(self, frame: np.ndarray):
        """프레임을 WebSocket을 통해 전송"""
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        for connection in self.active_connections:
            await connection.send_text(frame_data)

manager = ConnectionManager()

# ✅ MP4 파일 업로드 API
@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    """사용자가 업로드한 MP4 파일 저장"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # ✅ 업로드된 비디오를 처리하는 `process_video()` 실행
    asyncio.create_task(process_video(file_path))
    
    return {"filename": file.filename, "path": file_path}

# ✅ YOLO + DeepSORT 처리 & 실시간 프레임 스트리밍
async def process_video(file_path):
    """YOLO + DeepSORT 적용 후 WebSocket으로 실시간 프레임 전송"""
    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ YOLO 객체 탐지 수행
        results = yolo_model(frame)
        detections = []
        for r in results:
            for box in r.boxes.data:
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                detections.append(([x1, y1, x2, y2], conf, int(cls)))

        # ✅ DeepSORT 트래커 적용
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if track.is_confirmed():
                x1, y1, x2, y2 = track.to_tlbr()
                track_id = track.track_id
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ✅ 웹소켓으로 실시간 전송
        await manager.send_frame(frame)

        # ✅ 최종 영상 저장
        out.write(frame)

        await asyncio.sleep(0.03)  # 30 FPS 유지

    cap.release()
    out.release()
    print("✅ 영상 처리 완료!")

from fastapi import HTTPException

# ✅ YOLO 모델 초기화
yolo_model = YOLO("static/best_3000_xl.pt")  # YOLO 모델

# ✅ 주차 공간을 저장할 변수
assigned_parking_spot = None

@app.post("/assign_parking/")
async def assign_parking(data: dict):
    """사용자가 클릭한 좌표에서 가장 가까운 'free' 주차칸을 찾아 배정"""
    global assigned_parking_spot, user_selected_point

    user_x, user_y = data.get("x"), data.get("y")

    if user_x is None or user_y is None:
        raise HTTPException(status_code=400, detail="Invalid coordinates")

    # ✅ 사용자가 클릭한 지점 저장 (파란 점 표시)
    user_selected_point = (user_x, user_y)

    # ✅ 가장 가까운 주차칸 찾기
    min_distance = float("inf")
    nearest_spot = None

    for result in last_yolo_results:
        x1, y1, x2, y2, conf, spot_number = result
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        distance = ((center_x - user_x) ** 2 + (center_y - user_y) ** 2) ** 0.5

        if distance < min_distance:
            min_distance = distance
            nearest_spot = (center_x, center_y)

    if nearest_spot:
        assigned_parking_spot = nearest_spot  # 선택된 주차칸 저장
        return {"assigned_x": nearest_spot[0], "assigned_y": nearest_spot[1]}
    else:
        return {"message": "No available parking spot found"}



# ✅ YOLO + DeepSORT 처리 & 실시간 프레임 스트리밍
last_yolo_results = []  # 최신 YOLO 감지 결과 저장

# ✅ 사용자 클릭 위치 저장 변수 추가
user_selected_point = None  # 사용자가 클릭한 좌표 (파란 점)

# ✅ YOLO + DeepSORT 처리 & 실시간 프레임 스트리밍
last_yolo_results = []  # 최신 YOLO 감지 결과 저장
user_selected_point = None  # 사용자가 클릭한 좌표 (파란 점)
assigned_parking_spot = None  # 배정된 주차칸 좌표

async def process_video(file_path):
    """YOLO + DeepSORT 적용 후 WebSocket으로 실시간 프레임 전송"""
    global last_yolo_results, user_selected_point, assigned_parking_spot

    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("❌ 영상 파일을 열 수 없습니다:", file_path)
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ YOLO 객체 탐지 수행
        results = yolo_model(frame)
        detections = []
        parking_spot_counter = 1  # 주차칸 번호 카운터
        updated_results = []  # YOLO 감지된 객체 저장 리스트

        for r in results:
            for box in r.boxes.data:
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                class_id = int(cls)

                # ✅ YOLO 모델이 감지한 주차칸에 번호 부여 (1부터 순차적으로)
                updated_results.append((x1, y1, x2, y2, conf, parking_spot_counter))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 초록색 박스 (주차 공간)
                cv2.putText(frame, str(parking_spot_counter), (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 번호 출력
                parking_spot_counter += 1

                # ✅ 움직이는 차량 감지를 위한 YOLO 결과 추가
                if class_id in [2, 3, 5, 7]:  # 2=자동차, 3=오토바이, 5=버스, 7=트럭
                    detections.append(([x1, y1, x2, y2], conf, class_id))

        # ✅ YOLO 탐지 결과를 전역 변수 `last_yolo_results`에 업데이트
        last_yolo_results[:] = updated_results  # 리스트 전체를 업데이트

        # ✅ DeepSORT 트래커 적용 (움직이는 차량만 추적)
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if track.is_confirmed():
                x1, y1, x2, y2 = track.to_tlbr()
                track_id = track.track_id
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)  # 노란색 박스 (움직이는 차량)
                cv2.putText(frame, f"Car {track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # ✅ 사용자 선택한 특정 지점에 파란 점 유지
        if user_selected_point:
            cv2.circle(frame, (int(user_selected_point[0]), int(user_selected_point[1])), 10, (255, 0, 0), -1)  # 파란 점

        # ✅ 배정된 주차 공간에 파란 점 유지
        if assigned_parking_spot:
            cv2.circle(frame, (int(assigned_parking_spot[0]), int(assigned_parking_spot[1])), 10, (255, 0, 0), -1)  # 파란 점

        # ✅ 웹소켓으로 실시간 전송
        await manager.send_frame(frame)

    cap.release()
    print("✅ 영상 처리 완료!")




@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """클라이언트와 WebSocket 연결 관리"""
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(1)  # 클라이언트 연결 유지
    except WebSocketDisconnect:
        print("❌ 클라이언트 연결 종료")
        manager.disconnect(websocket)

# ✅ 비디오 파일 다운로드 API
@app.get("/download/")
async def download_video():
    """최종 처리된 MP4 영상 다운로드"""
    return {"download_url": f"http://localhost:8000/static/output.mp4"}
