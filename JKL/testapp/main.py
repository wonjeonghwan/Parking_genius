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

import os
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory='static'), name="static")

from fastapi import HTTPException

# 주차 공간 좌표 예제 (임시 데이터)
parking_spots = [
    {"x": 100, "y": 150, "class": "free"},
    {"x": 250, "y": 300, "class": "free"},
    {"x": 400, "y": 200, "class": "occupied"},
    {"x": 500, "y": 350, "class": "free"}
]

@app.post("/nearest_parking_spot/")
async def find_nearest_parking(data: dict):
    user_x, user_y = data.get("x"), data.get("y")

    if user_x is None or user_y is None:
        raise HTTPException(status_code=400, detail="Invalid coordinates")

    nearest_spot = None
    min_distance = float("inf")

    for spot in parking_spots:
        if spot["class"] == "free":
            distance = ((spot["x"] - user_x) ** 2 + (spot["y"] - user_y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_spot = spot

    if nearest_spot:
        return {"x": nearest_spot["x"], "y": nearest_spot["y"]}
    else:
        return {"message": "No free parking spot found"}


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
yolo_model = YOLO("static/best_3000_xl.pt")  # YOLOv8 모델
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
