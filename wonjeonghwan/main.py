import cv2
import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

app = FastAPI()

# YOLO 11x 모델 및 DeepSORT 초기화 (사용자가 지정할 모델 사용)
yolo_model = YOLO('C:/test/wonjeonghwan/best_3000_xl.pt')
tracker = DeepSort(max_age=30)  # DeepSORT 추적 모델

# 업로드된 파일 저장 경로
UPLOAD_DIR = Path("C:/test/RAW")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    """
    사용자가 업로드한 영상을 저장 후 분석 시작
    """
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}.mp4"
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # YOLO + DeepSORT 적용 후 처리된 파일 경로
    processed_path = process_video(file_path)

    return {
        "message": "영상 업로드 및 분석 완료",
        "stream_url": f"/stream_video/{processed_path.name}",
        "download_url": f"/download_video/{processed_path.name}"
    }


def process_video(video_path: Path) -> Path:
    """
    YOLO 11x + DeepSORT로 주차 공간 분석 후 결과 영상을 저장
    """
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = UPLOAD_DIR / f"processed_{video_path.name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_and_track(frame)
        out.write(frame)

    cap.release()
    out.release()
    return output_path


def detect_and_track(frame):
    results = yolo_model(frame)[0]  # YOLO 11 탐지 실행
    detections = []

    # YOLO 결과에서 박스와 신뢰도 추출
    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # 바운딩 박스 좌표
        conf = float(result.conf[0])                # 신뢰도

        # DeepSORT가 기대하는 형식으로 저장
        detections.append([x1, y1, x2, y2, conf])

    # DeepSORT 업데이트 (YOLO로 얻은 차량 박스 기반)
    tracks = tracker.update_tracks(detections, frame=frame)

    # 추적 결과 박스와 ID 표시
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  # (left, top, right, bottom)

        # 박스 그리기
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)

        # 트랙 ID 표시
        cv2.putText(frame, f"ID: {track_id}", (int(ltrb[0]), int(ltrb[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

@app.get("/stream_video/{filename}")
def stream_video(filename: str):
    """
    분석된 영상 스트리밍
    """
    file_path = UPLOAD_DIR / filename
    return FileResponse(file_path, media_type="video/mp4")


@app.get("/download_video/{filename}")
def download_video(filename: str):
    """
    분석된 영상 다운로드
    """
    file_path = UPLOAD_DIR / filename
    return FileResponse(file_path, media_type="video/mp4", filename=filename)


@app.get("/", response_class=HTMLResponse)
def main_page():
    """
    스트리밍 + 다운로드 버튼이 포함된 HTML 페이지 반환
    """
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>주차 공간 분석</title>
        <style>
            body {{ text-align: center; margin-top: 50px; }}
            video {{ width: 80%; max-width: 960px; margin-top: 20px; }}
            .controls {{ margin-top: 20px; }}
        </style>
    </head>
    <body>

    <h2>주차 공간 분석</h2>

    <form action="/upload_video/" method="post" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <button type="submit">업로드 및 분석 시작</button>
    </form>

    <br>

    <h3>분석된 영상 스트리밍</h3>
    <video controls id="videoPlayer">
        <source id="videoSource" src="" type="video/mp4">
        브라우저가 video 태그를 지원하지 않습니다.
    </video>

    <br>

    <a id="downloadLink" href="" download>
        <button>MP4 파일로 다운받기</button>
    </a>

    <script>
        document.querySelector("form").onsubmit = async (e) => {{
            e.preventDefault();
            const formData = new FormData(e.target);
            const response = await fetch("/upload_video/", {{
                method: "POST",
                body: formData
            }});
            const result = await response.json();
            
            document.getElementById("videoSource").src = result.stream_url;
            document.getElementById("videoPlayer").load();
            document.getElementById("downloadLink").href = result.download_url;
        }};
    </script>

    </body>
    </html>
    """
    return HTMLResponse(content=html_content)