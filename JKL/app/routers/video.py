from fastapi import APIRouter, File, UploadFile, HTTPException, Response, Request
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import os
import uuid

from services.video_service import extract_preview_frame, process_video
from pydantic import BaseModel

class ParkingSpotRequest(BaseModel):
    video_id: str
    x: int
    y: int

video_router = APIRouter(prefix="/video", tags=["Video Processing"])
UPLOAD_DIR = Path("app/resources/videos")
DOWNLOAD_DIR = Path("app/resources/downloaded")  # 새로운 다운로드 디렉토리
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

clicked_points = {}
pending_videos = {}

@video_router.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    """ 사용자가 업로드한 영상을 저장 후, 1초 프레임을 제공 """
    video_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{video_id}.mp4"

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    preview_path = extract_preview_frame(file_path)

    if preview_path is None:
        raise HTTPException(status_code=500, detail="1초 프레임 추출 실패")

    pending_videos[video_id] = file_path

    return {
        "message": "영상 업로드 완료, 1초 프레임을 확인하고 클릭하세요",
        "preview_url": f"/video/preview/{video_id}",
        "video_id": video_id
    }

@video_router.post("/select_parking_spot/")
async def select_parking_spot(request: ParkingSpotRequest):
    """ 사용자가 클릭한 주차 좌표를 저장하고 분석 시작 """
    video_id = request.video_id
    x = request.x
    y = request.y

    if video_id not in pending_videos:
        raise HTTPException(status_code=400, detail="해당 영상이 처리 대기 중이 아닙니다.")

    clicked_points[video_id] = (x, y)
    video_path = pending_videos.pop(video_id)

    processed_path = process_video(video_path, video_id, clicked_points)
    download_path = DOWNLOAD_DIR / processed_path.name

    # ✅ 파일 이동이 실패하는 경우를 디버깅
    try:
        shutil.move(processed_path, download_path)
        print(f"✅ 처리된 영상이 이동됨: {download_path}")  # 성공 로그 추가
    except Exception as e:
        print(f"❌ 영상 이동 실패: {e}")  # 에러 로그 추가
        raise HTTPException(status_code=500, detail="영상 이동 실패")

    return {
        "message": "주차 위치 저장 완료, 영상 처리를 시작합니다.",
        "download_url": f"/video/download/{download_path.name}"
    }

@video_router.get("/download/{filename}")
def download_video(filename: str):
    """ 처리된 영상 다운로드 """
    file_path = DOWNLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="다운로드할 영상이 존재하지 않습니다.")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)

@video_router.get("/preview/{video_id}")
def preview_frame(video_id: str):
    """ 1초 프레임 제공 """
    preview_path = UPLOAD_DIR / f"{video_id}.jpg"
    
    if not preview_path.exists():
        print(f"❌ 프레임 파일 없음: {preview_path}")  # 로그 추가
        raise HTTPException(status_code=404, detail="프레임 이미지가 존재하지 않습니다.")
    
    print(f"✅ 프레임 제공: {preview_path}")  # 로그 추가
    return FileResponse(preview_path, media_type="image/jpeg")
