import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path

yolo_model = YOLO("C:\\Users\\user\\Documents\\GitHub\\test\\JKL\\app\\models\\best_3000_xl.pt")
tracker = DeepSort(max_age=30)

def extract_preview_frame(video_path: Path) -> Path:
    """ 1초 프레임 추출 후 저장 """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_pos = int(fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    ret, frame = cap.read()

    if not ret:
        return None
    
    frame = cv2.resize(frame, (960, 540))

    preview_path = video_path.with_suffix(".jpg")
    cv2.imwrite(str(preview_path), frame)
    cap.release()
    return preview_path

def process_video(video_path: Path, video_id: str, clicked_points: dict) -> Path:
    """ YOLO & DeepSORT 기반 주차 공간 분석 """
    cap = cv2.VideoCapture(str(video_path))
    width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    output_path = video_path.with_name(f"processed_{video_path.name}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_and_track(frame, video_id, clicked_points)
        out.write(frame)

    cap.release()
    out.release()
    return output_path

def detect_and_track(frame, video_id, clicked_points):
    """ YOLO 객체 탐지 및 사용자의 클릭 정보 반영 """
    results = yolo_model(frame)[0]
    bbox_dict = {}
    free_boxes = []

    # YOLO 바운딩 박스에 번호 부여
    for idx, result in enumerate(results.boxes, start=1):
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = float(result.conf[0])
        class_id = int(result.cls[0])
        class_name = yolo_model.names[class_id] if class_id in yolo_model.names else "Unknown"

        bbox_dict[idx] = {"bbox": (x1, y1, x2, y2), "label": class_name, "confidence": conf}

        if class_name == "free":
            free_boxes.append((x1, y1, x2, y2))

    # 사용자가 선택한 주차 공간 추천
    if video_id in clicked_points:
        click_x, click_y = clicked_points[video_id]
        closest_space, _ = find_nearest_parking_space(click_x, click_y, free_boxes)
        if closest_space:
            cv2.circle(frame, closest_space, 10, (0, 0, 255), -1)

    # 바운딩 박스와 ID 표시
    for idx, data in bbox_dict.items():
        x1, y1, x2, y2 = data["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"#{idx}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

def find_nearest_parking_space(click_x, click_y, free_boxes):
    """ 클릭한 지점과 가장 가까운 'free' 주차 공간 찾기 """
    min_distance = float("inf")
    closest_center = None

    for (x1, y1, x2, y2) in free_boxes:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        distance = np.linalg.norm(np.array([click_x, click_y]) - np.array([center_x, center_y]))

        if distance < min_distance:
            min_distance = distance
            closest_center = (center_x, center_y)

    return closest_center, min_distance