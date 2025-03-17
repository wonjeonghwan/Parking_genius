import os
import tkinter as tk
from tkinter import filedialog
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Tkinter 창 설정
root = tk.Tk()
root.withdraw()

# XML 및 이미지 파일이 있는 폴더 선택
xml_folder_path = filedialog.askdirectory(title='XML 파일이 있는 폴더를 선택하세요')
image_folder_path = filedialog.askdirectory(title='이미지 파일이 있는 폴더를 선택하세요')

# XML 파일 리스트 가져오기
xml_files = [f for f in os.listdir(xml_folder_path) if f.endswith('.xml')]

for xml_file in xml_files:
    xml_file_path = os.path.join(xml_folder_path, xml_file)
    image_name = os.path.splitext(xml_file)[0] + ".jpg"
    image_path = None

    # 이미지 파일 검색
    for img_root, _, img_files in os.walk(image_folder_path):
        if image_name in img_files:
            image_path = os.path.join(img_root, image_name)
            break

    if not image_path or not os.path.exists(image_path):
        print(f"이미지 파일이 존재하지 않습니다: {image_name}")
        continue

    # XML 파일 파싱
    tree = ET.parse(xml_file_path)
    xml_root = tree.getroot()

    fig, ax = plt.subplots()
    image_data = plt.imread(image_path)
    ax.imshow(image_data)

    for space in xml_root.findall('.//space'):
        occupied = space.get('occupied', '0')  # 점유 여부 (기본값 0)
        contour = space.find('contour')

        if contour is None:
            continue

        # 윤곽선 시각화
        points = [(float(point.get('x')), float(point.get('y'))) for point in contour.findall('point')]
        if points:
            color = 'blue' if occupied == '1' else 'red'
            polygon_patch = patches.Polygon(points, closed=True, edgecolor=color, facecolor='none', linewidth=1.5)
            ax.add_patch(polygon_patch)

    # 결과 저장 폴더 생성
    xml_base_folder = os.path.basename(xml_folder_path)
    result_folder = os.path.join(image_folder_path, 'result', xml_base_folder)
    os.makedirs(result_folder, exist_ok=True)
    result_image_path = os.path.join(result_folder, image_name)

    plt.axis('off')
    plt.savefig(result_image_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)

    print(f"결과가 저장되었습니다: {result_image_path}")
