import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from collections import deque
import os

# Mediapipe 모듈 초기화
mp_face_mesh = mp.solutions.face_mesh

# 폴더 설정
input_folder = 'rec'
output_folder = os.path.join(input_folder, 'last_all_test')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# EAR 및 Cpoint 임계값 설정
EAR_THRESHOLD = 0.225
CPOINT_DISTANCE_THRESHOLD = 30
DROWSINESS_FRAME_THRESHOLD = 10

# 폰트 설정 (Linux 기본 폰트 경로)
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
small_font = ImageFont.truetype(font_path, 10)
large_font = ImageFont.truetype(font_path, 30)

# 눈 랜드마크 인덱스
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# EAR 계산 함수
def eye_aspect_ratio(eye_points):
    def euclidean_distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    ear_left = (euclidean_distance(eye_points[1], eye_points[5]) + 
                euclidean_distance(eye_points[2], eye_points[4])) / (2.0 * euclidean_distance(eye_points[0], eye_points[3]))
    return ear_left

# 입력 비디오 파일
video_file = "cam12.mp4"
input_path = os.path.join(input_folder, video_file)
output_path = os.path.join(output_folder, "processed_cam12.mp4")

cap = cv2.VideoCapture(input_path)
frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# EAR 및 거리 기록용
ear_values = []
distance_values = []
drowsy_frames = []
consecutive_drowsy_frames = 0

# 얼굴 중심 좌표 기록용 deque
landmark_hist = deque(maxlen=60000)

# Face Mesh 초기화
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 얼굴 랜드마크 감지
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)
        
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                left_eye_points = np.array([[face_landmarks.landmark[idx].x * frame_width, 
                                             face_landmarks.landmark[idx].y * frame_height] for idx in LEFT_EYE_IDX])
                right_eye_points = np.array([[face_landmarks.landmark[idx].x * frame_width, 
                                              face_landmarks.landmark[idx].y * frame_height] for idx in RIGHT_EYE_IDX])

                # EAR 계산 및 기록
                ear_value = eye_aspect_ratio(np.concatenate((left_eye_points, right_eye_points), axis=0))
                ear_values.append(ear_value)

                # 얼굴 중심점과 평균 중심점 계산
                face_center_x = int(face_landmarks.landmark[1].x * frame_width)
                face_center_y = int(face_landmarks.landmark[1].y * frame_height)
                landmark_hist.append((face_center_x, face_center_y))

                avg_face_center_x = int(np.mean([point[0] for point in landmark_hist]))
                avg_face_center_y = int(np.mean([point[1] for point in landmark_hist]))
                
                # Cpoint 거리 계산 및 기록
                cpoint_distance = np.sqrt((face_center_x - avg_face_center_x) ** 2 + (face_center_y - avg_face_center_y) ** 2)
                distance_values.append(cpoint_distance)

                # 텍스트 및 점 표시
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                draw.text((30, 30), f'EAR: {ear_value:.2f}', font=large_font, fill=(0, 255, 0))
                draw.text((30, 70), f'Cpoint Dist: {cpoint_distance:.1f}', font=large_font, fill=(0, 255, 0))

                # 각 랜드마크에 연두색 별표와 p1~p12 레이블 추가
                for i, (x, y) in enumerate(left_eye_points):
                    draw.text((x + 3, y - 10), f'p{i+1}', font=small_font, fill=(255, 255, 0))  # 노란색 레이블 추가
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 0))  # 연두색 점으로 표시
                for i, (x, y) in enumerate(right_eye_points):
                    draw.text((x + 3, y - 10), f'p{i+7}', font=small_font, fill=(255, 255, 0))  # 노란색 레이블 추가
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 0))  # 연두색 점으로 표시

                # Cpoint와 Avg Cpoint 점 및 텍스트 표시
                draw.ellipse((face_center_x - 2, face_center_y - 2, face_center_x + 2, face_center_y + 2), fill=(255, 140, 0))  # 주황색 Cpoint
                draw.text((face_center_x, face_center_y - 10), "Cpoint", font=small_font, fill=(255, 140, 0))  # 주황색 텍스트

                draw.ellipse((avg_face_center_x - 2, avg_face_center_y - 2, avg_face_center_x + 2, avg_face_center_y + 2), fill=(135, 206, 235))  # 하늘색 Avg Cpoint
                draw.text((avg_face_center_x, avg_face_center_y + 5), "Avg Cpoint", font=small_font, fill=(135, 206, 235))  # 하늘색 텍스트

                # Cpoint와 Avg Cpoint를 더 굵은 빨간색 선으로 연결
                draw.line([(face_center_x, face_center_y), (avg_face_center_x, avg_face_center_y)], fill=(255, 0, 0), width=2)

                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                # 졸음 판정
                if ear_value < EAR_THRESHOLD and cpoint_distance > CPOINT_DISTANCE_THRESHOLD:
                    consecutive_drowsy_frames += 1
                    if consecutive_drowsy_frames >= DROWSINESS_FRAME_THRESHOLD:
                        drowsy_frames.append(len(ear_values))
                        cv2.putText(frame, "Drowsiness Detected", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    consecutive_drowsy_frames = 0

        # 비디오 저장
        video_out.write(frame)

# 비디오 및 자원 해제
cap.release()
video_out.release()

# EAR 및 Cpoint 거리 그래프 생성
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# EAR 그래프
ax1.plot(ear_values, label="EAR", color="blue")
ax1.axhline(y=EAR_THRESHOLD, color="red", linestyle="--", label="EAR Threshold (0.225)")
below_threshold_ear = [i for i, ear in enumerate(ear_values) if ear < EAR_THRESHOLD]
ax1.scatter(below_threshold_ear, [ear_values[i] for i in below_threshold_ear], color="red", s=10, label="EAR Below Threshold")
ax1.set_ylabel("EAR")
ax1.legend()
ax1.set_title("EAR over Time")

# Cpoint Distance 그래프
ax2.plot(distance_values, label="Cpoint Distance", color="green")
ax2.axhline(y=CPOINT_DISTANCE_THRESHOLD, color="orange", linestyle="--", label="Distance Threshold (30)")
ax2.set_ylabel("Cpoint Distance")
ax2.set_xlabel("Frame")
ax2.legend()
ax2.set_title("Cpoint Distance over Time")

# 그래프 파일 저장
graph_path = os.path.join(output_folder, "ear_distance_drowsiness_graph.png")
plt.savefig(graph_path)
plt.close()

print(f"분석 완료: 결과 비디오와 그래프가 {output_folder} 폴더에 저장되었습니다.")
