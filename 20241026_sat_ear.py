import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import os

# Mediapipe 모듈 초기화
mp_face_mesh = mp.solutions.face_mesh

# 폴더 설정
input_folder = 'rec'
output_folder = os.path.join(input_folder, 'last_test')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# EAR 임계값 설정
EAR_THRESHOLD = 0.225
DROWSINESS_FRAME_THRESHOLD = 10

# 폰트 설정 (Linux 기본 폰트 경로)
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
small_font = ImageFont.truetype(font_path, 10)  # 작은 글씨 (p1~p12 레이블용)
large_font = ImageFont.truetype(font_path, 30)  # 큰 글씨 (EAR 값 표시용)

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
video_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# EAR 기록용
ear_values = []
drowsy_frames = []
consecutive_drowsy_frames = 0

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
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)
        frame_height, frame_width = frame.shape[:2]
        
        # 얼굴 랜드마크 확인
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                left_eye_points = np.array([[face_landmarks.landmark[idx].x * frame_width, 
                                             face_landmarks.landmark[idx].y * frame_height] for idx in LEFT_EYE_IDX])
                right_eye_points = np.array([[face_landmarks.landmark[idx].x * frame_width, 
                                              face_landmarks.landmark[idx].y * frame_height] for idx in RIGHT_EYE_IDX])

                ear_value = eye_aspect_ratio(np.concatenate((left_eye_points, right_eye_points), axis=0))
                ear_values.append(ear_value)

                # 랜드마크 표시 및 레이블 추가
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                
                # 각 랜드마크에 연두색 별표와 p1~p12 레이블 추가
                for i, (x, y) in enumerate(left_eye_points):
                    cv2.drawMarker(frame, (int(x), int(y)), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)  # 연두색 별표
                    draw.text((x + 3, y - 10), f'p{i+1}', font=small_font, fill=(255, 255, 0))  # 노란색 작은 레이블 추가
                for i, (x, y) in enumerate(right_eye_points):
                    cv2.drawMarker(frame, (int(x), int(y)), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2)  # 연두색 별표
                    draw.text((x + 3, y - 10), f'p{i+7}', font=small_font, fill=(255, 255, 0))  # 노란색 작은 레이블 추가

                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                # 졸음 판정: EAR < 0.21 상태가 연속 10 프레임 이상일 때
                if ear_value < EAR_THRESHOLD:
                    consecutive_drowsy_frames += 1
                    if consecutive_drowsy_frames >= DROWSINESS_FRAME_THRESHOLD:
                        drowsy_frames.append(len(ear_values))
                        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(frame_pil)
                        draw.text((30, 70), "Drowsiness Detected", font=large_font, fill=(255, 0, 0))
                        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                else:
                    consecutive_drowsy_frames = 0  # 졸음 상태가 아닌 경우 초기화

                # EAR 값 표시 (큰 글씨로)
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                draw.text((30, 30), f'EAR: {ear_value:.2f}', font=large_font, fill=(0, 255, 0))
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                
        video_out.write(frame)

# 비디오 및 자원 해제
cap.release()
video_out.release()

# EAR 그래프 생성
plt.figure(figsize=(10, 5))
plt.plot(ear_values, color="blue", label="EAR")
plt.scatter(drowsy_frames, [ear_values[i-1] for i in drowsy_frames], color="red", label="Drowsy EAR")
plt.axhline(y=EAR_THRESHOLD, color="red", linestyle="--", label="Drowsiness Threshold (0.21)")
plt.title("EAR Analysis with Drowsiness Detection")
plt.xlabel("Frame")
plt.ylabel("EAR")
plt.legend()
graph_path = os.path.join(output_folder, "ear_drowsiness_graph.png")
plt.savefig(graph_path)
plt.close()

print(f"분석 완료: 결과가 {output_folder} 폴더에 저장되었습니다.")
