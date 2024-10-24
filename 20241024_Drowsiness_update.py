import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from PIL import Image, ImageFont, ImageDraw
import time
import os

# Mediapipe 모듈 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 리눅스 시스템에 맞는 폰트 경로 설정
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # 리눅스 기본 폰트 경로
font = ImageFont.truetype(font_path, 30)

# 왼쪽 및 오른쪽 눈의 랜드마크 인덱스
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# EAR 계산 함수
def eye_aspect_ratio(eye_points):
    def euclidean_distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    # 왼쪽 눈 EAR 계산
    ear_left = (euclidean_distance(eye_points[1], eye_points[5]) + 
                euclidean_distance(eye_points[2], eye_points[4])) / (2.0 * euclidean_distance(eye_points[0], eye_points[3]))

    return ear_left

# 얼굴 중앙에서의 이동 계산 함수
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# 경고 메시지 추가 함수
def draw_warning(draw, message, position, font, fill=(255, 0, 0), outline=(0, 0, 0)):
    # 메시지 테두리 그리기
    x, y = position
    for offset in [-2, -1, 1, 2]:
        draw.text((x + offset, y), message, font=font, fill=outline)
        draw.text((x, y + offset), message, font=font, fill=outline)
    # 실제 메시지 그리기
    draw.text(position, message, font=font, fill=fill)

# 카메라 초기화
cap = cv2.VideoCapture(0)

# 얼굴 랜드마크 좌표 저장용 (최근 10분 평균을 유지하기 위한 deque)
landmark_hist = deque(maxlen=600)

# 비디오 저장 관련 초기화
video_out = None
recording = False
anomaly_start_time = 0
anomaly_detected = False
filename_counter = 1
baseline_ear = None

# anomaly 폴더 생성
if not os.path.exists('anomaly'):
    os.makedirs('anomaly')

# Face Mesh를 사용한 감지
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

        frame = cv2.flip(frame, 0)  # 상하 반전
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)

        frame_height, frame_width = frame.shape[:2]

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                left_eye_points = np.array([[face_landmarks.landmark[idx].x * frame.shape[1], 
                                             face_landmarks.landmark[idx].y * frame.shape[0]] for idx in LEFT_EYE_IDX])

                right_eye_points = np.array([[face_landmarks.landmark[idx].x * frame.shape[1], 
                                              face_landmarks.landmark[idx].y * frame.shape[0]] for idx in RIGHT_EYE_IDX])

                ear_value = eye_aspect_ratio(np.concatenate((left_eye_points, right_eye_points), axis=0))

                if baseline_ear is None:
                    baseline_ear = ear_value

                # EAR 값이 기준의 50% 이하일 때 졸음 상태로 판단
                if ear_value < baseline_ear * 0.5:
                    drowsiness_message = "Drowsiness Detected: Eyes Closed"
                    anomaly_message = drowsiness_message
                else:
                    drowsiness_message = ""

                # 얼굴 바운딩 박스 계산 (얼굴 랜드마크 중 최소/최대 좌표 찾기)
                x_min = min([int(landmark.x * frame_width) for landmark in face_landmarks.landmark])
                y_min = min([int(landmark.y * frame_height) for landmark in face_landmarks.landmark])
                x_max = max([int(landmark.x * frame_width) for landmark in face_landmarks.landmark])
                y_max = max([int(landmark.y * frame_height) for landmark in face_landmarks.landmark])

                # 바운딩 박스 그리기 (연두색)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                face_center_x = int(face_landmarks.landmark[1].x * frame.shape[1])
                face_center_y = int(face_landmarks.landmark[1].y * frame.shape[0])

                landmark_hist.append((face_center_x, face_center_y))

                avg_face_center_x = int(np.mean([point[0] for point in landmark_hist]))
                avg_face_center_y = int(np.mean([point[1] for point in landmark_hist]))

                distance_between_centers = euclidean_distance(face_center_x, face_center_y, avg_face_center_x, avg_face_center_y)

                current_time = time.time()

                if (ear_value < baseline_ear * 0.5 or distance_between_centers > 100) and (anomaly_start_time == 0 or current_time - anomaly_start_time >= 2):
                    anomaly_detected = True
                    if not recording:
                        recording = True
                        anomaly_start_time = current_time
                        filename_counter += 1
                        output_filename = f"anomaly/am_{time.strftime('%Y%m%d_%H%M%S')}_{filename_counter:03d}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

                    anomaly_message = drowsiness_message if drowsiness_message else "Attention Loss: Look Forward!"
                    video_out.write(frame)

                else:
                    anomaly_start_time = 0
                    anomaly_detected = False

                # 녹화 영상에 메시지 추가
                if recording:
                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(frame_pil)
                    draw_warning(draw, anomaly_message, (30, 70), font)
                    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                    video_out.write(frame)

                # EAR 값과 메시지 표시
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                draw.text((30, 30), f'EAR: {ear_value:.2f}', font=font, fill=(0, 255, 0))

                if drowsiness_message:
                    draw_warning(draw, drowsiness_message, (30, 70), font)

                draw.text((face_center_x + 10, face_center_y), f'Ccurrent: ({face_center_x}, {face_center_y})', font=font, fill=(255, 0, 120))
                draw.text((avg_face_center_x + 10, avg_face_center_y), f'Cavg: ({avg_face_center_x}, {avg_face_center_y})', font=font, fill=(0, 120, 0))

                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow('Drowsiness Detection', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

if video_out is not None:
    video_out.release()

cap.release()
cv2.destroyAllWindows()
