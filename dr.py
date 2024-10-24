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

    # 오른쪽 눈 EAR 계산
    ear_right = (euclidean_distance(eye_points[7], eye_points[11]) + 
                 euclidean_distance(eye_points[8], eye_points[10])) / (2.0 * euclidean_distance(eye_points[6], eye_points[9]))

    return (ear_left + ear_right) / 2  # 두 눈의 평균 EAR 반환

# 카메라 초기화
cap = cv2.VideoCapture(0)

# 얼굴 랜드마크 좌표 저장용 (최근 10분 평균을 유지하기 위한 deque)
landmark_hist = deque(maxlen=600)  # 600개의 프레임(약 10분)을 저장할 deque

# 창 크기를 조정할 수 있도록 설정
cv2.namedWindow('Drowsiness Detection', cv2.WINDOW_NORMAL)

# 비디오 저장 관련 초기화
video_out = None
recording = False
anomaly_start_time = 0
anomaly_detected = False
filename_counter = 1
baseline_ear = None  # 초기 EAR 값을 저장할 변수

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
    start_time = None
    alert_time = None
    anomaly_message = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 좌우 반전 및 상하 반전 처리
        frame = cv2.flip(frame, 0)  # 상하 반전

        # 프레임을 RGB로 변환 (Mediapipe가 처리할 수 있도록)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipe Face Mesh 감지 수행
        result = face_mesh.process(frame_rgb)

        # 프레임의 해상도(픽셀) 가져오기
        frame_height, frame_width = frame.shape[:2]

        # 랜드마크 및 텍스트 처리
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # 얼굴 랜드마크 그리기
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

                # 최초 EAR 값을 기준으로 설정
                if baseline_ear is None:
                    baseline_ear = ear_value

                # EAR 값이 기준의 50% 이하일 때 졸음 상태로 판단
                if ear_value < baseline_ear * 0.5:
                    drowsiness_message = "Drowsiness Detected: Eyes Closed"
                    anomaly_message = drowsiness_message
                else:
                    drowsiness_message = ""

                face_center_x = int(face_landmarks.landmark[1].x * frame.shape[1])
                face_center_y = int(face_landmarks.landmark[1].y * frame.shape[0])

                landmark_hist.append((face_center_x, face_center_y))

                avg_face_center_x = int(np.mean([point[0] for point in landmark_hist]))
                avg_face_center_y = int(np.mean([point[1] for point in landmark_hist]))

                def euclidean_distance(x1, y1, x2, y2):
                    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                distance_between_centers = euclidean_distance(face_center_x, face_center_y, avg_face_center_x, avg_face_center_y)

                current_time = time.time()

                # EAR이 기준 대비 50% 이하이거나 얼굴 중앙점이 100픽셀 이상 나가고 2초 이상 유지될 때
                if (ear_value < baseline_ear * 0.5 or distance_between_centers > 100) and (start_time is None or current_time - start_time >= 2):
                    anomaly_detected = True
                    if not recording:
                        recording = True
                        anomaly_start_time = current_time
                        filename_counter += 1
                        output_filename = f"anomaly/am_{time.strftime('%Y%m%d_%H%M%S')}_{filename_counter:03d}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))
                    
                    # 비디오 프레임에 현재 상태 메시지 추가
                    anomaly_message = drowsiness_message if drowsiness_message else "Attention Loss: Look Forward!"
                    video_out.write(frame)

                else:
                    start_time = None
                    anomaly_detected = False

                if recording:
                    # 녹화되는 영상에 메시지 추가
                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(frame_pil)
                    draw.text((30, 70), anomaly_message, font=font, fill=(255, 0, 0))
                    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                    video_out.write(frame)

                if recording and current_time - anomaly_start_time > 10:
                    if not anomaly_detected:  # 이상 상황이 2초 이상 발생하지 않았으면 녹화 중단하고 파일 삭제
                        recording = False
                        video_out.release()
                        os.remove(output_filename)
                    else:  # 이상 상황이 2초 이상 지속되었으면 파일 유지
                        recording = False
                        video_out.release()

                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)

                draw.text((30, 30), f'EAR: {ear_value:.2f}', font=font, fill=(0, 255, 0))
                if drowsiness_message:
                    draw.text((30, 70), drowsiness_message, font=font, fill=(255, 0, 0))

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
