import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
import os
import requests
from PIL import Image, ImageDraw, ImageFont

# Mediapipe 설정
mp_face_mesh = mp.solutions.face_mesh

# EAR 임계값 및 Cpoint 설정
EAR_THRESHOLD = 0.225
CPOINT_THRESHOLD = 0.5  # 예시 값 (Cpoint 기준값은 실제 조건에 맞게 설정)
FPS = 20  # 초당 프레임 수 (웹캠 설정에 맞춰 조정)

# 졸음 탐지 이후 30초간 프레임 버퍼링 설정
POST_DROWSY_SECONDS = 30
buffer_size = POST_DROWSY_SECONDS * FPS
post_drowsy_buffer = []

# 서버 업로드 URL
upload_url = 'http://000.000.000.000:0000/upload'

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

# 영상 캡처 시작
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Face Mesh 초기화
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    drowsy_detected = False
    drowsy_start_time = None  # 졸음 시작 시간
    post_drowsy_recording = False  # 졸음 탐지 이후 30초 기록 활성화

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
                cpoint_value = np.random.rand()  # Cpoint 예시 값 (실제 Cpoint 계산으로 대체)

                # 얼굴 중앙 좌표 계산
                center_x, center_y = int(face_landmarks.landmark[1].x * frame_width), int(face_landmarks.landmark[1].y * frame_height)

                # 졸음 탐지 조건
                if ear_value < EAR_THRESHOLD and cpoint_value > CPOINT_THRESHOLD:
                    if not drowsy_detected:
                        drowsy_detected = True
                        drowsy_start_time = datetime.now()
                        post_drowsy_recording = True  # 졸음 탐지 후 기록 시작
                        print("졸음 탐지됨")

                # 졸음 탐지 이후 30초 기록 완료 여부 체크
                if post_drowsy_recording:
                    # 시각화 추가
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_frame)
                    
                    # 텍스트 및 그래픽 요소 추가
                    draw.text((10, 10), f"EAR: {ear_value:.2f}", fill=(0, 255, 0))
                    draw.text((10, 30), "Attention Loss: Look Forward!", fill=(255, 0, 0))
                    draw.text((center_x, center_y), f"Center: ({center_x}, {center_y})", fill=(0, 255, 0))
                    draw.rectangle([(center_x - 50, center_y - 50), (center_x + 50, center_y + 50)], outline=(0, 255, 0), width=2)

                    # OpenCV 형식으로 변환 후 버퍼에 추가
                    frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
                    post_drowsy_buffer.append(frame)

                    # 버퍼에 30초 분량이 채워지면 파일 저장 및 전송
                    if len(post_drowsy_buffer) >= buffer_size:
                        post_drowsy_recording = False  # 기록 종료
                        print("졸음 후 30초 기록 완료, 파일 저장 및 전송 시작")

                        # 파일 이름 설정
                        timestamp = drowsy_start_time.strftime('%Y%m%d%H%M%S')
                        filename = f"es_{timestamp}_postdrowsy.mp4"
                        filepath = os.path.join("/tmp", filename)

                        # 비디오 파일 저장
                        out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (frame_width, frame_height))
                        for buffered_frame in post_drowsy_buffer:
                            out.write(buffered_frame)
                        out.release()

                        # 파일 전송 (HTTP POST 요청)
                        with open(filepath, 'rb') as f:
                            files = {'file': f}
                            response = requests.post(upload_url, files=files)
                            if response.status_code == 200:
                                print(f"{filename} 파일 전송 완료")
                            else:
                                print(f"{filename} 파일 전송 실패")

                        # 전송 후 버퍼 초기화
                        post_drowsy_buffer = []
                        drowsy_detected = False

        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
