import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import os
from datetime import datetime
import requests
import time
from collections import deque
from ultralytics import YOLO

# Mediapipe 모듈 초기화
mp_face_mesh = mp.solutions.face_mesh

# 리눅스 시스템에 맞는 폰트 경로 설정
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font_large = ImageFont.truetype(font_path, 20)  # EAR와 거리 텍스트 크기
font_small = ImageFont.truetype(font_path, 5)  # p1~p12 포인트 텍스트 크기

# 서버 URL 설정
upload_url = 'http://00000000000000:1200/upload'

# 왼쪽 및 오른쪽 눈의 랜드마크 인덱스
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# YOLO 모델 초기화 (안전벨트 탐지용)
seatbelt_model = YOLO('best_e_dt3000.pt')

# EAR 계산 함수
def eye_aspect_ratio(eye_points):
    def euclidean_distance(p1, p2):
        return np.linalg.norm(p1 - p2)
    ear_left = (euclidean_distance(eye_points[1], eye_points[5]) +
                euclidean_distance(eye_points[2], eye_points[4])) / (2.0 * euclidean_distance(eye_points[0], eye_points[3]))
    ear_right = (euclidean_distance(eye_points[7], eye_points[11]) +
                 euclidean_distance(eye_points[8], eye_points[10])) / (2.0 * euclidean_distance(eye_points[6], eye_points[9]))
    return (ear_left + ear_right) / 2

# 카메라 초기화
cap = cv2.VideoCapture(0)

# 비디오 파일 초기화
FPS = 20
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc 변수 정의 (비디오 코덱 설정)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 두 개의 출력 파일 이름 설정
visual_output_filename = "current_recording_visual.mp4"
original_output_filename = "current_recording_original.mp4"

# 졸음 탐지 상태 및 타이머 초기화
drowsy_start_time = None
sent_video = False  # 영상 전송 여부 플래그
cooldown_start_time = None  # 16초 대기 시간 플래그
recording_additional_15_seconds = False  # 추가 녹화 상태 플래그
additional_recording_start_time = None  # 추가 녹화 시작 시간

# 얼굴 중앙 좌표 히스토리
landmark_hist = deque(maxlen=600)  # 약 10분간 중앙 좌표 저장

# ** 프레임 버퍼 추가: 졸음 감지 이전 15초 프레임 저장 **
pre_drowsy_buffer = deque(maxlen=15 * FPS)  # 15초 동안의 프레임 저장

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

        # 상하 반전 및 RGB 변환 0은 상하반전 1은 좌우반전
        frame = cv2.flip(frame, 0)
        
        # 원본 프레임을 그대로 기록 (시각 정보 없이)
        original_frame = frame.copy()

        # YOLO 모델로 안전벨트 감지 - OpenCV 프레임 사용
        yolo_results = seatbelt_model(frame)
        seatbelt_detected = False
        for result in yolo_results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            # 안전벨트 탐지 결과에 바운딩 박스와 텍스트를 프레임에 추가
            for box, score, cls in zip(boxes, scores, classes):
                if score < 0.55:  # 신뢰도 55% 미만 필터링
                    continue
                x1, y1, x2, y2 = map(int, box)
                label = f"SEATBELT : {score:.2f}"

                # 바운딩 박스를 프레임에 직접 그려줌 (연두색)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)  # 연두색 바운딩 박스
                cv2.putText(frame, label, (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)
                seatbelt_detected = True  # 안전벨트 감지 플래그 설정

        # 안전벨트가 감지되지 않은 경우 "No Seatbelt" 경고 텍스트 추가
        if not seatbelt_detected:
            cv2.putText(frame, "No Seatbelt", (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # YOLO 처리 후에 시각화를 위해 OpenCV 프레임을 PIL 이미지로 변환
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # 얼굴 인식 및 졸음 감지
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        drowsy_detected = False  # 졸음 탐지 초기화

        # 쿨다운이 끝났다면 `sent_video`를 초기화
        if cooldown_start_time and time.time() - cooldown_start_time >= 16:
            cooldown_start_time = None
            sent_video = False

        # 얼굴 랜드마크가 감지된 경우 EAR 계산 및 시각 정보 추가
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                left_eye_points = np.array([[face_landmarks.landmark[idx].x * frame_width, 
                                             face_landmarks.landmark[idx].y * frame_height] for idx in LEFT_EYE_IDX])
                right_eye_points = np.array([[face_landmarks.landmark[idx].x * frame_width, 
                                              face_landmarks.landmark[idx].y * frame_height] for idx in RIGHT_EYE_IDX])

                ear_value = eye_aspect_ratio(np.concatenate((left_eye_points, right_eye_points), axis=0))

                # EAR 색상 설정 및 텍스트 표시
                ear_color = (255, 0, 0) if ear_value < 0.2 else (0, 255, 0)
                draw.text((30, 30), f'EAR: {ear_value:.2f}', font=font_large, fill=ear_color)

                # 얼굴의 중앙 좌표 (Cpoint) 계산
                face_center_x = int(face_landmarks.landmark[1].x * frame_width)
                face_center_y = int(face_landmarks.landmark[1].y * frame_height)
                landmark_hist.append((face_center_x, face_center_y))

                # 최근 중앙 좌표의 평균 좌표 계산 (C Avg Point)
                avg_face_center_x = int(np.mean([point[0] for point in landmark_hist]))
                avg_face_center_y = int(np.mean([point[1] for point in landmark_hist]))

                # 중앙 좌표 간 거리 계산 및 표시
                distance_between_centers = np.sqrt((avg_face_center_x - face_center_x) ** 2 +
                                                   (avg_face_center_y - face_center_y) ** 2)
                distance_color = (0, 255, 0) if distance_between_centers <= 40 else (255, 0, 0)
                draw.text((face_center_x + 10, face_center_y + 10), f"Distance: {distance_between_centers:.2f}", font=font_large, fill=distance_color)

                # 졸음 탐지 기준: EAR < 0.2 및 위치 이탈이 40픽셀 초과인 경우
                if ear_value < 0.2 and distance_between_centers > 40:
                    drowsy_start_time = time.time() if drowsy_start_time is None else drowsy_start_time
                    if time.time() - drowsy_start_time >= 1:
                        drowsy_detected = True
                        draw.text((30, 60), "Drowsy!", font=font_large, fill=(255, 0, 0))
                else:
                    drowsy_start_time = None

                # 얼굴 바운딩 박스 계산 및 그리기 (파란색)
                x_coords = [landmark.x * frame_width for landmark in face_landmarks.landmark]
                y_coords = [landmark.y * frame_height for landmark in face_landmarks.landmark]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=(0, 0, 255), width=3)

                # Cpoint와 C Avg Point 사이의 빨간색 선분 그리기
                draw.line([(face_center_x, face_center_y), (avg_face_center_x, avg_face_center_y)], fill=(255, 0, 0), width=2)
                
                # Cpoint와 C Avg Point에 표시
                draw.ellipse((face_center_x - 5, face_center_y - 5, face_center_x + 5, face_center_y + 5), fill=(0, 255, 255))
                draw.ellipse((avg_face_center_x - 5, avg_face_center_y - 5, avg_face_center_x + 5, avg_face_center_y + 5), fill=(255, 255, 0))

                # p1~p12 포인트 표시
                all_eye_points = np.concatenate((left_eye_points, right_eye_points), axis=0)
                for i, (x, y) in enumerate(all_eye_points):
                    label = f"p{i+1}"
                    draw.text((x + 5, y - 5), label, font=font_small, fill=(255, 165, 0))
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 165, 0))

        # 시각 정보가 추가된 PIL 이미지를 OpenCV 형식으로 변환하여 최종 시각화 프레임 생성
        frame_visual = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # ** 프레임 버퍼에 시각화 및 원본 프레임 저장 **
        pre_drowsy_buffer.append((frame_visual, original_frame))

        # 졸음이 감지되었고, 전송되지 않은 경우에만 추가 녹화 시작
        if drowsy_detected and not sent_video:
            sent_video = True
            recording_additional_15_seconds = True
            additional_recording_start_time = time.time()

            # 비디오 작성기 설정
            out_visual = cv2.VideoWriter(visual_output_filename, fourcc, FPS, (frame_width, frame_height))
            out_original = cv2.VideoWriter(original_output_filename, fourcc, FPS, (frame_width, frame_height))

            # ** 버퍼에 저장된 프레임 기록 (졸음 감지 이전 15초) **
            for buffered_frame_visual, buffered_frame_original in pre_drowsy_buffer:
                out_visual.write(buffered_frame_visual)
                out_original.write(buffered_frame_original)

        # 추가 15초 녹화가 활성화된 경우
        if recording_additional_15_seconds:
            # 시각 정보가 추가된 프레임 및 원본 프레임 기록
            out_visual.write(frame_visual)  # 시각화된 프레임 기록
            out_original.write(original_frame)  # 원본 프레임 기록

            # 15초 녹화 후 파일 저장 및 전송
            if time.time() - additional_recording_start_time >= 15:
                recording_additional_15_seconds = False
                out_visual.release()
                out_original.release()

                # 타임스탬프 설정 및 파일 이름 지정
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                visual_filename = f"/tmp/HOME_VISUAL_{timestamp}.mp4"
                original_filename = f"/tmp/HOME_ORIGINAL_{timestamp}.mp4"
                
                # 파일 이름 변경 및 전송
                os.rename(visual_output_filename, visual_filename)
                os.rename(original_output_filename, original_filename)

                # 파일 전송 (시각 정보 포함 파일)
                with open(visual_filename, 'rb') as f:
                    files = {'file': f}
                    requests.post(upload_url, files=files)
                
                # 파일 전송 (시각 정보 미포함 원본 파일)
                with open(original_filename, 'rb') as f:
                    files = {'file': f}
                    requests.post(upload_url, files=files)

                # 전송 후 파일 삭제
                os.remove(visual_filename)
                os.remove(original_filename)

                # 16초 쿨다운 시작
                cooldown_start_time = time.time()

        # ESC 키 입력시 종료
        if cv2.waitKey(5) & 0xFF == 27:
            break

# 캡처 객체 해제
cap.release()
cv2.destroyAllWindows()
