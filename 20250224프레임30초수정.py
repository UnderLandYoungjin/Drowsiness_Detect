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
font_large = ImageFont.truetype(font_path, 20)
font_small = ImageFont.truetype(font_path, 5)

# 서버 URL 설정
upload_url = 'http://221.152.105.81:1200/upload'

# 눈 랜드마크 인덱스
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# YOLO 모델 초기화
seatbelt_model = YOLO('best_e_dt3000.pt')

# 녹화 관련 상수 설정
FPS = 20
BUFFER_SECONDS = 15  # 버퍼에 저장할 시간
RECORDING_SECONDS = 30  # 총 녹화 시간
BUFFER_SIZE = BUFFER_SECONDS * FPS
TOTAL_FRAMES = RECORDING_SECONDS * FPS

class FrameBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.recording_buffer = []
        self.is_recording = False
        self.start_time = None
        self.frame_count = 0

    def add_frame(self, visual_frame, original_frame):
        self.buffer.append((visual_frame.copy(), original_frame.copy()))
        if self.is_recording:
            self.recording_buffer.append((visual_frame.copy(), original_frame.copy()))
            self.frame_count += 1

    def start_recording(self):
        self.is_recording = True
        self.start_time = time.time()
        self.recording_buffer = list(self.buffer)  # 버퍼의 내용을 복사
        self.frame_count = len(self.recording_buffer)

    def should_stop_recording(self):
        return self.frame_count >= TOTAL_FRAMES

    def get_recording_frames(self):
        # 정확히 필요한 프레임 수만큼 반환
        return self.recording_buffer[:TOTAL_FRAMES]

    def clear_recording(self):
        self.recording_buffer = []
        self.is_recording = False
        self.start_time = None
        self.frame_count = 0

def eye_aspect_ratio(eye_points):
    def euclidean_distance(p1, p2):
        return np.linalg.norm(p1 - p2)
    ear_left = (euclidean_distance(eye_points[1], eye_points[5]) +
                euclidean_distance(eye_points[2], eye_points[4])) / (2.0 * euclidean_distance(eye_points[0], eye_points[3]))
    ear_right = (euclidean_distance(eye_points[7], eye_points[11]) +
                 euclidean_distance(eye_points[8], eye_points[10])) / (2.0 * euclidean_distance(eye_points[6], eye_points[9]))
    return (ear_left + ear_right) / 2

def save_and_send_video(frames, frame_size):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    visual_filename = f"/tmp/HPNT_01_V_{timestamp}.mp4"
    original_filename = f"/tmp/HPNT_01_O_{timestamp}.mp4"

    # 비디오 작성기 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_visual = cv2.VideoWriter(visual_filename, fourcc, FPS, frame_size)
    out_original = cv2.VideoWriter(original_filename, fourcc, FPS, frame_size)

    try:
        # 프레임 작성
        for frame_visual, frame_original in frames:
            out_visual.write(frame_visual)
            out_original.write(frame_original)

        # 파일 저장
        out_visual.release()
        out_original.release()

        # 파일 전송
        with open(visual_filename, 'rb') as f:
            requests.post(upload_url, files={'file': f})
        with open(original_filename, 'rb') as f:
            requests.post(upload_url, files={'file': f})

    finally:
        # 파일 정리
        if os.path.exists(visual_filename):
            os.remove(visual_filename)
        if os.path.exists(original_filename):
            os.remove(original_filename)

def main():
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS)  # FPS 설정

    # 프레임 크기 설정
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 상태 변수 초기화
    frame_buffer = FrameBuffer(BUFFER_SIZE)
    landmark_hist = deque(maxlen=600)
    drowsy_start_time = None
    sent_video = False
    cooldown_start_time = None

    # Face Mesh 초기화
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()  # 프레임 처리 시작 시간

                # 프레임 회전
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame_height, frame_width = frame.shape[:2]
                original_frame = frame.copy()

                # YOLO 모델로 안전벨트 감지
                yolo_results = seatbelt_model(frame)
                seatbelt_detected = False
                
                for result in yolo_results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    
                    for box, score in zip(boxes, scores):
                        if score < 0.50:
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        label = f"SEATBELT : {score:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)
                        cv2.putText(frame, label, (x1 + 5, y2 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)
                        seatbelt_detected = True

                if not seatbelt_detected:
                    cv2.putText(frame, "No Seatbelt", 
                               (frame_width // 2 - 50, frame_height // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # PIL 이미지로 변환
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)

                # 얼굴 감지 및 분석
                result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                drowsy_detected = False

                if cooldown_start_time and time.time() - cooldown_start_time >= 16:
                    cooldown_start_time = None
                    sent_video = False

                if result.multi_face_landmarks:
                    for face_landmarks in result.multi_face_landmarks:
                        # 눈 좌표 계산
                        left_eye_points = np.array([[face_landmarks.landmark[idx].x * frame_width,
                                                   face_landmarks.landmark[idx].y * frame_height]
                                                  for idx in LEFT_EYE_IDX])
                        right_eye_points = np.array([[face_landmarks.landmark[idx].x * frame_width,
                                                    face_landmarks.landmark[idx].y * frame_height]
                                                   for idx in RIGHT_EYE_IDX])

                        # EAR 계산
                        ear_value = eye_aspect_ratio(np.concatenate((left_eye_points, right_eye_points), axis=0))
                        ear_color = (255, 0, 0) if ear_value < 0.2 else (0, 255, 0)
                        draw.text((30, 30), f'EAR: {ear_value:.2f}', font=font_large, fill=ear_color)

                        # 얼굴 중심점 계산
                        face_center_x = int(face_landmarks.landmark[1].x * frame_width)
                        face_center_y = int(face_landmarks.landmark[1].y * frame_height)
                        landmark_hist.append((face_center_x, face_center_y))

                        # 평균 중심점 계산
                        avg_face_center_x = int(np.mean([point[0] for point in landmark_hist]))
                        avg_face_center_y = int(np.mean([point[1] for point in landmark_hist]))

                        # 거리 계산
                        distance_between_centers = np.sqrt((avg_face_center_x - face_center_x) ** 2 +
                                                         (avg_face_center_y - face_center_y) ** 2)
                        distance_color = (0, 255, 0) if distance_between_centers <= 40 else (255, 0, 0)
                        draw.text((face_center_x + 10, face_center_y + 10),
                                 f"Distance: {distance_between_centers:.2f}",
                                 font=font_large, fill=distance_color)

                        # 졸음 감지
                        if ear_value < 0.2 or distance_between_centers > 40:
                            drowsy_start_time = time.time() if drowsy_start_time is None else drowsy_start_time
                            if time.time() - drowsy_start_time >= 1:
                                drowsy_detected = True
                                draw.text((30, 60), "Drowsy!", font=font_large, fill=(255, 0, 0))
                        else:
                            drowsy_start_time = None

                        # 시각화 요소 추가
                        x_coords = [landmark.x * frame_width for landmark in face_landmarks.landmark]
                        y_coords = [landmark.y * frame_height for landmark in face_landmarks.landmark]
                        x_min, x_max = int(min(x_coords)), int(max(x_coords))
                        y_min, y_max = int(min(y_coords)), int(max(y_coords))
                        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=(0, 0, 255), width=3)

                        # 중심점 연결선 및 포인트 표시
                        draw.line([(face_center_x, face_center_y),
                                  (avg_face_center_x, avg_face_center_y)],
                                 fill=(255, 0, 0), width=2)
                        
                        draw.ellipse((face_center_x - 5, face_center_y - 5,
                                    face_center_x + 5, face_center_y + 5),
                                   fill=(0, 255, 255))
                        draw.ellipse((avg_face_center_x - 5, avg_face_center_y - 5,
                                    avg_face_center_x + 5, avg_face_center_y + 5),
                                   fill=(255, 255, 0))

                        # 눈 포인트 표시
                        all_eye_points = np.concatenate((left_eye_points, right_eye_points), axis=0)
                        for i, (x, y) in enumerate(all_eye_points):
                            label = f"p{i+1}"
                            draw.text((x + 5, y - 5), label, font=font_small, fill=(255, 165, 0))
                            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 165, 0))

                # 프레임 변환 및 저장
                frame_visual = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                
                # 프레임 버퍼에 저장
                frame_buffer.add_frame(frame_visual, original_frame)

                # 실시간 화면 표시
                cv2.imshow("Drowsiness Detection", frame_visual)

                # 졸음 감지 시 녹화 처리
                if drowsy_detected and not sent_video and not frame_buffer.is_recording:
                    frame_buffer.start_recording()

                # 녹화 중 프레임 수 체크
                if frame_buffer.is_recording and frame_buffer.should_stop_recording():
                    sent_video = True
                    frames = frame_buffer.get_recording_frames()
                    frame_buffer.clear_recording()
                    
                    # 비디오 저장 및 전송
                    save_and_send_video(frames, (frame_width, frame_height))
                    cooldown_start_time = time.time()

                # 프레임 처리 시간 조절을 위한 대기
                elapsed_time = time.time() - start_time
                wait_time = max(1./FPS - elapsed_time, 0.001)
                if wait_time > 0:
                    time.sleep(wait_time)

                # ESC 키로 종료
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        finally:
            # 자원 해제
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
