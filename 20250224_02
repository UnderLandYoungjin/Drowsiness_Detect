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

# 한글 폰트 찾기
def find_font():
    # 가능한 폰트 경로들
    font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
        "/usr/share/fonts/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold/NanumGothicBold.ttf",
        # Malgun Gothic (Windows 한글 폰트)
        "/usr/share/fonts/truetype/malgun/malgun.ttf",
        # 기본 폰트도 추가
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    ]
    
    for path in font_paths:
        try:
            return ImageFont.truetype(path, 20)
        except:
            continue
            
    # 모든 경로가 실패하면 PIL의 기본 폰트 사용
    print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    try:
        return ImageFont.load_default()
    except:
        print("폰트 로드 실패")
        exit(1)

# 폰트 설정 (크기 감소)
font_large = find_font()
font_size_large = 16  # 기존 20에서 16으로 감소
font_size_small = 4   # 기존 5에서 4로 감소

try:
    font_large = ImageFont.truetype(font_large.path, font_size_large)
    font_small = ImageFont.truetype(font_large.path, font_size_small)
except:
    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()

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

def draw_status_box(draw, frame_width, frame_height, drowsy_status, seatbelt_status, ear_value=0, distance=0, belt_conf=0, face_detected=False):
    # 박스 크기와 위치 설정 (높이 증가)
    box_width = 140
    box_height = 140  # 높이 추가 증가
    margin = 2
    x = margin
    y = margin
    
    # 배경 박스 그리기
    draw.rectangle([(x, y), (x + box_width, y + box_height)], 
                  fill=(0, 0, 0), outline=(255, 255, 255), width=2)
    
    # 상태 텍스트
    drowsy_text = "졸음여부"
    belt_text = "안전벨트"
    
    # 상태 값 (얼굴 미감지 상태 추가)
    if not face_detected:
        drowsy_value = "미감지"
        drowsy_color = (255, 165, 0)  # 주황색으로 표시
    else:
        drowsy_value = "졸음" if drowsy_status else "정상"
        drowsy_color = (255, 0, 0) if drowsy_status else (0, 255, 0)

    belt_value = "착용" if seatbelt_status else "미착용"
    belt_color = (0, 255, 0) if seatbelt_status else (255, 0, 0)
    
    # 눈감김 비율 계산 (얼굴 미감지시 N/A로 표시)
    if face_detected:
        eye_close_percent = max(0, min(100, (1 - ear_value / 0.3) * 100))
        attention_percent = 100-min(100, distance/2)
    else:
        eye_close_percent = 0
        attention_percent = 0
    
    # 텍스트 위치 (행간 축소)
    x_text = x + 10
    y_text1 = y + 10          # 상태
    y_text2 = y_text1 + 25    # 안전벨트 (간격 축소: 30 -> 25)
    y_text3 = y_text2 + 25    # 벨트 확률
    y_text4 = y_text3 + 25    # 눈감김 비율
    y_text5 = y_text4 + 25    # 전방주시
    
    # 텍스트 배경 (검은색 윤곽선 효과)
    for offset in [(-1,-1), (-1,1), (1,-1), (1,1)]:
        draw.text((x_text + offset[0], y_text1 + offset[1]), f"{drowsy_text}: {drowsy_value}", 
                  font=font_large, fill=(0, 0, 0))
        draw.text((x_text + offset[0], y_text2 + offset[1]), f"{belt_text}: {belt_value}", 
                  font=font_large, fill=(0, 0, 0))
        draw.text((x_text + offset[0], y_text3 + offset[1]), f"벨트확률: {belt_conf:.1f}%", 
                  font=font_large, fill=(0, 0, 0))
        
        if face_detected:
            draw.text((x_text + offset[0], y_text4 + offset[1]), f"눈감김: {eye_close_percent:.1f}%", 
                     font=font_large, fill=(0, 0, 0))
            draw.text((x_text + offset[0], y_text5 + offset[1]), f"전방주시: {attention_percent:.1f}%", 
                     font=font_large, fill=(0, 0, 0))
        else:
            draw.text((x_text + offset[0], y_text4 + offset[1]), "눈감김: N/A", 
                     font=font_large, fill=(0, 0, 0))
            draw.text((x_text + offset[0], y_text5 + offset[1]), "전방주시: N/A", 
                     font=font_large, fill=(0, 0, 0))
    
    # 메인 텍스트
    draw.text((x_text, y_text1), f"{drowsy_text}: {drowsy_value}", 
              font=font_large, fill=drowsy_color)
    draw.text((x_text, y_text2), f"{belt_text}: {belt_value}", 
              font=font_large, fill=belt_color)
    
    # 데이터 값의 색상 설정
    belt_prob_color = (0, 255, 0) if belt_conf >= 50 else (255, 0, 0)
    eye_color = (0, 255, 0) if eye_close_percent < 50 else (255, 0, 0)
    attention_color = (0, 255, 0) if attention_percent >= 60 else (255, 0, 0)
    
    draw.text((x_text, y_text3), f"벨트확률: {belt_conf:.1f}%", 
              font=font_large, fill=belt_prob_color)
              
    if face_detected:
        draw.text((x_text, y_text4), f"눈감김: {eye_close_percent:.1f}%", 
                  font=font_large, fill=eye_color)
        draw.text((x_text, y_text5), f"전방주시: {attention_percent:.1f}%", 
                  font=font_large, fill=attention_color)
    else:
        draw.text((x_text, y_text4), "눈감김: N/A", 
                  font=font_large, fill=(128, 128, 128))  # 회색으로 표시
        draw.text((x_text, y_text5), "전방주시: N/A", 
                  font=font_large, fill=(128, 128, 128))  # 회색으로 표시

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

                start_time = time.time()

                # 프레임 회전
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame_height, frame_width = frame.shape[:2]
                original_frame = frame.copy()

                # YOLO 모델로 안전벨트 감지
                yolo_results = seatbelt_model(frame)
                seatbelt_detected = False
                belt_confidence = 0.0
                
                for result in yolo_results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    
                    for box, score in zip(boxes, scores):
                        if score < 0.50:
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)
                        seatbelt_detected = True
                        belt_confidence = score * 100  # 확률을 퍼센트로 변환

                # PIL 이미지로 변환
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)

                # 얼굴 감지 및 분석
                result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                drowsy_detected = False
                ear_value = 0
                face_detected = False  # 얼굴 감지 상태 추가
                distance_between_centers = 0

                if cooldown_start_time and time.time() - cooldown_start_time >= 16:
                    cooldown_start_time = None
                    sent_video = False

                if result.multi_face_landmarks:
                    face_detected = True  # 얼굴이 감지됨
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

                        # 졸음 감지
                        if ear_value < 0.2 or distance_between_centers > 40:
                            drowsy_start_time = time.time() if drowsy_start_time is None else drowsy_start_time
                            if time.time() - drowsy_start_time >= 1:
                                drowsy_detected = True
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

                # 상태 박스 그리기 (항상 표시)
                current_distance = distance_between_centers if 'distance_between_centers' in locals() else 0
                current_ear = ear_value if ear_value else 0
                current_belt_conf = belt_confidence if 'belt_confidence' in locals() else 0
                draw_status_box(draw, frame_width, frame_height, drowsy_detected, seatbelt_detected, 
                              current_ear, current_distance, current_belt_conf, face_detected)

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
