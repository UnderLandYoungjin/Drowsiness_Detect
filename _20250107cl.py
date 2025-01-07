import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import os
from datetime import datetime
import requests
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List, Optional
from ultralytics import YOLO

@dataclass
class Config:
    """설정값들을 관리하는 클래스"""
    UPLOAD_URL: str = 'http://221.152.105.81:1200/upload'
    FPS: int = 20
    FONT_PATH: str = "C:/Windows/Fonts/arial.ttf"
    TEMP_FOLDER: str = "C:/tmp"
    EAR_THRESHOLD: float = 0.2
    DROWSY_TIME_THRESHOLD: float = 1.0
    DISTANCE_THRESHOLD: float = 40.0
    SEATBELT_CONFIDENCE_THRESHOLD: float = 0.55
    RECORDING_TIME: float = 15.0
    COOLDOWN_TIME: float = 16.0
    LEFT_EYE_IDX: List[int] = None
    RIGHT_EYE_IDX: List[int] = None

    def __post_init__(self):
        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
        if not os.path.exists(self.TEMP_FOLDER):
            os.makedirs(self.TEMP_FOLDER)

class VideoRecorder:
    """비디오 녹화를 관리하는 클래스"""
    def __init__(self, frame_width: int, frame_height: int, config: Config):
        self.config = config
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.initialize_writers()

    def initialize_writers(self):
        """비디오 작성기 초기화"""
        self.visual_output_filename = "current_recording_visual.mp4"
        self.original_output_filename = "current_recording_original.mp4"
        self.out_visual = cv2.VideoWriter(
            self.visual_output_filename, self.fourcc, 
            self.config.FPS, (self.frame_width, self.frame_height)
        )
        self.out_original = cv2.VideoWriter(
            self.original_output_filename, self.fourcc, 
            self.config.FPS, (self.frame_width, self.frame_height)
        )

    def write_frames(self, frame_visual: np.ndarray, frame_original: np.ndarray):
        """프레임 기록"""
        self.out_visual.write(frame_visual)
        self.out_original.write(frame_original)

    def save_and_upload(self, timestamp: str):
        """녹화 파일 저장 및 업로드"""
        self.out_visual.release()
        self.out_original.release()

        visual_filename = os.path.join(self.config.TEMP_FOLDER, f"P_V_{timestamp}.mp4")
        original_filename = os.path.join(self.config.TEMP_FOLDER, f"P_O_{timestamp}.mp4")

        if os.path.exists(self.visual_output_filename) and os.path.exists(self.original_output_filename):
            os.rename(self.visual_output_filename, visual_filename)
            os.rename(self.original_output_filename, original_filename)
            
            # 파일 전송
            for filename in [visual_filename, original_filename]:
                with open(filename, 'rb') as f:
                    files = {'file': f}
                    requests.post(self.config.UPLOAD_URL, files=files)
                os.remove(filename)

            self.initialize_writers()
            return True
        return False

class DrowsinessDetector:
    """졸음 감지를 관리하는 클래스"""
    def __init__(self, config: Config):
        self.config = config
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmark_hist = deque(maxlen=600)
        self.font_large = ImageFont.truetype(config.FONT_PATH, 20)
        self.font_small = ImageFont.truetype(config.FONT_PATH, 5)

    def calculate_ear(self, eye_points: np.ndarray) -> float:
        """눈 종횡비(EAR) 계산"""
        def euclidean_distance(p1, p2):
            return np.linalg.norm(p1 - p2)
            
        p1, p2, p3, p4, p5, p6 = eye_points[:6]
        p7, p8, p9, p10, p11, p12 = eye_points[6:]
        
        ear_left = (euclidean_distance(p2, p6) + euclidean_distance(p3, p5)) / (2.0 * euclidean_distance(p1, p4))
        ear_right = (euclidean_distance(p8, p12) + euclidean_distance(p9, p11)) / (2.0 * euclidean_distance(p7, p10))
        
        return (ear_left + ear_right) / 2

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """프레임 처리 및 졸음 감지"""
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)
        frame_pil = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(frame_pil)
        drowsy_detected = False

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                left_eye_points = np.array([[face_landmarks.landmark[idx].x * frame.shape[1], 
                                           face_landmarks.landmark[idx].y * frame.shape[0]] 
                                           for idx in self.config.LEFT_EYE_IDX])
                right_eye_points = np.array([[face_landmarks.landmark[idx].x * frame.shape[1], 
                                            face_landmarks.landmark[idx].y * frame.shape[0]] 
                                            for idx in self.config.RIGHT_EYE_IDX])

                ear_value = self.calculate_ear(np.concatenate((left_eye_points, right_eye_points), axis=0))
                drowsy_detected = ear_value < self.config.EAR_THRESHOLD

                self._draw_facial_features(draw, frame.shape, face_landmarks, ear_value, left_eye_points, right_eye_points)

        return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR), drowsy_detected

    def _draw_facial_features(self, draw: ImageDraw, frame_shape: Tuple, 
                            face_landmarks: any,  # mediapipe landmarks
                            ear_value: float, left_eye_points: np.ndarray, right_eye_points: np.ndarray):
        """얼굴 특징 시각화"""
        # EAR 표시
        ear_color = (255, 0, 0) if ear_value < self.config.EAR_THRESHOLD else (0, 255, 0)
        draw.text((30, 30), f'EAR: {ear_value:.2f}', font=self.font_large, fill=ear_color)

        # 얼굴 중심점 계산 및 표시
        face_center = self._calculate_face_center(face_landmarks, frame_shape)
        self.landmark_hist.append(face_center)
        avg_face_center = self._calculate_average_face_center()
        
        # 거리 계산 및 표시
        distance = np.linalg.norm(np.array(face_center) - np.array(avg_face_center))
        self._draw_distance_info(draw, face_center, distance)

        # 기타 시각적 요소 표시
        self._draw_face_box(draw, face_landmarks, frame_shape)
        self._draw_center_line(draw, face_center, avg_face_center)
        self._draw_eye_points(draw, np.concatenate((left_eye_points, right_eye_points), axis=0))

    def _calculate_face_center(self, face_landmarks, frame_shape: Tuple) -> Tuple[int, int]:
        """얼굴 중심점 계산"""
        x = int(face_landmarks.landmark[1].x * frame_shape[1])
        y = int(face_landmarks.landmark[1].y * frame_shape[0])
        return (x, y)

    def _calculate_average_face_center(self) -> Tuple[int, int]:
        """평균 얼굴 중심점 계산"""
        x = int(np.mean([point[0] for point in self.landmark_hist]))
        y = int(np.mean([point[1] for point in self.landmark_hist]))
        return (x, y)

    def _draw_distance_info(self, draw: ImageDraw, face_center: Tuple[int, int], distance: float):
        """거리 정보 표시"""
        color = (0, 255, 0) if distance <= self.config.DISTANCE_THRESHOLD else (255, 0, 0)
        draw.text((face_center[0] + 10, face_center[1] + 10), 
                 f"Distance: {distance:.2f}", font=self.font_large, fill=color)

    def _draw_face_box(self, draw: ImageDraw, face_landmarks, frame_shape: Tuple):
        """얼굴 박스 표시"""
        x_coords = [landmark.x * frame_shape[1] for landmark in face_landmarks.landmark]
        y_coords = [landmark.y * frame_shape[0] for landmark in face_landmarks.landmark]
        box = [int(min(x_coords)), int(min(y_coords)), 
               int(max(x_coords)), int(max(y_coords))]
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], 
                      outline=(0, 0, 255), width=3)

    def _draw_center_line(self, draw: ImageDraw, face_center: Tuple[int, int], 
                         avg_face_center: Tuple[int, int]):
        """중심선 표시"""
        draw.line([face_center, avg_face_center], fill=(255, 0, 0), width=2)
        for center in [face_center, avg_face_center]:
            draw.ellipse((center[0] - 5, center[1] - 5, 
                         center[0] + 5, center[1] + 5), 
                        fill=(0, 255, 255))

    def _draw_eye_points(self, draw: ImageDraw, eye_points: np.ndarray):
        """눈 포인트 표시"""
        for i, (x, y) in enumerate(eye_points):
            label = f"p{i+1}"
            draw.text((x + 5, y - 5), label, font=self.font_small, fill=(255, 165, 0))
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 165, 0))

class SeatbeltDetector:
    """안전벨트 감지를 관리하는 클래스"""
    def __init__(self, model_path: str, config: Config):
        self.model = YOLO(model_path)
        self.config = config

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """안전벨트 감지 수행"""
        results = self.model(frame)
        seatbelt_detected = False

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            
            for box, score in zip(boxes, scores):
                if score < self.config.SEATBELT_CONFIDENCE_THRESHOLD:
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                label = f"SEAT_BELT : {score:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)
                cv2.putText(frame, label, (x1 + 5, y2 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)
                seatbelt_detected = True

        return frame, seatbelt_detected

def main():
    """메인 함수"""
    
    config = Config()
    cap = cv2.VideoCapture(0)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_recorder = VideoRecorder(frame_width, frame_height, config)
    drowsiness_detector = DrowsinessDetector(config)
    seatbelt_detector = SeatbeltDetector('best_e_dt3000.pt', config)

    drowsy_start_time = None
    sent_video = False
    cooldown_start_time = None
    recording_additional_15_seconds = False
    additional_recording_start_time = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 안전벨트 감지
            frame, seatbelt_detected = seatbelt_detector.detect(frame)
            
            # 졸음 감지
            frame_visual, drowsy_detected = drowsiness_detector.process_frame(frame)

            # 쿨다운 체크
            if cooldown_start_time and time.time() - cooldown_start_time >= config.COOLDOWN_TIME:
                cooldown_start_time = None
                sent_video = False

            # 졸음 감지 로직
            if drowsy_detected:
                if drowsy_start_time is None:
                    drowsy_start_time = time.time()
                elif time.time() - drowsy_start_time >= config.DROWSY_TIME_THRESHOLD and not sent_video:
                    sent_video = True
                    recording_additional_15_seconds = True
                    additional_recording_start_time = time.time()
                    print("졸음 감지됨: 추가 15초 녹화 시작")
            else:
                drowsy_start_time = None

            # 프레임 기록
            video_recorder.write_frames(frame_visual, frame)

            # 추가 녹화 처리
            if recording_additional_15_seconds:
                if time.time() - additional_recording_start_time >= config.RECORDING_TIME:
                    recording_additional_15_seconds = False
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    if video_recorder.save_and_upload(timestamp):
                        cooldown_start_time = time.time()
                    else:
                        print("녹화 파일이 존재하지 않습니다.")

            # 안전벨트 경고 메시지
            if not seatbelt_detected:
                cv2.putText(frame_visual, "No Seatbelt", (250, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 결과 출력
            cv2.namedWindow('Drowsiness and Seatbelt Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Drowsiness and Seatbelt Detection', 1280, 960)  # 원하는 창 크기로 조절
            cv2.imshow('Drowsiness and Seatbelt Detection', frame_visual)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    finally:
        # 리소스 정리
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
