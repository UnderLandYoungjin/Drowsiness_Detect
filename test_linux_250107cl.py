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
    # 리눅스 기본 폰트 경로 설정
    FONT_PATH: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # 대부분의 리눅스 시스템에 설치된 폰트
    # 리눅스 스타일의 임시 디렉토리 경로
    TEMP_FOLDER: str = "/tmp/dms_recordings"
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
        # 리눅스 스타일의 디렉토리 생성
        os.makedirs(self.TEMP_FOLDER, exist_ok=True)
        
        # 폰트 파일 존재 여부 확인 및 대체 폰트 설정
        if not os.path.exists(self.FONT_PATH):
            # 대체 폰트 목록
            alternative_fonts = [
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf"
            ]
            
            for font in alternative_fonts:
                if os.path.exists(font):
                    self.FONT_PATH = font
                    break
            else:
                raise FileNotFoundError("No suitable font found in the system")

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
        # 리눅스 스타일의 파일 경로
        self.visual_output_filename = os.path.join(self.config.TEMP_FOLDER, "current_recording_visual.mp4")
        self.original_output_filename = os.path.join(self.config.TEMP_FOLDER, "current_recording_original.mp4")
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
                try:
                    with open(filename, 'rb') as f:
                        files = {'file': f}
                        response = requests.post(self.config.UPLOAD_URL, files=files)
                        response.raise_for_status()
                    os.remove(filename)
                except (requests.RequestException, OSError) as e:
                    print(f"Error during file upload/removal: {e}")
                    return False

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
        
        try:
            self.font_large = ImageFont.truetype(config.FONT_PATH, 20)
            self.font_small = ImageFont.truetype(config.FONT_PATH, 5)
        except OSError as e:
            print(f"Error loading font: {e}")
            # PIL의 기본 폰트 사용
            self.font_large = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

    # ... [나머지 DrowsinessDetector 클래스 메소드들은 동일] ...

class SeatbeltDetector:
    """안전벨트 감지를 관리하는 클래스"""
    def __init__(self, model_path: str, config: Config):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model file not found: {model_path}")
        self.model = YOLO(model_path)
        self.config = config

    # ... [나머지 SeatbeltDetector 클래스 메소드들은 동일] ...

def main():
    """메인 함수"""
    try:
        config = Config()
        
        # 카메라 장치 선택
        # 리눅스에서는 일반적으로 /dev/video0 등으로 접근
        camera_index = 0
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            # 카메라를 열 수 없는 경우 다른 장치 번호 시도
            for i in range(1, 5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    break
            if not cap.isOpened():
                raise RuntimeError("No camera device found")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_recorder = VideoRecorder(frame_width, frame_height, config)
        drowsiness_detector = DrowsinessDetector(config)
        
        # YOLO 모델 경로를 현재 디렉토리 기준으로 설정
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_e_dt3000.pt')
        seatbelt_detector = SeatbeltDetector(model_path, config)

        drowsy_start_time = None
        sent_video = False
        cooldown_start_time = None
        recording_additional_15_seconds = False
        additional_recording_start_time = None

        print("Starting DMS system...")
        print(f"Using camera device: {camera_index}")
        print(f"Frame size: {frame_width}x{frame_height}")
        print(f"Temporary folder: {config.TEMP_FOLDER}")
        print(f"Using font: {config.FONT_PATH}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
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
                    print("Drowsiness detected: Starting 15s recording")
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
                        print(f"Successfully uploaded video at {timestamp}")
                        cooldown_start_time = time.time()
                    else:
                        print("Failed to save or upload recording")

            # 안전벨트 경고 메시지
            if not seatbelt_detected:
                cv2.putText(frame_visual, "No Seatbelt", (250, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 결과 출력
            cv2.imshow('Drowsiness and Seatbelt Detection', frame_visual)

            # ESC 키로 종료
            if cv2.waitKey(5) & 0xFF == 27:
                print("ESC pressed, stopping...")
                break

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # 리소스 정리
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("System shutdown complete")

if __name__ == "__main__":
    main()
