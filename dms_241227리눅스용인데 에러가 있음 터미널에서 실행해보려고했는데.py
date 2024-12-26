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
import logging
import sys
from time import sleep

# 로깅 설정
logging.basicConfig(
    filename='drowsiness_detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 전역 변수 설정
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye_points):
    try:
        def euclidean_distance(p1, p2):
            return np.linalg.norm(p1 - p2)
        ear_left = (euclidean_distance(eye_points[1], eye_points[5]) +
                    euclidean_distance(eye_points[2], eye_points[4])) / (2.0 * euclidean_distance(eye_points[0], eye_points[3]))
        ear_right = (euclidean_distance(eye_points[7], eye_points[11]) +
                     euclidean_distance(eye_points[8], eye_points[10])) / (2.0 * euclidean_distance(eye_points[6], eye_points[9]))
        return (ear_left + ear_right) / 2
    except Exception as e:
        logging.error(f"EAR calculation error: {str(e)}")
        return 0.0

def main():
    while True:  # 메인 루프 - 에러 발생 시 재시작
        try:
            # Mediapipe 모듈 초기화
            mp_face_mesh = mp.solutions.face_mesh

            # 리눅스 시스템에 맞는 폰트 경로 설정
            try:
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                font_large = ImageFont.truetype(font_path, 20)
                font_small = ImageFont.truetype(font_path, 5)
            except Exception as e:
                logging.error(f"Font loading error: {str(e)}")
                raise

            # YOLO 모델 초기화
            try:
                seatbelt_model = YOLO('best_e_dt3000.pt')
            except Exception as e:
                logging.error(f"YOLO model loading error: {str(e)}")
                raise

            # 카메라 초기화
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    raise Exception("Failed to open camera")
            except Exception as e:
                logging.error(f"Camera initialization error: {str(e)}")
                time.sleep(5)  # 카메라 초기화 실패시 5초 대기 후 재시도
                continue

            # 비디오 설정
            FPS = 20
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # 출력 파일 이름 설정
            visual_output_filename = "current_recording_visual.mp4"
            original_output_filename = "current_recording_original.mp4"

            # 상태 변수 초기화
            drowsy_start_time = None
            sent_video = False
            cooldown_start_time = None
            recording_additional_15_seconds = False
            additional_recording_start_time = None
            landmark_hist = deque(maxlen=600)
            pre_drowsy_buffer = deque(maxlen=15 * FPS)

            with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as face_mesh:
                while cap.isOpened():
                    try:
                        ret, frame = cap.read()
                        if not ret:
                            raise Exception("Failed to read frame")

                        frame = cv2.flip(frame, 0)
                        original_frame = frame.copy()

                        # YOLO 모델로 안전벨트 감지
                        try:
                            yolo_results = seatbelt_model(frame)
                            seatbelt_detected = False
                            
                            for result in yolo_results:
                                boxes = result.boxes.xyxy.cpu().numpy()
                                scores = result.boxes.conf.cpu().numpy()
                                classes = result.boxes.cls.cpu().numpy()

                                for box, score, cls in zip(boxes, scores, classes):
                                    if score < 0.70:
                                        continue
                                    x1, y1, x2, y2 = map(int, box)
                                    label = f"SEATBELT : {score:.2f}"
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)
                                    cv2.putText(frame, label, (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)
                                    seatbelt_detected = True

                            if not seatbelt_detected:
                                cv2.putText(frame, "No Seatbelt", (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        except Exception as e:
                            logging.error(f"YOLO detection error: {str(e)}")
                            continue

                        # PIL 이미지 변환
                        try:
                            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(frame_pil)
                        except Exception as e:
                            logging.error(f"PIL conversion error: {str(e)}")
                            continue

                        # 얼굴 감지 및 EAR 계산
                        try:
                            result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            drowsy_detected = False

                            if cooldown_start_time and time.time() - cooldown_start_time >= 16:
                                cooldown_start_time = None
                                sent_video = False

                            if result.multi_face_landmarks:
                                for face_landmarks in result.multi_face_landmarks:
                                    left_eye_points = np.array([[face_landmarks.landmark[idx].x * frame_width, 
                                                               face_landmarks.landmark[idx].y * frame_height] for idx in LEFT_EYE_IDX])
                                    right_eye_points = np.array([[face_landmarks.landmark[idx].x * frame_width, 
                                                                face_landmarks.landmark[idx].y * frame_height] for idx in RIGHT_EYE_IDX])

                                    ear_value = eye_aspect_ratio(np.concatenate((left_eye_points, right_eye_points), axis=0))

                                    ear_color = (255, 0, 0) if ear_value < 0.2 else (0, 255, 0)
                                    draw.text((30, 30), f'EAR: {ear_value:.2f}', font=font_large, fill=ear_color)

                                    face_center_x = int(face_landmarks.landmark[1].x * frame_width)
                                    face_center_y = int(face_landmarks.landmark[1].y * frame_height)
                                    landmark_hist.append((face_center_x, face_center_y))

                                    avg_face_center_x = int(np.mean([point[0] for point in landmark_hist]))
                                    avg_face_center_y = int(np.mean([point[1] for point in landmark_hist]))

                                    distance_between_centers = np.sqrt((avg_face_center_x - face_center_x) ** 2 +
                                                                     (avg_face_center_y - face_center_y) ** 2)
                                    distance_color = (0, 255, 0) if distance_between_centers <= 40 else (255, 0, 0)
                                    draw.text((face_center_x + 10, face_center_y + 10), 
                                            f"Distance: {distance_between_centers:.2f}", 
                                            font=font_large, fill=distance_color)

                                    if ear_value < 0.2 and distance_between_centers > 40:
                                        drowsy_start_time = time.time() if drowsy_start_time is None else drowsy_start_time
                                        if time.time() - drowsy_start_time >= 1:
                                            drowsy_detected = True
                                            draw.text((30, 60), "Drowsy!", font=font_large, fill=(255, 0, 0))
                                    else:
                                        drowsy_start_time = None

                                    # 시각화 요소 그리기
                                    x_coords = [landmark.x * frame_width for landmark in face_landmarks.landmark]
                                    y_coords = [landmark.y * frame_height for landmark in face_landmarks.landmark]
                                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                                    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=(0, 0, 255), width=3)
                                    draw.line([(face_center_x, face_center_y), (avg_face_center_x, avg_face_center_y)], 
                                            fill=(255, 0, 0), width=2)
                                    draw.ellipse((face_center_x - 5, face_center_y - 5, face_center_x + 5, face_center_y + 5), 
                                               fill=(0, 255, 255))
                                    draw.ellipse((avg_face_center_x - 5, avg_face_center_y - 5, avg_face_center_x + 5, avg_face_center_y + 5), 
                                               fill=(255, 255, 0))

                                    all_eye_points = np.concatenate((left_eye_points, right_eye_points), axis=0)
                                    for i, (x, y) in enumerate(all_eye_points):
                                        label = f"p{i+1}"
                                        draw.text((x + 5, y - 5), label, font=font_small, fill=(255, 165, 0))
                                        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 165, 0))

                        except Exception as e:
                            logging.error(f"Face detection error: {str(e)}")
                            continue

                        # 프레임 변환 및 저장
                        try:
                            frame_visual = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                            pre_drowsy_buffer.append((frame_visual, original_frame))
                            cv2.imshow("Drowsiness Detection", frame_visual)

                        except Exception as e:
                            logging.error(f"Frame conversion error: {str(e)}")
                            continue

                        # 졸음 감지 시 비디오 저장
                        if drowsy_detected and not sent_video:
                            try:
                                sent_video = True
                                recording_additional_15_seconds = True
                                additional_recording_start_time = time.time()

                                out_visual = cv2.VideoWriter(visual_output_filename, fourcc, FPS, (frame_width, frame_height))
                                out_original = cv2.VideoWriter(original_output_filename, fourcc, FPS, (frame_width, frame_height))

                                for buffered_frame_visual, buffered_frame_original in pre_drowsy_buffer:
                                    out_visual.write(buffered_frame_visual)
                                    out_original.write(buffered_frame_original)

                            except Exception as e:
                                logging.error(f"Video initialization error: {str(e)}")
                                continue

                        # 추가 녹화 처리
                        if recording_additional_15_seconds:
                            try:
                                out_visual.write(frame_visual)
                                out_original.write(original_frame)

                                if time.time() - additional_recording_start_time >= 15:
                                    recording_additional_15_seconds = False
                                    out_visual.release()
                                    out_original.release()

                                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                                    visual_filename = f"/tmp/HPNT_V_{timestamp}.mp4"
                                    original_filename = f"/tmp/HPNT_O_{timestamp}.mp4"

                                    os.rename(visual_output_filename, visual_filename)
                                    os.rename(original_output_filename, original_filename)

                                    # 파일 업로드 (3번 재시도)
                                    for retry in range(3):
                                        try:
                                            with open(visual_filename, 'rb') as f:
                                                response = requests.post('http://221.152.105.81:1200/upload', 
                                                                      files={'file': f}, 
                                                                      timeout=10)
                                                if response.status_code != 200:
                                                    raise Exception(f"Upload failed with status {response.status_code}")
                                            
                                            with open(original_filename, 'rb') as f:
                                                response = requests.post('http://221.152.105.81:1200/upload', 
                                                                      files={'file': f}, 
                                                                      timeout=10)
                                                if response.status_code != 200:
                                                    raise Exception(f"Upload failed with status {response.status_code}")
                                            
                                            break
                                        except Exception as e:
                                            if retry == 2:  # 마지막 시도에서 실패
                                                logging.error(f"Final upload attempt failed: {str(e)}")
                                            else:
                                                logging.warning(f"Upload attempt {retry + 1} failed: {str(e)}")
                                                time.sleep(2)  # 재시도 전 2초 대기

                                    try:
                                        os.remove(visual_filename)
                                        os.remove(original_filename)
                                    except Exception as e:
                                        logging.error(f"File deletion error: {str(e)}")

                                    cooldown_start_time = time.time()

                            except Exception as e:
                                logging.error(f"Recording processing error: {str(e)}")
                                if 'out_visual' in locals():
                                    out_visual.release()
                                if 'out_original' in locals():
                                    out_original.release()

                        if cv2.waitKey(5) & 0xFF == 27:
                            break

                    except Exception as e:
                        logging.error(f"Main loop iteration error: {str(e)}")
                        continue

        except Exception as e:
            logging.error(f"Critical error in main process: {str(e)}")
            logging.info("Attempting restart in 5 seconds...")
            
            # 리소스 정리
            try:
                if 'cap' in locals():
                    cap.release()
                if 'out_visual' in locals():
                    out_visual.release()
                if 'out_original' in locals():
                    out_original.release()
                cv2.destroyAllWindows()
            except Exception as cleanup_error:
                logging.error(f"Cleanup error: {str(cleanup_error)}")
            
            time.sleep(5)  # 재시작 전 대기
            continue
        
        finally:
            try:
                # 최종 리소스 정리
                if 'cap' in locals():
                    cap.release()
                if 'out_visual' in locals():
                    out_visual.release()
                if 'out_original' in locals():
                    out_original.release()
                cv2.destroyAllWindows()
            except Exception as e:
                logging.error(f"Final cleanup error: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Program terminated by user")
        # 최종 정리
        try:
            cv2.destroyAllWindows()
        except:
            pass
        sys.exit(0)
    except Exception as e:
        logging.critical(f"Unhandled exception: {str(e)}")
        sys.exit(1)
