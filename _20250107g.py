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

# Windows 시스템에 맞는 폰트 경로 설정 (Windows 11, Arial 폰트 예시)
font_path = "C:/Windows/Fonts/arial.ttf"  # Windows Arial 폰트 경로
font_large = ImageFont.truetype(font_path, 20)  # EAR와 거리 텍스트 크기
font_small = ImageFont.truetype(font_path, 5)   # p1~p12 포인트 텍스트 크기

# 서버 URL 설정
upload_url = 'http://221.152.105.81:1200/upload'

# 왼쪽 및 오른쪽 눈의 랜드마크 인덱스
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# YOLO 모델 초기화 (안전벨트 탐지용) -> best_e_dt3000.pt 모델 사용
seatbelt_model = YOLO('best_e_dt3000.pt')

# ----------------------------------------------------------------
# 함수: EAR 계산 함수
# ----------------------------------------------------------------
def eye_aspect_ratio(eye_points):
    """
    눈 좌표 배열(eye_points)을 받아서 EAR 값을 계산하는 함수이다.
    """
    def euclidean_distance(p1, p2):
        return np.linalg.norm(p1 - p2)
    # 왼쪽 눈 EAR 계산
    ear_left = (euclidean_distance(eye_points[1], eye_points[5]) +
                euclidean_distance(eye_points[2], eye_points[4])) / (2.0 * euclidean_distance(eye_points[0], eye_points[3]))
    # 오른쪽 눈 EAR 계산
    ear_right = (euclidean_distance(eye_points[7], eye_points[11]) +
                 euclidean_distance(eye_points[8], eye_points[10])) / (2.0 * euclidean_distance(eye_points[6], eye_points[9]))
    return (ear_left + ear_right) / 2

# ----------------------------------------------------------------
# 함수: 파일 업로드 함수(네트워크 예외 처리 추가)
# ----------------------------------------------------------------
def upload_file(file_path):
    """
    녹화된 파일을 서버에 업로드하는 함수이다.
    네트워크 예외 처리를 포함한다.
    """
    if not os.path.exists(file_path):
        print(f"파일이 존재하지 않음: {file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(upload_url, files=files, timeout=10)  # 10초 타임아웃 예시
            response.raise_for_status()  # HTTP 에러 발생 시 예외 처리
            print(f"업로드 성공: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"업로드 실패(네트워크 오류): {file_path}, 오류: {e}")

# ----------------------------------------------------------------
# 메인 처리
# ----------------------------------------------------------------
def main():
    # 카메라(혹은 동영상) 초기화
    cap = cv2.VideoCapture(0)
    
    # 촬영 환경 (예: Nikon DSLR D780 -> 웹캠 대체 사용 가능)
    # 시스템 예: Windows 11, Intel Core i7-12700K, NVIDIA GeForce RTX 3060, RAM 16GB

    FPS = 20
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 두 개의 출력 파일 이름 설정
    visual_output_filename = "current_recording_visual.mp4"
    original_output_filename = "current_recording_original.mp4"

    # 비디오 작성기 설정 (MP4 코덱)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_visual = cv2.VideoWriter(visual_output_filename, fourcc, FPS, (frame_width, frame_height))
    out_original = cv2.VideoWriter(original_output_filename, fourcc, FPS, (frame_width, frame_height))

    # 졸음 탐지 상태 및 타이머 초기화
    drowsy_start_time = None
    sent_video = False  # 영상 전송 여부 플래그
    cooldown_start_time = None  # 16초 대기 시간 플래그
    recording_additional_15_seconds = False  # 추가 녹화 상태 플래그
    additional_recording_start_time = None  # 추가 녹화 시작 시간

    # 얼굴 중앙 좌표 히스토리 (약 10분간 중앙 좌표 저장)
    landmark_hist = deque(maxlen=600)

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

            # YOLO 모델로 안전벨트 탐지
            yolo_results = seatbelt_model(frame)
            seatbelt_detected = False

            for result in yolo_results:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                # 안전벨트 탐지 결과 처리
                for box, score, cls in zip(boxes, scores, classes):
                    if score < 0.55:  # 신뢰도 55% 미만은 필터링
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    label = f"SEAT_BELT : {score:.2f}"

                    # 안전벨트 영역 표시 (연두색)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 127), 2)
                    cv2.putText(frame, label, (x1 + 5, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 127), 2)
                    seatbelt_detected = True

            # 얼굴 인식 및 졸음 감지 (PIL로 처리)
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(frame_rgb)

            frame_pil = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(frame_pil)

            drowsy_detected = False

            # 쿨다운이 끝났다면 sent_video를 초기화
            if cooldown_start_time and time.time() - cooldown_start_time >= 16:
                cooldown_start_time = None
                sent_video = False

            # 얼굴 랜드마크가 감지된 경우
            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    # 왼쪽/오른쪽 눈 랜드마크 추출
                    left_eye_points = np.array([
                        [face_landmarks.landmark[idx].x * frame_width, 
                         face_landmarks.landmark[idx].y * frame_height] for idx in LEFT_EYE_IDX
                    ])
                    right_eye_points = np.array([
                        [face_landmarks.landmark[idx].x * frame_width, 
                         face_landmarks.landmark[idx].y * frame_height] for idx in RIGHT_EYE_IDX
                    ])

                    # EAR 계산
                    ear_value = eye_aspect_ratio(np.concatenate((left_eye_points, right_eye_points), axis=0))

                    # EAR 색상 결정 및 텍스트 표시
                    ear_color = (255, 0, 0) if ear_value < 0.2 else (0, 255, 0)
                    draw.text((30, 30), f'EAR: {ear_value:.2f}', font=font_large, fill=ear_color)

                    # 졸음 감지 기준
                    if ear_value < 0.2:
                        drowsy_start_time = time.time() if drowsy_start_time is None else drowsy_start_time
                        if time.time() - drowsy_start_time >= 1:
                            drowsy_detected = True
                            draw.text((30, 60), "Drowsy!", font=font_large, fill=(255, 0, 0))
                    else:
                        drowsy_start_time = None

                    # 얼굴 중앙 좌표(Cpoint) 계산
                    face_center_x = int(face_landmarks.landmark[1].x * frame_width)
                    face_center_y = int(face_landmarks.landmark[1].y * frame_height)
                    landmark_hist.append((face_center_x, face_center_y))

                    # 최근 중앙 좌표의 평균 좌표(C Avg Point)
                    avg_face_center_x = int(np.mean([point[0] for point in landmark_hist]))
                    avg_face_center_y = int(np.mean([point[1] for point in landmark_hist]))

                    # 중앙 좌표 간 거리 계산
                    distance_between_centers = np.sqrt((avg_face_center_x - face_center_x) ** 2 +
                                                       (avg_face_center_y - face_center_y) ** 2)
                    distance_color = (0, 255, 0) if distance_between_centers <= 40 else (255, 0, 0)
                    draw.text(
                        (face_center_x + 10, face_center_y + 10), 
                        f"Distance: {distance_between_centers:.2f}", 
                        font=font_large, fill=distance_color
                    )

                    # 얼굴 바운딩 박스 (파란색)
                    x_coords = [landmark.x * frame_width for landmark in face_landmarks.landmark]
                    y_coords = [landmark.y * frame_height for landmark in face_landmarks.landmark]
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=(0, 0, 255), width=3)

                    # Cpoint와 C Avg Point 사이 빨간색 선
                    draw.line(
                        [(face_center_x, face_center_y), (avg_face_center_x, avg_face_center_y)],
                        fill=(255, 0, 0), width=2
                    )

                    # Cpoint (청록색)와 C Avg Point (노란색)
                    draw.ellipse(
                        (face_center_x - 5, face_center_y - 5, face_center_x + 5, face_center_y + 5), 
                        fill=(0, 255, 255)
                    )
                    draw.ellipse(
                        (avg_face_center_x - 5, avg_face_center_y - 5, avg_face_center_x + 5, avg_face_center_y + 5), 
                        fill=(255, 255, 0)
                    )

                    # p1 ~ p12 표시
                    all_eye_points = np.concatenate((left_eye_points, right_eye_points), axis=0)
                    for i, (ex, ey) in enumerate(all_eye_points):
                        label = f"p{i+1}"
                        draw.text((ex + 5, ey - 5), label, font=font_small, fill=(255, 165, 0))
                        draw.ellipse((ex - 2, ey - 2, ex + 2, ey + 2), fill=(255, 165, 0))

            # 안전벨트 미착용 경고
            if not seatbelt_detected:
                draw.text((250, 300), "No Seatbelt", font=font_large, fill=(255, 0, 0))

            # PIL -> OpenCV 변환
            frame_visual = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            # 프레임 기록
            out_visual.write(frame_visual)
            out_original.write(frame)

            # 졸음 감지되었고, 전송되지 않았다면 추가 녹화 15초 시작
            if drowsy_detected and not sent_video:
                sent_video = True
                recording_additional_15_seconds = True
                additional_recording_start_time = time.time()
                print("졸음 감지됨: 추가 15초 녹화 시작")

            # 추가 15초 녹화 동작
            if recording_additional_15_seconds:
                if time.time() - additional_recording_start_time >= 15:
                    recording_additional_15_seconds = False
                    out_visual.release()
                    out_original.release()

                    # 임시 폴더
                    temp_folder = "C:/tmp"
                    if not os.path.exists(temp_folder):
                        os.makedirs(temp_folder)

                    # 타임스탬프
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    visual_filename = os.path.join(temp_folder, f"P_V_{timestamp}.mp4")
                    original_filename = os.path.join(temp_folder, f"P_O_{timestamp}.mp4")

                    # 파일 이동 (예외 처리 추가)
                    try:
                        if os.path.exists(visual_output_filename) and os.path.exists(original_output_filename):
                            os.rename(visual_output_filename, visual_filename)
                            os.rename(original_output_filename, original_filename)
                            
                            # 파일 전송 (시각 정보 포함 파일)
                            upload_file(visual_filename)
                            # 파일 전송 (시각 정보 미포함 파일)
                            upload_file(original_filename)

                            # 전송 후 삭제
                            os.remove(visual_filename)
                            os.remove(original_filename)
                        else:
                            print("녹화 파일이 존재하지 않음")
                    except Exception as e:
                        print(f"파일 처리 중 오류 발생: {e}")

                    # VideoWriter 객체 재생성
                    out_visual = cv2.VideoWriter(visual_output_filename, fourcc, FPS, (frame_width, frame_height))
                    out_original = cv2.VideoWriter(original_output_filename, fourcc, FPS, (frame_width, frame_height))

                    # 쿨다운 타이머 시작
                    cooldown_start_time = time.time()

            # 화면 표시

                        # ---------------------------------------------------
            # 화면 크기를 크게 하기 위해 윈도우 설정 부분 추가
            # ---------------------------------------------------
            cv2.namedWindow('Drowsiness and Seatbelt Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Drowsiness and Seatbelt Detection', 1280, 960)  # 원하는 창 크기로 조절
            cv2.imshow('Drowsiness and Seatbelt Detection', frame_visual)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    # 종료 처리
    out_visual.release()
    out_original.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
