import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import os

# EAR 계산 함수
def eye_aspect_ratio(eye_points):
    def euclidean_distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    ear_left = (euclidean_distance(eye_points[1], eye_points[5]) + 
                euclidean_distance(eye_points[2], eye_points[4])) / (2.0 * euclidean_distance(eye_points[0], eye_points[3]))

    ear_right = (euclidean_distance(eye_points[7], eye_points[11]) + 
                 euclidean_distance(eye_points[8], eye_points[10])) / (2.0 * euclidean_distance(eye_points[6], eye_points[9]))

    return (ear_left + ear_right) / 2  # 두 눈의 평균 EAR 반환

# TEST 폴더 생성 함수
def create_test_folder():
    test_dir = 'TEST'
    
    # TEST 폴더가 없으면 생성
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    test_num = 0
    while True:
        folder_name = os.path.join(test_dir, f'ALL_TEST_{test_num:02d}')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            return folder_name
        test_num += 1

# 그래프 저장 함수
def save_graph(timestamps, values, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, values, label=ylabel)
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()

# 혼동 행렬 및 성능 지표 저장 함수
def evaluate_model(test_folder, true_labels, predicted_labels):
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(test_folder, 'confusion_matrix.png'))
    plt.close()

# ROC 곡선 저장 함수
def plot_roc_curve(true_labels, predicted_probabilities, test_folder):
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(test_folder, 'roc_curve.png'))
    plt.close()

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 눈 랜드마크 인덱스
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# 비디오 파일 사용 (ds1.mp4)
video_path = 'output.mp4'
cap = cv2.VideoCapture(video_path)

# TEST 폴더와 하위 폴더 생성
test_folder = create_test_folder()
print(f"Results will be saved in: {test_folder}")

# 비디오 저장 설정
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 원본 비디오의 FPS 유지
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 졸음 감지 구간 비디오 저장
drowsiness_video_path = os.path.join(test_folder, 'drowsiness_detected.mp4')
out_drowsiness = cv2.VideoWriter(drowsiness_video_path, fourcc, fps, (frame_width, frame_height))

# 얼굴 중앙점의 600프레임 평균 기록 (10분 기준)
landmark_hist = deque(maxlen=600)

# EAR 및 중앙점 이동 기록
ear_values = []
center_distances = []
timestamps = []
frame_movement = []  # 프레임마다 얼굴 움직임 기록
true_labels = []  # 실제 졸음 상태
predicted_labels = []  # 모델의 졸음 예측 결과
drowsiness_detected = False

# 시간 기록용
start_time = 0

# 얼굴 메쉬를 회색으로, 두께를 줄이도록 설정
mesh_spec = mp_drawing.DrawingSpec(color=(128, 128, 128), thickness=1, circle_radius=1)  # 연한 회색 메쉬

# Face Mesh를 사용한 감지
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    prev_frame_center = None  # 이전 프레임의 중앙점
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 좌우 반전 처리
        frame = cv2.flip(frame, 1)
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # 현재 시간(초)
        timestamps.append(current_time)

        # Mediapipe에 맞게 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)

        # 얼굴 랜드마크 처리
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # 얼굴 랜드마크 그리기 (연한 회색)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mesh_spec  # 회색, 얇은 메쉬
                )

                # 왼쪽 및 오른쪽 눈 좌표 가져오기
                left_eye_points = np.array([[face_landmarks.landmark[idx].x * frame.shape[1], 
                                             face_landmarks.landmark[idx].y * frame.shape[0]] for idx in LEFT_EYE_IDX])
                right_eye_points = np.array([[face_landmarks.landmark[idx].x * frame.shape[1], 
                                              face_landmarks.landmark[idx].y * frame.shape[0]] for idx in RIGHT_EYE_IDX])

                # EAR 계산
                ear_value = eye_aspect_ratio(np.concatenate((left_eye_points, right_eye_points), axis=0))
                ear_values.append(ear_value)

                # 얼굴 중앙점 계산
                face_center_x = int(face_landmarks.landmark[1].x * frame.shape[1])
                face_center_y = int(face_landmarks.landmark[1].y * frame.shape[0])

                # 중앙점 기록
                landmark_hist.append((face_center_x, face_center_y))

                # 최근 10분 평균 중앙점 계산
                avg_face_center_x = int(np.mean([point[0] for point in landmark_hist]))
                avg_face_center_y = int(np.mean([point[1] for point in landmark_hist]))

                # 중앙점 간 거리 계산
                def euclidean_distance(x1, y1, x2, y2):
                    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                distance_between_centers = euclidean_distance(face_center_x, face_center_y, avg_face_center_x, avg_face_center_y)
                center_distances.append(distance_between_centers)

                # 프레임 간 중앙점 이동 계산
                if prev_frame_center:
                    frame_move = euclidean_distance(face_center_x, face_center_y, prev_frame_center[0], prev_frame_center[1])
                    frame_movement.append(frame_move)
                else:
                    frame_movement.append(0)
                prev_frame_center = (face_center_x, face_center_y)

                # 졸음 상태 감지
                if distance_between_centers > 50 or ear_value < 0.2:
                    drowsiness_message = "Drowsiness Detected!"
                    drowsiness_detected = True
                    predicted_labels.append(1)
                else:
                    drowsiness_message = ""
                    drowsiness_detected = False
                    predicted_labels.append(0)

                # 실제 졸음 상태 (임의로 설정, 실제 데이터를 적용할 수 있음)
                if ear_value < 0.2:  # 임계값에 따라 졸음 상태를 추정
                    true_labels.append(1)
                else:
                    true_labels.append(0)

                # 얼굴 중앙점과 평균 중앙점 표시
                cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)
                cv2.circle(frame, (avg_face_center_x, avg_face_center_y), 5, (255, 255, 0), -1)
                cv2.line(frame, (face_center_x, face_center_y), (avg_face_center_x, avg_face_center_y), (0, 255, 255), 2)

                # EAR 값 및 경고 메시지 표시
                cv2.putText(frame, f'EAR: {ear_value:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if drowsiness_message:
                    cv2.putText(frame, drowsiness_message, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 실시간으로 프레임을 비디오에 저장
                out_drowsiness.write(frame)

        # 결과 화면 출력
        cv2.imshow('Drowsiness Detection', frame)

        # ESC 키를 누르면 종료
        if cv2.waitKey(5) & 0xFF == 27:
            break

# 비디오 파일 종료
cap.release()
out_drowsiness.release()
cv2.destroyAllWindows()

# 결과 그래프 저장
ear_graph_path = os.path.join(test_folder, 'ear_values_graph.png')
center_graph_path = os.path.join(test_folder, 'center_movement_graph.png')
frame_move_graph_path = os.path.join(test_folder, 'frame_movement_graph.png')

save_graph(timestamps, ear_values, 'EAR', 'EAR Values Over Time', ear_graph_path)
save_graph(timestamps, center_distances, 'Distance', 'Center Movement Over Time', center_graph_path)
save_graph(timestamps, frame_movement, 'Frame Movement', 'Frame Movement Over Time', frame_move_graph_path)

# 모델 성능 평가 및 저장
evaluate_model(test_folder, true_labels, predicted_labels)

# ROC 곡선 그리기
plot_roc_curve(true_labels, predicted_labels, test_folder)

print(f"Drowsiness video saved to: {drowsiness_video_path}")
print(f"EAR graph saved to: {ear_graph_path}")
print(f"Center movement graph saved to: {center_graph_path}")
print(f"Frame movement graph saved to: {frame_move_graph_path}")
