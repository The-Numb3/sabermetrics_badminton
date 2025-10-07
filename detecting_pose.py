import cv2
import mediapipe as mp

# ------------------- 설정 -------------------

# MediaPipe Pose 솔루션을 초기화합니다.
mp_pose = mp.solutions.pose

# 여기에 분석하고 싶은 이미지 파일의 경로를 넣으세요.
# 예시: image_path = 'C:/Users/YourName/Pictures/my_photo.jpg'
image_path = "./frames/media_w440787024_1.ts/frame_100.jpg"
# ------------------- 실행 -------------------

# 1. OpenCV로 이미지 파일을 읽어옵니다.
image = cv2.imread(image_path)

# 2. 이미지를 성공적으로 읽었는지 확인합니다.
if image is None:
    print(f"오류: '{image_path}' 이미지를 찾을 수 없거나 열 수 없습니다.")
else:
    # 3. MediaPipe Pose를 context manager로 사용 (권장 방식)
    with mp_pose.Pose(
        static_image_mode=True,  # 단일 이미지 처리에 최적화
        model_complexity=1,      # 속도와 정확도의 중간 단계 모델 사용
        min_detection_confidence=0.5
    ) as pose:
        # 4. 랜드마크 좌표를 실제 픽셀 위치로 변환하기 위해 이미지의 높이와 너비를 가져옵니다.
        height, width, _ = image.shape

        # 5. MediaPipe 처리를 위해 OpenCV의 BGR 색상 이미지를 RGB로 변환합니다.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 6. MediaPipe Pose 모델로 이미지에서 포즈를 감지합니다.
        results = pose.process(image_rgb)

        # 7. 포즈 랜드마크(관절 위치)가 성공적으로 감지되었는지 확인합니다.
        if results.pose_landmarks:
            print("포즈 감지 성공! 이미지에 점을 그립니다.")
            
            # 8. 감지된 모든 랜드마크(33개)를 하나씩 순회합니다.
            for landmark in results.pose_landmarks.landmark:
                
                # 9. (핵심) 랜드마크의 좌표 계산
                # landmark.x와 landmark.y는 0.0 ~ 1.0 사이의 비율 값입니다.
                # 이 비율 값에 이미지의 너비와 높이를 곱해서 실제 픽셀 좌표(cx, cy)를 계산합니다.
                cx = int(landmark.x * width)
                cy = int(landmark.y * height)
                
                # 10. (핵심) OpenCV의 circle 함수를 사용해 계산된 좌표에 점(원)을 그립니다.
                # cv2.circle(그릴 이미지, (중심 x좌표, 중심 y좌표), 반지름(픽셀), (BGR 색상), 두께)
                # 두께를 -1로 하면 원 내부가 채워집니다.
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1) # 녹색 점, 반지름 5픽셀

        else:
            print("이미지에서 포즈를 감지하지 못했습니다.")

        # 11. 점이 그려진 결과 이미지를 'Pose Landmarks'라는 이름의 새 창에 보여줍니다.
        cv2.imshow('Pose Landmarks as Dots', image)

        # 12. 사용자가 키보드의 아무 키나 누를 때까지 창을 계속 열어 둡니다.
        cv2.waitKey(0)

        # 13. 열려있는 모든 창을 닫습니다.
        cv2.destroyAllWindows()

# context manager를 사용하므로 pose.close()가 자동으로 호출됩니다.