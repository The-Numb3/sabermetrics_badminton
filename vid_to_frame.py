import cv2
import os

video_path = './video/media_w1113984215_38.ts'  # 상대경로로 변경
output_folder = './frames'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print('동영상을 열수 없습니다.')
    exit()


frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 10 == 0:
        # 파일명 지정: frame_0.jpg, frame_10.jpg, ...
        file_name = f"frame_{frame_count}.jpg"
        save_path = os.path.join(output_folder, file_name)
        
        # 프레임을 이미지 파일로 저장
        cv2.imwrite(save_path, frame)
        print(f"'{save_path}' 저장 완료")

    frame_count += 1

cap.release()
print("작업 완료.")