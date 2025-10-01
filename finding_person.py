from ultralytics import YOLO
import cv2

# 1) 모델 로드: 가장 가벼운 yolov8n (COCO 사전학습)
model = YOLO("yolov8n.pt")

# 2) 추론: 특정 이미지에서 '사람(class 0)'만
img_path = './frames/media_w440787024_1.ts/frame_100.jpg'  # 분석할 이미지 경로
results = model.predict(
    source=img_path,
    classes=[0],        # 사람만
    conf=0.4,           # 신뢰도 임계값
    iou=0.5,            # NMS IoU
    verbose=False
)

# 3) 결과 시각화 (박스/라벨 그려서 창에 띄우기)
res = results[0]
img = cv2.imread(img_path)

for box in res.boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    conf = float(box.conf[0])
    label = f"person {conf:.2f}"
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
