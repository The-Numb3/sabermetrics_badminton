import math
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

image_path = './frames/media_w440787024_1.ts/frame_400.jpg'

#선 검출 파라미터 튜닝할만한 부분: self.kernel 크기, 허프변환(cv2.HoughLines)의 threshold값

class court_detection:
    def __init__(self, path):
        self.image_path = path
        self.img = cv2.imread(self.image_path)
        self.orig = self.img.copy()
        if self.img is None:
            raise FileNotFoundError(self.image_path)
        
        #이미지 컬러스페이스 변환, bgr -> gray:
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        #CLAHE 적용, 흑백 명암대비 상향
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.img = clahe.apply(self.img)

        #가우시안 블러 적용
        self.img = cv2.GaussianBlur(self.img, (5,5), 0)

        #밝은선 강조
        self.img = cv2.addWeighted(self.img, 0.5, cv2.normalize(self.img, None, 0, 255, cv2.NORM_MINMAX), 0.5, 0)

        #모폴로지로 구멍 메우기/노이즈 제거
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, self.kernel, iterations=1)

        #엣지 검출
        self.img = cv2.Canny(self.img, 50, 150, apertureSize=3, L2gradient=True)

        #허프직선변환
        self.lines = cv2.HoughLines(self.img, 1, np.pi/180, threshold=200)

        #라인 클러스터링
        self.lines = self.line_clustering_dbscan(self.lines)

        #출력하는 부분
        if self.lines is not None:
            for l in self.lines[:,0,:]:
                rho, theta = l
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                # 직선의 양 끝점 계산 (이미지 경계까지 충분히 연장)
                x1 = int(x0 + 2000*(-b))
                y1 = int(y0 + 2000*(a))
                x2 = int(x0 - 2000*(-b))
                y2 = int(y0 - 2000*(a))
                cv2.line(self.orig, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.imshow('HoughLines result', self.orig)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        

        #각도/거리 리스트 생성
        self.rhos = []
        self.angles = []
        for l in self.lines[:,0,:]:
            rho, theta = l
            ang = theta
            self.rhos.append(rho)
            self.angles.append(ang)

    #self.lines의 결과로 직선을 이미지에 그리는 함수
    def overlay_line(self, line, color=(0,0,255), thickness=2):
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # 직선의 양 끝점 계산 (이미지 경계까지 충분히 연장)
        x1 = int(x0 + 2000*(-b))
        y1 = int(y0 + 2000*(a))
        x2 = int(x0 - 2000*(-b))
        y2 = int(y0 - 2000*(a))
        cv2.line(self.img, (x1, y1), (x2, y2), color, thickness)

    def line_clustering_dbscan(self, lines, threshold= 20, min_samples=1):

        #선의 개수가 충분하지 않다고 판단
        if lines is None:
            raise RuntimeError("충분한 직선을 찾지 못했습니다, 선 검출을 다시하세요")
        
        lines = lines[:,0,:] # (N,1,2) -> (N,2)
        
        rho_list = lines[:,0]
        theta_list = lines[:,1]

        xs = rho_list * np.cos(theta_list)
        ys = rho_list * np.sin(theta_list)
        pts = np.column_stack([xs, ys])

        db = DBSCAN(eps=float(threshold), min_samples=int(min_samples))
        labels = db.fit_predict(pts)

        clusters = []
        for lbl in sorted(set(labels)):
            if lbl == -1:
                continue
            idx = (labels == lbl)
            cx = pts[idx, 0].mean()
            cy = pts[idx, 1].mean()
            rho = float(math.hypot(cx, cy))
            theta = float(math.atan2(cy, cx))
            clusters.append((rho, theta, int(idx.sum())))

        # 반환 형식을 cv2.HoughLines와 유사한 형태로 맞춥니다: (N, 1, 2) 배열, 각 항목은 [rho, theta]
        if len(clusters) == 0:
            return None

        clusters_arr = np.array([[[c[0], c[1]]] for c in clusters], dtype=np.float32)
        return clusters_arr

court_detection(image_path)