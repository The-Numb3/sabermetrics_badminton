import cv2
import numpy as np

# 이미지 파일 경로
image_path = './frames/frame_0.jpg'  

class court_detection:
    def __call__(self, path):
        # 1. 이미지 불러오기
        self.image_path = path
        self.img = cv2.imread(self.image_path)
        if self.img is None:
            raise FileNotFoundError(self.image_path)
        
        # 2. 이미지 크기 리사이즈 (너비가 1600px 이상이면 줄이기)
        self.h, self.w = self.img.shape[0:2]
        self.scale = 1.0
        self.max_size = 1600
        if self.w > self.max_size:
            self.scale = self.max_size / self.w
            self.img = cv2.resize(self.img, (int(self.w*self.scale), int(self.scale*self.h)), interpolation=cv2.INTER_AREA)
            self.h, self.w = self.img.shape[0:2]
        
        # 출력용 이미지 복사
        self.vis = self.img.copy()
        self.overlay = self.vis.copy()

        # 3. HSV에서 흰색 계열 마스크 추출 (코트 라인 강조)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.lower = np.array([0, 0, 180], dtype=np.uint8)
        self.upper = np.array([180, 60, 255], dtype=np.uint8)
        self.mask = cv2.inRange(self.hsv, self.lower, self.upper)
        
        '''        # 4. 명암 대비 보강 → 이진 마스크 생성
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.enhanced = cv2.addWeighted(self.gray, 0.5, cv2.normalize(self.gray, None, 0, 255, cv2.NORM_MINMAX), 0.5, 0)
        self.mask2 = cv2.threshold(self.enhanced, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        self.mask = cv2.bitwise_and(self.mask, self.mask2)'''

        # 5. 모폴로지 연산으로 잡음 제거 + 선 두껍게
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, self.kernel, iterations=1)

        # 6. 캐니 엣지 검출 후 허프 직선 변환
        self.edges = cv2.Canny(self.mask, 50, 150, apertureSize=3, L2gradient=True)

        # HoughLines → 무한 직선 검출 (rho, theta 반환)
        self.lines = cv2.HoughLines(self.edges, 1, np.pi/180, threshold=120)

        # --- 허프 변환 결과 시각화: 모든 검출된 직선 빨간색으로 표시 ---
        vis_lines = self.img.copy()
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
                cv2.line(vis_lines, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.imshow('HoughLines result', vis_lines)
            cv2.waitKey(0)
            cv2.destroyWindow('HoughLines result')

        if self.lines is None or len(self.lines) < 4:
            raise RuntimeError("충분한 직선을 찾지 못했습니다. 조명/대비/파라미터를 조정하세요.")

        # 7. 각도/거리 리스트 생성
        self.rhos = []
        self.angles = []
        for l in self.lines[:,0,:]:
            rho, theta = l
            ang = theta
            self.rhos.append(rho)
            self.angles.append(ang)

        if len(self.rhos) < 4:
            raise RuntimeError("유효한 긴 직선이 부족합니다.")
        
        # 8. 각도를 KMeans로 두 그룹(가로/세로)으로 분리
        self.angles = np.array(self.angles, dtype=np.float32)
        self.labels = self.classify_angles(self.angles)

        # 직선 방정식(A,B,C)을 가로/세로 그룹으로 분류
        self.fam0 = [] #가로 직선 후보
        self.fam1 = [] #세로 직선 후보
        for rho, theta, label in zip(self.rhos,self.angles, self.labels):
            L = self.rho_theta_to_abc(rho, theta)
            if L is None:
                continue
            if label == 0:
                self.fam0.append(L)
            else:
                self.fam1.append(L)
        print(f"가로직선 후보: {len(self.fam0)}, 세로직선 후보: {len(self.fam1)}")
        # 9. 각 그룹에서 가장 바깥쪽 두 개 선만 선택 (코트 외곽선)
        self.pick0 = self.pick_outer_two_lines(self.fam0, self.img.shape) if len(self.fam0) >= 2 else None
        self.pick1 = self.pick_outer_two_lines(self.fam1, self.img.shape) if len(self.fam1) >= 2 else None
        (self.L0a, self.L0b) = self.pick0
        (self.L1a, self.L1b) = self.pick1
        for L in [self.L0a, self.L0b, self.L1a, self.L1b]:
            print(f"선택된 직선: {L}")

        '''# 13. 결과 시각화 (외곽선/군집 직선/교차점)
        for L in [self.L0a, self.L0b, self.L1a, self.L1b]:
            self.overlay_lines(L)
        for L in self.fam0:
            self.overlay_lines(L,(0,255,0))
        for L in self.fam1:
            self.overlay_lines(L,(0,0,255))


        # 14. 결과 출력
        cv2.imshow("overlay", self.overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        # 10. 선택된 직선들 교차점 계산 → 네 모서리 추출
        self.pts = []
        for La in [self.L0a, self.L0b]:
            for Lb in [self.L1a, self.L1b]:
                self.pt = self.intersect_lines(La, Lb)
                if self.pt is not None and 0 <= self.pt[0] < self.w and 0 <= self.pt[1] < self.h:
                    self.pts.append(self.pt)
                    print(f"교차점: {self.pt}") # 교차점 출력

        # 11. 교차점 정리 (중복 제거 후 4개만 유지)
        if len(self.pts) != 4:
            self.uniq = []
            for p in self.pts:
                if all(np.hypot(*(p-q)) > 3.0 for q in self.uniq):
                    self.uniq.append(p)
            if len(self.uniq) < 4:
                raise RuntimeError("코트교점계산 실패")
            self.pts = self.uniq[:4]
        else:
            self.uniq = []
            for p in self.pts:
                if all(np.hypot(*(p-q)) > 3.0 for q in self.uniq):
                    self.uniq.append(p)
            self.pts = self.uniq[:4]

        # 12. 모서리 순서 정렬 (좌상-우상-우하-좌하)
        box = self.order_corners(self.pts)

        # 13. 결과 시각화 (외곽선/군집 직선/교차점)
        for L in [self.L0a, self.L0b, self.L1a, self.L1b]:
            self.overlay_lines(L)
        for L in self.fam0:
            self.overlay_lines(L,(0,255,0))
        for L in self.fam1:
            self.overlay_lines(L,(0,0,255))
        for name, pt in zip(['tl', 'tr', 'br', 'bl'], self.pts):
            pt_int = tuple(map(int, pt))
            cv2.circle(self.overlay, pt_int, 6, (0, 255, 0), -1)
            cv2.putText(self.overlay, name, (pt_int[0]+6, pt_int[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 14. 결과 출력
        cv2.imshow("overlay", self.overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ------------------ 유틸리티 함수 ------------------

    def overlay_lines(self,L,color=(0,0,255)):
        """(A,B,C) 직선 방정식으로 이미지 경계까지 직선을 그림"""
        a,b,c = L
        H, W = self.img.shape[:2]
        if abs(b) > 1e-6:
            y0 = int(-c/b)
            yW = int((-a*W - c)/b)
            cv2.line(self.overlay, (0, y0), (W, yW), color, 2)
        else:
            x = int(-c/a)
            cv2.line(self.overlay, (x, 0), (x, H), color, 2)

    def classify_angles(self, angles):
        """각도 배열을 받아 직접 분류: 0~π/4 또는 3π/4~π 범위면 label=0, 그 외는 label=1"""
        labels = []
        for ang in angles:
            if (0 <= ang < (np.pi/4)) or ((3*np.pi/4) <= ang < np.pi):
                labels.append(0)
            else:
                labels.append(1)
        return np.array(labels)
     
    def pick_outer_two_lines(self, lines_family, img_shape):
        """
        평행선 집합에서 가장 바깥쪽 2개 직선을 선택
        (이미지 중심 기준 거리값 rho_signed 최소/최대)
        """
        H, W = img_shape[:2]
        mcx, mcy = W/2.0, H/2.0

        records = []
        for (a,b,c) in lines_family:
            rho_signed = a*mcx + b*mcy + c
            records.append((rho_signed, (a,b,c)))

        if len(records) < 2:
            return None

        records.sort(key=lambda x: x[0])
        outer_lines = [records[0][1], records[-1][1]]
        return outer_lines[0], outer_lines[1]
    
    def intersect_lines(self, L1, L2):
        """두 직선(Ax+By+C=0) 교차점 계산"""
        a1,b1,c1 = L1
        a2,b2,c2 = L2
        d = a1*b2 - a2*b1
        if abs(d) < 1e-8: 
            return None
        x = (b1*c2 - b2*c1)/d
        y = (c1*a2 - c2*a1)/d
        return np.array([x, y], dtype=np.float32)

    def order_corners(self, pts):
        """네 꼭짓점을 좌상-우상-우하-좌하 순서로 정렬"""
        pts = np.array(pts)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()

        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    
    def rho_theta_to_abc(self, rho, theta):
        """허프 직선 (rho, theta) → 일반형 Ax+By+C=0 변환"""
        a = np.cos(theta)
        b = np.sin(theta)
        c = -rho
        return (a, b, c)

# 실행
court_detection()(image_path)
