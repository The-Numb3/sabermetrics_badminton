import cv2
import numpy as np



# 이미지 파일을 불러옵니다.
image_path = './frames/frame_0.jpg'  


class court_detection:
    def __call__(self, path):
        self.image_path = path
        self.img = cv2.imread(self.image_path)
        if self.img is None:
            raise FileNotFoundError(self.image_path)
        
        #리사이징
        self.h, self.w = self.img.shape[0:2]
        self.scale = 1.0
        self.max_size = 1600
        if self.w > self.max_size:
            self.scale = self.max_size / self.w
            self.img = cv2.resize(self.img, (int(self.w*self.scale), int(self.scale*self.h)), interpolation=cv2.INTER_AREA)
            self.h, self.w = self.img.shape[0:2]
        
        self.vis = self.img.copy()
        self.overlay = self.vis.copy()

        #1 흰색 선 분리
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.lower = np.array([0, 0, 180], dtype=np.uint8)
        self.upper = np.array([180, 60, 255], dtype=np.uint8)
        self.mask = cv2.inRange(self.hsv, self.lower, self.upper)
        
        #명암대비향상 + 조금 더 탄탄한 비교
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        #밝은 선 강조
        self.enhanced = cv2.addWeighted(self.gray, 0.5, cv2.normalize(self.gray, None, 0, 255, cv2.NORM_MINMAX), 0.5, 0)
        self.mask2 = cv2.threshold(self.enhanced, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        self.mask = cv2.bitwise_and(self.mask, self.mask2)

        # 모폴로지로 구멍 메우기/노이즈 제거
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, self.kernel, iterations=1)

        # 2) 엣지 & 허프 직선
        self.edges = cv2.Canny(self.mask, 50, 150, apertureSize=3, L2gradient=True)
        self.linesP = cv2.HoughLinesP(self.edges, 1, np.pi/180, threshold=120, minLineLength=int(min(self.h,self.w)*0.25), maxLineGap=20)

        if self.linesP is None or len(self.linesP) < 4:
            raise RuntimeError("충분한 직선을 찾지 못했습니다. 조명/대비/파라미터를 조정하세요.")

        #3) 각도 계산 & 군집화
        self.segs = []
        self.angles = []
        for l in self.linesP[:,0,:]:
            x1,y1,x2,y2 = l
            ang = self.angle_of_segment((x1,y1),(x2,y2))
            length = np.hypot(x2-x1, y2-y1)
            if length < min(self.h,self.w)*0.2:
                continue
            self.segs.append(((x1,y1),(x2,y2)))
            self.angles.append(ang)
        
        if len(self.segs) < 4:
            raise RuntimeError("유효한 긴 직선이 부족합니다.")
        
        self.angles = np.array(self.angles, dtype=np.float32)
        self.labels = self.kmeans_angles(self.angles)

        self.fam0 = []
        self.fam1 = []

        for (p1, p2), label in zip(self.segs, self.labels):
            L = self.line_from_segment(p1, p2)
            if L is None:
                continue
            if label == 0:
                self.fam0.append(L)
            else:
                self.fam1.append(L)

        #각 군집에서 바깥 2개 선 뽑기
        self.pick0 = self.pick_outer_two_lines(self.fam0, self.img.shape) if len(self.fam0) >= 2 else None
        self.pick1 = self.pick_outer_two_lines(self.fam1, self.img.shape) if len(self.fam1) >= 2 else None

        (self.L0a, self.L0b) = self.pick0
        (self.L1a, self.L1b) = self.pick1

        self.pts = []
        for La in [self.L0a, self.L0b]:
            for Lb in [self.L1a, self.L1b]:
                self.pt = self.intersect_lines(La, Lb)
                if self.pt is not None and 0 <= self.pt[0] < self.w and 0 <= self.pt[1] < self.h:
                    self.pts.append(self.pt)

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

        box = self.order_corners(self.pts)
        # 직선 그리기
        for L in [self.L0a, self.L0b, self.L1a, self.L1b]:
            self.overlay_lines(L)
        
        for L in self.fam0:
            self.overlay_lines(L,(0,255,0))
        for L in self.fam1:
            self.overlay_lines(L,(0,0,255))
        # 모서리 점 표시
        for name, pt in zip(['tl', 'tr', 'br', 'bl'], self.pts):
            pt_int = tuple(map(int, pt))
            cv2.circle(self.overlay, pt_int, 6, (0, 255, 0), -1)
            cv2.putText(self.overlay, name, (pt_int[0]+6, pt_int[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("overlay", self.overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #오버레이 이미지 출력하는 함수
    def overlay_lines(self,L,color=(0,0,255)):
        if self.linesP is not None:
            for line in self.linesP:
                x1, y1, x2, y2 = line[0]
                cv2.line(self.overlay, (x1, y1), (x2, y2), color, 2)

        a,b,c = L
        H, W = self.img.shape[:2]
        if abs(b) > 1e-6:
            y0 = int(-c/b)
            yW = int((-a*W - c)/b)
            cv2.line(self.overlay, (0, y0), (W, yW), (255, 0, 0), 2)
        else:
            x = int(-c/a)
            cv2.line(self.overlay, (x, 0), (x, H), (255, 0, 0), 2)

    
    def angle_of_segment(self, p1, p2):
        """두 점을 잇는 선분의 각도를 계산합니다. (0~180도)"""
        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0:
            angle += 180
        return angle
    
    def kmeans_angles(self, angles):
        """각도 배열을 받아 두 개의 군집으로 나눕니다."""
        angles = angles.reshape(-1, 1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, _ = cv2.kmeans(angles.astype(np.float32), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return labels.flatten()
    
    def line_from_segment(self, p1, p2):
        """두 점을 잇는 직선의 일반형 방정식 계수를 반환합니다: Ax + By + C = 0"""
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2*y1 - x1*y2
        norm = np.hypot(A, B)
        if norm < 1e-8:
            return None
        if A < 0 or (abs(A) < 1e-6 and B < 0):
            A, B, C = -A, -B, -C
        return (A/norm, B/norm, C/norm)
    
    def pick_outer_two_lines(self, lines_family, img_shape):
        """
        같은 방향(대략 평행) 선들의 집합에서 이미지 바깥쪽 경계 2개를 선택.
        방법: 각 선의 rho(원점으로부터 거리)를 모아 최소/최대를 고른다.
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
        a1,b1,c1 = L1
        a2,b2,c2 = L2
        d = a1*b2 - a2*b1
        if abs(d) < 1e-8: 
            return None
        x = (b1*c2 - b2*c1)/d
        y = (c1*a2 - c2*a1)/d
        return np.array([x, y], dtype=np.float32)

    def order_corners(self, pts):
        """코트의 네 모서리를 좌상, 우상, 우하, 좌하 순서로 정렬합니다."""
        pts = np.array(pts)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()

        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

court_detection()(image_path)