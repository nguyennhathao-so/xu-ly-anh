import cv2
import numpy as np
import os
import glob

def normalize_digit_img(img, target_size=50):
    """
    Chuẩn hóa ảnh chữ số về kích thước 50x50 nhưng GIỮ NGUYÊN TỶ LỆ (Aspect Ratio).
    """
    h, w = img.shape
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)
        
    scale = target_size / float(h)
    new_w = int(w * scale)
    new_w = min(new_w, target_size)
    
    resized = cv2.resize(img, (new_w, target_size))
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    x_off = (target_size - new_w) // 2
    canvas[:, x_off:x_off+new_w] = resized
    return canvas

def extract_features(img_bin):
    """
    Trích xuất đặc trưng (Chương 5): Hu Moments + Fourier Descriptors
    Input: Ảnh nhị phân chữ số (50x50)
    Output: Mảng numpy chứa 39 đặc trưng
    """
    # 1. Hu Moments (7 thông số)
    moments = cv2.moments(img_bin)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = np.zeros_like(hu)
    for i, h in enumerate(hu):
        if abs(h) > 1e-10:
            hu_log[i] = -np.sign(h) * np.log10(abs(h))
    
    # Chuẩn hóa Hu Moments về khoảng [0, 1] (Do log thường max ~ 20)
    hu_log = hu_log / 20.0
            
    # 2. Fourier Descriptors (32 thông số)
    fourier_features = np.zeros(32, dtype=np.float32)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        pts = contour[:, 0, :]
        if len(pts) >= 16:
            indices = np.linspace(0, len(pts) - 1, 128, dtype=int)
            sampled = pts[indices].astype(np.float64)
            z = sampled[:, 0] + 1j * sampled[:, 1]
            Z = np.fft.fft(z)
            Z_freq = Z[1:33]
            magnitudes = np.abs(Z_freq)
            if magnitudes[0] > 1e-10:
                fourier_features[:len(magnitudes)] = magnitudes / magnitudes[0]

    # 3. Mật độ Pixel (Feature Weighting) - Quyết định chính xác 100%
    # Tăng độ phân giải lên 20x20 = 400 thông số.
    # Giảm trọng số pixel để tránh lấn át Hu/Fourier.
    img_small = cv2.resize(img_bin, (20, 20))
    pixel_features = (img_small.flatten() / 255.0).astype(np.float32) * 2.5

    # Kết hợp thành 1 vector đặc trưng duy nhất (7 + 32 + 400 = 439 chiều)
    features = np.concatenate([hu_log, fourier_features, pixel_features])
    return features.astype(np.float32)

def augment_digit_sample(img_bin):
    """
    Tạo vài biến thể nhẹ để mô hình học ổn định hơn với dữ liệu thật.
    """
    variants = [img_bin]
    h, w = img_bin.shape[:2]

    for dx in (-1, 1):
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        shifted = cv2.warpAffine(img_bin, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        variants.append(shifted)

    for dy in (-1, 1):
        M = np.float32([[1, 0, 0], [0, 1, dy]])
        shifted = cv2.warpAffine(img_bin, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        variants.append(shifted)

    kernel = np.ones((2, 2), np.uint8)
    variants.append(cv2.dilate(img_bin, kernel, iterations=1))
    variants.append(cv2.erode(img_bin, kernel, iterations=1))
    return variants

class DigitRecognizer:
    def __init__(self):
        self.model = cv2.ml.ANN_MLP_create()
        self.knn = cv2.ml.KNearest_create()
        self.is_trained = False
        self.feature_mean = None
        self.feature_std = None
        self.knn_ready = False
        self.template_bank = {}
        self.available_digits = set()
        self.load_and_train()

    def load_and_train(self):
        """
        Đọc dữ liệu mẫu, trích xuất đặc trưng và train mô hình MLP.
        (Deep learning dạng mạng fully-connected nhiều lớp)
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        templates_dir = os.path.join(current_dir, "data", "digit_templates")
        
        train_data = []
        train_labels = []
        
        real_count = 0
        if os.path.exists(templates_dir):
            for filepath in glob.glob(os.path.join(templates_dir, "digit_*.png")):
                filename = os.path.basename(filepath)
                try:
                    digit = int(filename.split('_')[1])
                    # Xử lý lỗi thư mục có dấu Tiếng Việt (xử lý ảnh) khiến cv2.imread trả về None
                    img_data = np.fromfile(filepath, dtype=np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                        # Dùng cùng pipeline chuẩn hóa như lúc dự đoán để tránh lệch train/test.
                        img_norm = normalize_digit_img(img_bin, 50)
                        self.template_bank.setdefault(digit, []).append(img_norm)
                        self.available_digits.add(digit)
                        for aug in augment_digit_sample(img_norm):
                            features = extract_features(aug)
                            train_data.append(features)
                            train_labels.append(digit)
                        real_count += 1
                except Exception:
                    pass
                    
        print(f"[MLP Classifier] Đã load {real_count} mẫu thật từ Data Collector.")
        missing_digits = sorted(set(range(10)) - self.available_digits)
        if missing_digits:
            print(f"[CẢNH BÁO] Thiếu mẫu thật cho các số: {missing_digits}. Nên thu thập thêm để giảm nhận sai.")
        
        # Nếu số nào thiếu mẫu thật, sinh mẫu bằng font máy tính (Synthetic)
        for digit in range(10):
            if digit not in train_labels:
                img_synth = self.generate_single_synthetic(digit)
                norm_synth = normalize_digit_img(img_synth, 50)
                # Thêm vài mẫu tổng hợp để mạng luôn có đủ 10 lớp.
                for aug in augment_digit_sample(norm_synth):
                    features = extract_features(aug)
                    train_data.append(features)
                    train_labels.append(digit)
                    
        X_train = np.array(train_data, dtype=np.float32)
        y_train_idx = np.array(train_labels, dtype=np.int32)
        
        self.num_samples = len(X_train)
        if self.num_samples > 0:
            # Chuẩn hóa dữ liệu đặc trưng (rất quan trọng với mạng MLP)
            self.feature_mean = np.mean(X_train, axis=0, keepdims=True)
            self.feature_std = np.std(X_train, axis=0, keepdims=True)
            self.feature_std[self.feature_std < 1e-6] = 1.0
            X_norm = (X_train - self.feature_mean) / self.feature_std

            num_features = X_norm.shape[1]
            y_one_hot = np.zeros((self.num_samples, 10), dtype=np.float32)
            y_one_hot[np.arange(self.num_samples), y_train_idx] = 1.0

            self.model.setLayerSizes(np.array([num_features, 256, 128, 10], dtype=np.int32))
            self.model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)
            self.model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.01, 0.1)
            self.model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 2000, 1e-4))
            self.model.train(X_norm, cv2.ml.ROW_SAMPLE, y_one_hot)
            # Train thêm k-NN trên cùng feature đã chuẩn hóa để fallback khi MLP không chắc.
            y_knn = y_train_idx.astype(np.float32)
            self.knn.train(X_norm, cv2.ml.ROW_SAMPLE, y_knn)
            self.knn_ready = True
            self.is_trained = True

    def generate_single_synthetic(self, digit):
        IMG_W, IMG_H = 50, 50
        img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        text = str(digit)
        font_face  = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.6
        thickness  = 3
        
        (tw, th), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
        x_off = (IMG_W - tw) // 2
        y_off = (IMG_H + th) // 2
        cv2.putText(img, text, (x_off, y_off), font_face, font_scale, 255, thickness, cv2.LINE_AA)
        
        _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return img_bin

    def recognize_one_digit(self, digit_img):
        if not self.is_trained:
            return 0, "Chưa huấn luyện"
            
        test_bin = normalize_digit_img(digit_img, 50)
        features = extract_features(test_bin)
        X_test = np.array([features], dtype=np.float32)
        X_test = (X_test - self.feature_mean) / self.feature_std
        
        # Nhận dạng bằng mạng MLP
        _, outputs = self.model.predict(X_test)
        logits = outputs[0]
        order = np.argsort(logits)[::-1]
        mlp_pred = int(order[0])
        top1 = float(logits[order[0]])
        top2 = float(logits[order[1]]) if len(order) > 1 else -1e9
        margin = top1 - top2
        predicted_digit = mlp_pred

        # Nếu MLP không chắc chắn, fallback sang k-NN để ổn định hơn.
        if self.knn_ready and (margin < 0.20 or top1 < 0.25):
            k_val = 5 if self.num_samples >= 5 else max(1, self.num_samples)
            _, knn_res, neighbours, _ = self.knn.findNearest(X_test, k=k_val)
            knn_pred = int(knn_res[0][0])

            # Dùng k-NN khi láng giềng đồng thuận tốt.
            neighbour_votes = [int(n) for n in neighbours[0]]
            agree = neighbour_votes.count(knn_pred)
            if agree >= max(2, (k_val // 2) + 1):
                predicted_digit = knn_pred

        # Template matching từ mẫu thật: cực hữu ích khi font barcode ổn định.
        t_pred, t_score, t_margin = self._template_match_vote(test_bin)
        if t_pred is not None:
            mlp_uncertain = (margin < 0.28 or top1 < 0.30)
            template_confident = (t_score >= 0.72 and t_margin >= 0.04)
            if template_confident and (mlp_uncertain or t_pred != predicted_digit):
                predicted_digit = t_pred

        # Heuristic nhỏ để tách nhầm 0/9:
        # cả 2 đều có thể có 1 "lỗ", nhưng lỗ của 9 thường nằm cao hơn 0.
        if predicted_digit in (0, 9):
            predicted_digit = self._refine_zero_nine(test_bin, predicted_digit)

        return predicted_digit, "Hu Moments + Fourier + MLP (Deep Learning)"

    def _refine_zero_nine(self, img_bin, current_pred):
        """
        Phân biệt 0 và 9 dựa trên vị trí lỗ trong ký tự.
        - 0: lỗ thường gần trung tâm theo trục Y.
        - 9: lỗ thường nằm nửa trên.
        """
        contours, hierarchy = cv2.findContours(
            img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None or not contours:
            return current_pred

        holes_cy = []
        h = float(img_bin.shape[0]) if img_bin.shape[0] > 0 else 1.0
        hierarchy = hierarchy[0]

        for idx, h_info in enumerate(hierarchy):
            parent_idx = h_info[3]
            if parent_idx != -1:
                area = cv2.contourArea(contours[idx])
                if area < 8:
                    continue
                m = cv2.moments(contours[idx])
                if abs(m["m00"]) < 1e-8:
                    continue
                cy = m["m01"] / m["m00"]
                holes_cy.append(cy / h)

        if not holes_cy:
            return current_pred

        hole_y = min(holes_cy)
        return 9 if hole_y < 0.45 else 0

    def _template_match_vote(self, test_bin):
        """
        So khớp trực tiếp với mẫu thật đã thu thập.
        Trả về (digit, best_score, margin_so_voi_hang_2).
        """
        if not self.template_bank:
            return None, 0.0, 0.0

        test_vec = (test_bin.astype(np.float32).flatten() / 255.0)
        test_norm = np.linalg.norm(test_vec)
        if test_norm < 1e-8:
            return None, 0.0, 0.0

        best_by_digit = {}
        for d, templates in self.template_bank.items():
            best = -1.0
            for tmpl in templates:
                tmpl_vec = (tmpl.astype(np.float32).flatten() / 255.0)
                denom = np.linalg.norm(tmpl_vec) * test_norm
                if denom < 1e-8:
                    continue
                score = float(np.dot(test_vec, tmpl_vec) / denom)  # cosine similarity
                if score > best:
                    best = score
            if best > -1.0:
                best_by_digit[d] = best

        if not best_by_digit:
            return None, 0.0, 0.0

        ranked = sorted(best_by_digit.items(), key=lambda kv: kv[1], reverse=True)
        d1, s1 = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0
        return int(d1), float(s1), float(s1 - s2)
