import cv2
import numpy as np

def deskew_image(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 1. Sử dụng Canny để tìm cạnh (tốt hơn chỉ dùng threshold)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 2. HoughLines tìm các đường thẳng
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    if lines is None:
        return image

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = (theta * 180 / np.pi)
        
        # 3. Lọc góc: Chúng ta chỉ quan tâm các đường gần ngang (80-100 độ) 
        # hoặc gần dọc (0-10 hoặc 170-180 độ)
        if 80 < angle < 100:
            angles.append(angle - 90)
        elif angle < 10:
            angles.append(angle)
        elif angle > 170:
            angles.append(angle - 180)

    if not angles:
        return image

    # 4. Lấy trung vị để loại bỏ nhiễu
    median_angle = np.median(angles)
    
    if abs(median_angle) < 0.1: # Quá nhỏ thì không cần xoay
        return image

    # 5. Xoay ảnh
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    
    # Sử dụng BORDER_CONSTANT với màu trắng (hoặc đen) thay vì REPLICATE để ảnh sạch hơn
    deskewed = cv2.warpAffine(image, rotation_matrix, (w, h), 
                              flags=cv2.INTER_CUBIC, 
                              borderMode=cv2.BORDER_CONSTANT, 
                              borderValue=(255, 255, 255))
    return deskewed

def preprocess_image(original_image):
    """
    Tiền xử lý và trích xuất vùng chứa số từ ảnh mã vạch (Bước 1) - Đã sửa lỗi mất đầu số.
    """
    # 1. Chuyển sang ảnh xám
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # 1.5. Nắn thẳng ảnh
    gray = deskew_image(gray)
    
    # 2. Tìm viền chứa nội dung
    _, thresh_basic = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh_basic)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped_content = gray[y:y+h, x:x+w]
        cropped_thresh = thresh_basic[y:y+h, x:x+w]
    else:
        cropped_content = gray
        cropped_thresh = thresh_basic
        h = gray.shape[0]
        
    # 3. Cắt vùng chứa số bằng Horizontal Projection (CẢI TIẾN)
    row_sums = np.sum(cropped_thresh, axis=1)
    
    start_search = int(h * 0.4) # Mở rộng vùng tìm kiếm lên 40% thay vì 50%
    end_search = int(h * 0.9)
    
    if end_search > start_search:
        # Tìm vị trí khe hở giữa mã vạch và số
        gap_local_y = np.argmin(row_sums[start_search:end_search])
        gap_y = start_search + gap_local_y
        
        # --- ĐOẠN SỬA ĐỔI CHÍNH ---
        # Lùi điểm cắt lên trên 5-10 pixel để không chạm vào đầu số
        padding = 10 
        gap_y = max(0, gap_y - padding) 
        # --------------------------
    else:
        gap_y = int(h * 0.6) 
        
    number_roi = cropped_content[gap_y:h, :]
    
    # 4. Phóng to (Scale)
    scale_factor = 3.0
    zoomed_roi = cv2.resize(number_roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # 5. Cân bằng tương phản và Nhị phân hóa
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(zoomed_roi)
    
    _, thresh_final = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    info_text = ("Phương pháp Projection (Đã fix mất đầu số):\n"
                 "1. Thêm Padding: Đẩy vị trí cắt lên trên 10px để bảo vệ phần trên của chữ số.\n"
                 "2. Mở rộng Search Window: Tìm kiếm khe hở từ 40% chiều cao ảnh.\n"
                 "3. Nhị phân hóa: CLAHE + Otsu giúp chữ số tách bạch khỏi nền nhiễu.")
                 
    return thresh_final, info_text