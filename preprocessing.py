import cv2
import numpy as np

def preprocess_image(original_image):
    """
    Tiền xử lý và trích xuất vùng chứa số từ ảnh mã vạch (Bước 1).
    """
    # 1. Chuyển sang ảnh xám
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # 2. Tìm viền chứa toàn bộ nội dung (loại bỏ lề trắng thừa)
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
        
    # 3. Cắt vùng chứa số bằng Horizontal Projection (Chiếu ngang)
    row_sums = np.sum(cropped_thresh, axis=1)
    
    start_search = int(h * 0.5)
    end_search = int(h * 0.95)
    
    if end_search > start_search:
        gap_local_y = np.argmin(row_sums[start_search:end_search])
        gap_y = start_search + gap_local_y
    else:
        gap_y = int(h * 0.7) 
        
    number_roi = cropped_content[gap_y:h, :]
    
    # 4. BIẾN ĐỔI HÌNH HỌC (Chương 3) - Phóng to (Scale)
    scale_factor = 3.0
    zoomed_roi = cv2.resize(number_roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # 5. Cân bằng tương phản và Nhị phân hóa (Chương 2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(zoomed_roi)
    
    _, thresh_final = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    info_text = ("Phương pháp Projection (Chiếu ngang):\n"
                 "1. Crop linh hoạt: Dò tìm hàng ngang có ít mực nhất (khoảng trắng) để cắt bỏ sạch sẽ mã vạch.\n"
                 "2. Biến đổi hình học (Ch. 3): Phóng to 3x (Bicubic).\n"
                 "3. Nhị phân hóa: CLAHE + Otsu (Không còn vạch đen dính vào).")
                 
    return thresh_final, info_text
