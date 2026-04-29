import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk

# Tắt cảnh báo font của matplotlib
import warnings
warnings.filterwarnings("ignore")

def extract_features_visual(img_bin):
    """
    Trích xuất và trả về riêng biệt các đặc trưng để vẽ biểu đồ
    """
    # 1. Hu Moments
    moments = cv2.moments(img_bin)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = np.zeros_like(hu)
    for i, h in enumerate(hu):
        if abs(h) > 1e-10:
            hu_log[i] = -np.sign(h) * np.log10(abs(h))
            
    # 2. Fourier Descriptors
    fourier_features = np.zeros(32, dtype=np.float32)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_to_draw = None
    if contours:
        contour_to_draw = max(contours, key=cv2.contourArea)
        pts = contour_to_draw[:, 0, :]
        if len(pts) >= 16:
            indices = np.linspace(0, len(pts) - 1, 128, dtype=int)
            sampled = pts[indices].astype(np.float64)
            z = sampled[:, 0] + 1j * sampled[:, 1]
            Z = np.fft.fft(z)
            Z_freq = Z[1:33]
            magnitudes = np.abs(Z_freq)
            if magnitudes[0] > 1e-10:
                fourier_features[:len(magnitudes)] = magnitudes / magnitudes[0]

    return hu_log, fourier_features, contour_to_draw

def main():
    # Chọn file ảnh từ thư mục mẫu
    root = Tk()
    root.withdraw() # Ẩn cửa sổ chính
    current_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(current_dir, "data", "digit_templates")
    
    filepath = filedialog.askopenfilename(
        initialdir=templates_dir,
        title="Chọn 1 ảnh mẫu trong thư mục digit_templates",
        filetypes=(("PNG files", "*.png"), ("All files", "*.*"))
    )
    
    if not filepath:
        print("Đã hủy chọn ảnh.")
        return
        
    # Đọc ảnh với unicode path
    img_data = np.fromfile(filepath, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Lỗi đọc ảnh!")
        return
        
    _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    hu_log, fourier_features, contour = extract_features_visual(img_bin)
    
    # Tạo ảnh có vẽ viền Contour
    img_color = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)
    if contour is not None:
        cv2.drawContours(img_color, [contour], -1, (255, 0, 0), 1)

    # === VẼ BIỂU ĐỒ ===
    plt.figure(figsize=(14, 4))
    
    # 1. Ảnh gốc + Contour
    plt.subplot(1, 3, 1)
    plt.title("Anh chu so & Contour")
    plt.imshow(img_color)
    plt.axis("off")
    
    # 2. Hu Moments
    plt.subplot(1, 3, 2)
    plt.title("7 Hu Moments (Log Scale)")
    x_hu = np.arange(1, 8)
    plt.bar(x_hu, hu_log, color='orange')
    plt.xticks(x_hu, [f"h{i}" for i in x_hu])
    plt.axhline(0, color='black', linewidth=1)
    
    # 3. Fourier Descriptors
    plt.subplot(1, 3, 3)
    plt.title("32 Fourier Descriptors (Bien do)")
    x_fourier = np.arange(1, 33)
    plt.bar(x_fourier, fourier_features, color='green')
    plt.xlim(0, 33)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
