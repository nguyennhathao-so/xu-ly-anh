import sys
import os
import time

# Thêm đường dẫn thư mục gốc vào sys.path để có thể import các module bên ngoài
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

from preprocessing import preprocess_image
from segmentation import segment_digits
from recognition import normalize_digit_img

class DataCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Công cụ Thu thập Mẫu Dữ Liệu (Data Collection Tool)")
        self.root.geometry("800x400")

        self.segmented_digits = []
        self.digit_labels = []
        self.templates_dir = os.path.join(current_dir, "digit_templates")

        # Nút tải ảnh
        self.btn_load = tk.Button(self.root, text="1. Tải ảnh mã vạch & Cắt chữ", command=self.process_image, font=("Arial", 12), bg="#4CAF50", fg="white")
        self.btn_load.pack(pady=10)

        # Khu vực hiển thị ảnh cắt
        self.frame_digits = tk.Frame(self.root, bg="#f5f5f5", height=100)
        self.frame_digits.pack(fill=tk.X, padx=10, pady=10)
        self.frame_digits.pack_propagate(False)

        # Khu vực nhập liệu
        self.frame_input = tk.Frame(self.root)
        self.frame_input.pack(pady=20)

        tk.Label(self.frame_input, text="Nhập dãy số (Tương ứng với ảnh trên):", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        self.entry_labels = tk.Entry(self.frame_input, font=("Courier New", 16), width=20)
        self.entry_labels.pack(side=tk.LEFT, padx=5)

        self.btn_save = tk.Button(self.frame_input, text="2. Lưu Mẫu", command=self.save_samples, font=("Arial", 12), bg="#2196F3", fg="white", state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=10)

        self.lbl_status = tk.Label(self.root, text="Sẵn sàng.", font=("Arial", 10), fg="gray")
        self.lbl_status.pack(pady=10)

    def process_image(self):
        filepath = filedialog.askopenfilename(
            title="Chọn ảnh mã vạch",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )
        if not filepath:
            return

        original_image = cv2.imread(filepath)
        if original_image is None:
            messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
            return

        # Dọn dẹp UI
        for lbl in self.digit_labels:
            lbl.destroy()
        self.digit_labels.clear()
        self.segmented_digits.clear()
        self.entry_labels.delete(0, tk.END)

        # Tiền xử lý (Bước 1)
        thresh_final, _ = preprocess_image(original_image)

        # Cắt chữ (Bước 3)
        self.segmented_digits = segment_digits(thresh_final)

        if not self.segmented_digits:
            messagebox.showwarning("Cảnh báo", "Không tìm thấy chữ số nào trong ảnh!")
            self.btn_save.config(state=tk.DISABLED)
            return

        # Hiển thị lên UI
        for digit_img in self.segmented_digits:
            h_d, w_d = digit_img.shape
            scale = 50.0 / h_d if h_d > 0 else 1
            new_w, new_h = int(w_d * scale), 50
            if new_w > 0 and new_h > 0:
                resized = cv2.resize(digit_img, (new_w, new_h))
                rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                pil_img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=pil_img)

                lbl = tk.Label(self.frame_digits, image=imgtk, bg="black", bd=1, relief="solid")
                lbl.image = imgtk
                lbl.pack(side=tk.LEFT, padx=5, pady=5)
                self.digit_labels.append(lbl)

        self.btn_save.config(state=tk.NORMAL)
        self.lbl_status.config(text=f"Đã cắt được {len(self.segmented_digits)} ảnh chữ số. Vui lòng nhập nhãn và lưu.")

    def save_samples(self):
        labels_str = self.entry_labels.get().strip()
        
        if len(labels_str) != len(self.segmented_digits):
            messagebox.showerror("Lỗi", f"Số lượng ký tự nhập vào ({len(labels_str)}) KHÔNG KHỚP với số ảnh cắt được ({len(self.segmented_digits)}).")
            return

        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir)

        count = 0
        timestamp = int(time.time())
        
        for i, char in enumerate(labels_str):
            if char.isdigit():
                digit_img = self.segmented_digits[i]
                # Chuẩn hóa về 50x50 nhưng GIỮ NGUYÊN TỶ LỆ trước khi lưu
                norm_img = normalize_digit_img(digit_img, 50)
                
                filename = f"digit_{char}_{timestamp}_{i}.png"
                filepath = os.path.join(self.templates_dir, filename)
                
                # Xử lý lỗi thư mục có dấu Tiếng Việt (xử lý ảnh) khiến cv2.imwrite chạy ngầm thất bại
                is_success, im_buf_arr = cv2.imencode(".png", norm_img)
                if is_success:
                    im_buf_arr.tofile(filepath)
                count += 1
                
        messagebox.showinfo("Thành công", f"Đã lưu {count} mẫu ảnh vào thư mục 'data/digit_templates'.")
        self.lbl_status.config(text=f"Đã lưu {count} mẫu mới. Hãy chọn ảnh khác để tiếp tục thu thập.")
        self.btn_save.config(state=tk.DISABLED)
        self.entry_labels.delete(0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectorApp(root)
    root.mainloop()
