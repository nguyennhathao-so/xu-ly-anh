import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from recognition import DigitRecognizer
from preprocessing import preprocess_image
from segmentation import segment_digits

class BarcodeDigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống Nhận dạng Số dưới Mã vạch - Step by Step")
        self.root.geometry("1100x700")

        self.filepath = None
        self.original_image = None
        self.processed_image_step1 = None
        
        # Khởi tạo module nhận dạng
        self.recognizer = DigitRecognizer()

        # ===== Tạo Canvas cuộn TOÀN BỘ giao diện =====
        main_canvas = tk.Canvas(self.root)
        vscroll = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=main_canvas.yview)
        main_canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame nội dung thực sự nằm bên trong canvas
        self.main_frame = tk.Frame(main_canvas)
        main_canvas_window = main_canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

        def on_main_frame_configure(event):
            main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        self.main_frame.bind("<Configure>", on_main_frame_configure)

        # Cuộn bằng chuột (mouse wheel)
        def on_mousewheel(event):
            main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        main_canvas.bind_all("<MouseWheel>", on_mousewheel)

        # ===== Từ đây tất cả widget dùng self.main_frame thay vì self.root =====

        # Khung chứa nút bấm
        btn_frame = tk.Frame(self.main_frame)
        btn_frame.pack(pady=10)

        self.btn_load = tk.Button(btn_frame, text="1. Tải ảnh", command=self.load_image, font=("Arial", 12), bg="#4CAF50", fg="white")
        self.btn_load.grid(row=0, column=0, padx=10)

        self.btn_step1 = tk.Button(btn_frame, text="2. Chạy Bước 1 (Tiền xử lý)", command=self.run_step1, font=("Arial", 12), state=tk.DISABLED)
        self.btn_step1.grid(row=0, column=1, padx=10)
        
        self.btn_step3 = tk.Button(btn_frame, text="3. Chạy Bước 3 (Cắt chữ)", command=self.run_step3, font=("Arial", 12), state=tk.DISABLED)
        self.btn_step3.grid(row=0, column=2, padx=10)

        self.btn_step4 = tk.Button(btn_frame, text="4. Nhận dạng số", command=self.run_step4, font=("Arial", 12), bg="#2196F3", fg="white", state=tk.DISABLED)
        self.btn_step4.grid(row=0, column=3, padx=10)

        # Khung chứa ảnh
        self.img_frame = tk.Frame(self.main_frame)
        self.img_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Cột ảnh gốc
        self.frame_orig = tk.Frame(self.img_frame, width=500, height=450)
        self.frame_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.frame_orig.pack_propagate(False)
        tk.Label(self.frame_orig, text="Ảnh gốc", font=("Arial", 12, "bold")).pack()
        self.lbl_orig = tk.Label(self.frame_orig, bg="#e0e0e0")
        self.lbl_orig.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Cột ảnh xử lý
        self.frame_proc = tk.Frame(self.img_frame, width=500, height=450)
        self.frame_proc.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.frame_proc.pack_propagate(False)
        tk.Label(self.frame_proc, text="Ảnh sau xử lý (Bước 1)", font=("Arial", 12, "bold")).pack()
        self.lbl_proc = tk.Label(self.frame_proc, bg="#e0e0e0")
        self.lbl_proc.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.lbl_info = tk.Label(
            self.frame_proc, text="", fg="blue", font=("Arial", 10),
            justify=tk.LEFT, anchor="w", wraplength=480
        )
        self.lbl_info.pack(fill=tk.X, padx=8, pady=5)

        
        # Khung chứa kết quả Bước 3 - có Scrollbar ngang
        self.frame_step3 = tk.Frame(self.main_frame, height=160)
        self.frame_step3.pack(fill=tk.X, padx=20, pady=10)
        self.frame_step3.pack_propagate(False)
        tk.Label(self.frame_step3, text="Kết quả Bước 3 (Các chữ số được cắt rời):", font=("Arial", 12, "bold")).pack()
        
        # Canvas + Scrollbar ngang để cuộn khi có nhiều chữ số
        canvas_frame = tk.Frame(self.frame_step3)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.digits_canvas = tk.Canvas(canvas_frame, height=110, bg="#f5f5f5")
        scrollbar_x = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.digits_canvas.xview)
        self.digits_canvas.configure(xscrollcommand=scrollbar_x.set)
        
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.digits_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Frame bên trong Canvas chứa các ảnh chữ số
        self.digits_frame = tk.Frame(self.digits_canvas, bg="#f5f5f5")
        self.digits_canvas.create_window((0, 0), window=self.digits_frame, anchor="nw")
        
        def on_digits_configure(event):
            self.digits_canvas.configure(scrollregion=self.digits_canvas.bbox("all"))
        self.digits_frame.bind("<Configure>", on_digits_configure)
        
        self.digit_labels = []

        # Khu vực hiển thị kết quả nhận dạng cuối cùng
        result_outer = tk.Frame(self.main_frame, bg="#1a1a2e", pady=12)
        result_outer.pack(fill=tk.X, padx=20, pady=(5, 15))
        tk.Label(result_outer, text="Kết quả Nhận dạng (Bước 4):",
                 font=("Arial", 12, "bold"), bg="#1a1a2e", fg="#aaaacc").pack()
        self.lbl_result = tk.Label(result_outer, text="— chưa nhận dạng —",
                                   font=("Courier New", 26, "bold"),
                                   bg="#1a1a2e", fg="#00e5ff", pady=8)
        self.lbl_result.pack()
        self.lbl_method = tk.Label(result_outer, text="",
                                   font=("Arial", 9), bg="#1a1a2e", fg="#888899")
        self.lbl_method.pack()

    def load_image(self):
        self.filepath = filedialog.askopenfilename(
            title="Chọn ảnh mã vạch",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )
        if self.filepath:
            # Đọc ảnh bằng cv2
            self.original_image = cv2.imread(self.filepath)
            if self.original_image is None:
                messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
                return
            
            # Hiển thị ảnh gốc
            self.display_image(self.original_image, self.lbl_orig)
            self.btn_step1.config(state=tk.NORMAL)
            
            # Reset phần xử lý
            self.lbl_proc.config(image='')
            self.lbl_info.config(text="")
            self.btn_step3.config(state=tk.DISABLED)
            for lbl in self.digit_labels:
                lbl.destroy()
            self.digit_labels.clear()

    def display_image(self, cv_img, label_widget):
        # Resize để hiển thị vừa khung hình (tối đa 480x480)
        h, w = cv_img.shape[:2]
        max_size = 480
        scale = min(max_size/w, max_size/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img_resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Chuyển BGR sang RGB nếu là ảnh màu
        if len(img_resized.shape) == 3:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_resized # ảnh xám
            
        # Convert sang format tkinter dùng được
        pil_img = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        
        label_widget.imgtk = imgtk
        label_widget.config(image=imgtk)

    def run_step1(self):
        if self.original_image is None:
            return

        # Gọi hàm tiền xử lý từ module preprocessing.py
        thresh_final, info_text = preprocess_image(self.original_image)
        
        self.processed_image_step1 = thresh_final
        self.btn_step3.config(state=tk.NORMAL)
        
        # Hiển thị kết quả lên giao diện
        self.display_image(self.processed_image_step1, self.lbl_proc)
        self.lbl_info.config(text=info_text)

    def run_step3(self):
        if self.processed_image_step1 is None:
            return
            
        # Xóa các ảnh cũ trên UI
        for lbl in self.digit_labels:
            lbl.destroy()
        self.digit_labels.clear()
        
        # Gọi hàm cắt chữ số từ module segmentation.py
        self.segmented_digits = segment_digits(self.processed_image_step1)
        
        # Hiển thị từng chữ số lên UI
        for digit_img in self.segmented_digits:
            # Resize để hiển thị lên Giao diện (Cố định chiều cao = 50px)
            h_d, w_d = digit_img.shape
            scale = 50.0 / h_d if h_d > 0 else 1
            new_w, new_h = int(w_d * scale), 50
            if new_w > 0 and new_h > 0:
                resized = cv2.resize(digit_img, (new_w, new_h))
                # Convert sang RGB để hiển thị trên Tkinter
                rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                pil_img = Image.fromarray(rgb)
                imgtk = ImageTk.PhotoImage(image=pil_img)
                
                # Tạo Label chứa ảnh chữ số
                lbl = tk.Label(self.digits_frame, image=imgtk, bg="black", bd=1, relief="solid")
                lbl.image = imgtk # Giữ tham chiếu để không bị garbage collected
                lbl.pack(side=tk.LEFT, padx=5, pady=10)
                self.digit_labels.append(lbl)
        
        self.btn_step4.config(state=tk.NORMAL)

    # =====================================================
    # BƯỚC 4: NHẬN DẠNG CHỮ SỐ (Chương 3 & 5)
    # =====================================================

    def run_step4(self):
        if not hasattr(self, 'segmented_digits') or not self.segmented_digits:
            messagebox.showwarning("Chú ý", "Vui lòng chạy Bước 3 trước!")
            return

        self.lbl_result.config(text="Đang nhận dạng...", fg="#ffcc00")
        self.root.update_idletasks()

        result_str    = ""
        count = 0

        for seg_img in self.segmented_digits:
            digit_val, method = self.recognizer.recognize_one_digit(seg_img)
            result_str += str(digit_val)
            count += 1

        self.lbl_result.config(text=result_str, fg="#00e5ff")
        self.lbl_method.config(
            text=f"Thuật toán: Hu Moments + Fourier + MLP (Deep Learning) | Đã nhận dạng: {count} ký tự"
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = BarcodeDigitApp(root)
    root.mainloop()
