import cv2
import numpy as np

def has_inner_hole(bin_img):
    """
    Kiểm tra nhanh ký tự có lỗ bên trong không (0/6/8/9 thường có).
    """
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return False

    hierarchy = hierarchy[0]
    for idx, h_info in enumerate(hierarchy):
        parent_idx = h_info[3]
        if parent_idx != -1:
            area = cv2.contourArea(contours[idx])
            if area >= 10:
                return True
    return False

def remove_barcode_bar_artifacts(bin_img):
    """
    Xóa các vạch mã vạch còn sót (thường là cột trắng rất cao, mảnh, bám phía trên).
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    cleaned = np.zeros_like(bin_img)
    h_img, w_img = bin_img.shape[:2]

    for label_idx in range(1, num_labels):
        x = stats[label_idx, cv2.CC_STAT_LEFT]
        y = stats[label_idx, cv2.CC_STAT_TOP]
        w = stats[label_idx, cv2.CC_STAT_WIDTH]
        h = stats[label_idx, cv2.CC_STAT_HEIGHT]
        area = stats[label_idx, cv2.CC_STAT_AREA]

        if area <= 0:
            continue

        aspect = h / float(max(w, 1))
        touches_top = y <= int(0.10 * h_img)
        thin_bar = aspect >= 5.5 and w <= max(4, int(0.02 * w_img))

        # Loại các vạch nhiễu rất cao/mảnh đi từ phía trên xuống.
        if touches_top and thin_bar:
            continue

        cleaned[labels == label_idx] = 255

    return cleaned

def keep_main_digit_band(bin_img):
    """
    Giữ lại dải hàng chứa chữ số chính, loại các đốm/vệt ở phía trên.
    """
    h, _ = bin_img.shape[:2]
    row_sums = np.sum(bin_img, axis=1)
    ink_rows = row_sums > (np.max(row_sums) * 0.08 if np.max(row_sums) > 0 else 1)

    bands = []
    in_band = False
    y_start = 0
    for y, has_ink in enumerate(ink_rows):
        if has_ink and not in_band:
            y_start = y
            in_band = True
        elif not has_ink and in_band:
            bands.append((y_start, y))
            in_band = False
    if in_band:
        bands.append((y_start, h))

    if not bands:
        return bin_img

    # Chọn band nhiều "mực" và đủ dày; ưu tiên band thấp hơn (vùng chữ số thường nằm thấp).
    best_band = bands[0]
    best_score = -1
    for y1, y2 in bands:
        band_h = max(1, y2 - y1)
        band_ink = np.sum(row_sums[y1:y2])
        center_bias = (y1 + y2) / 2.0 / max(h, 1)
        score = band_ink * (0.6 + 0.4 * (band_h / max(h, 1))) * (1.0 + 0.35 * center_bias)
        if score > best_score:
            best_score = score
            best_band = (y1, y2)

    y1, y2 = best_band
    pad = 2
    y1 = max(0, y1 - pad)
    y2 = min(h, y2 + pad)
    out = np.zeros_like(bin_img)
    out[y1:y2, :] = bin_img[y1:y2, :]
    return out

def get_normal_avg_width(segs):
    ws = sorted([e - s for s, e in segs])
    # Lấy 75% nhỏ nhất
    cutoff = max(1, int(len(ws) * 0.75))
    small_ws = ws[:cutoff]
    return sum(small_ws) / len(small_ws)

def split_one_segment(x_s, x_e, img, depth=0):
    """Tách đệ quy một segment dựa trên Vertical Projection nội bộ."""
    if depth > 3:  # Tránh đệ quy vô hạn
        return [(x_s, x_e)]
    seg_w = x_e - x_s
    region = img[:, x_s:x_e]
    col_s = np.sum(region, axis=0).astype(np.float64)
    
    # Tìm valley nhỏ nhất trong vùng giữa [20%, 80%]
    mid_s = max(1, seg_w // 5)
    mid_e = min(seg_w - 1, 4 * seg_w // 5)
    if mid_e <= mid_s:
        return [(x_s, x_e)]
    
    local_min_idx = np.argmin(col_s[mid_s:mid_e])
    split_col = mid_s + local_min_idx  # tọa độ tương đối trong segment
    
    # Chỉ cắt nếu valley thực sự thấp (< 30% max của segment)
    valley_val = col_s[split_col]
    max_val = np.max(col_s) if np.max(col_s) > 0 else 1
    if valley_val > max_val * 0.3:
        return [(x_s, x_e)]  # Không tìm thấy chỗ cắt rõ ràng
    
    # Tách thành 2 phần và kiểm tra từng phần có hợp lệ không
    left_end  = x_s + split_col
    right_start = x_s + split_col
    if left_end <= x_s + 3 or right_start >= x_e - 3:
        return [(x_s, x_e)]
    return [(x_s, left_end), (right_start, x_e)]

def split_by_morph_fallback(x_s, x_e, img):
    """
    Fallback tách segment dính bằng erosion dọc + connected components.
    Chỉ nhận khi tách ra đúng 2 phần hợp lệ để tránh tách sai.
    """
    region = img[:, x_s:x_e]
    if region.size == 0:
        return [(x_s, x_e)]

    h, w = region.shape[:2]
    if w < 10:
        return [(x_s, x_e)]

    # Erode nhẹ để "bẻ" chỗ dính mảnh giữa 2 chữ số.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    eroded = cv2.erode(region, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    if num_labels <= 2:
        return [(x_s, x_e)]

    # Lấy các component đủ lớn và có dạng chữ số.
    comps = []
    for label_idx in range(1, num_labels):
        cx = stats[label_idx, cv2.CC_STAT_LEFT]
        cy = stats[label_idx, cv2.CC_STAT_TOP]
        cw = stats[label_idx, cv2.CC_STAT_WIDTH]
        ch = stats[label_idx, cv2.CC_STAT_HEIGHT]
        area = stats[label_idx, cv2.CC_STAT_AREA]

        if area < 18:
            continue
        if cw < 2 or ch < max(8, int(0.35 * h)):
            continue
        comps.append((cx, cy, cw, ch, area))

    if len(comps) != 2:
        return [(x_s, x_e)]

    comps.sort(key=lambda t: t[0])
    (x1, _, w1, _, _), (x2, _, w2, _, _) = comps

    # Hai phần phải tách rời tương đối và không quá chênh lệch cực đoan.
    gap = x2 - (x1 + w1)
    if gap < -1:
        return [(x_s, x_e)]

    min_w = min(w1, w2)
    max_w = max(w1, w2)
    if min_w <= 0 or max_w / float(min_w) > 4.0:
        return [(x_s, x_e)]

    split_col = int((x1 + w1 + x2) / 2.0)
    split_col = max(2, min(w - 2, split_col))
    return [(x_s, x_s + split_col), (x_s + split_col, x_e)]

def refine_wide_boxes_iterative(boxes, img_clean, max_rounds=3):
    """
    Tách tiếp các box quá rộng bằng vertical-valley + fallback morphology.
    Mục tiêu: xử lý các cặp số còn dính sau pass chính.
    """
    if not boxes:
        return boxes

    refined = sorted(boxes, key=lambda b: b[0])
    for _ in range(max_rounds):
        widths = [w for (_, _, w, _) in refined]
        if not widths:
            break
        median_w = max(1.0, float(np.median(widths)))
        changed = False
        next_boxes = []

        for (x, y, w, h) in refined:
            # Chỉ thử tách những box rộng bất thường.
            if w <= median_w * 1.42:
                next_boxes.append((x, y, w, h))
                continue

            parts = split_one_segment(x, x + w, img_clean)
            if len(parts) == 1:
                parts = split_by_morph_fallback(x, x + w, img_clean)

            if len(parts) == 2:
                (p1s, p1e), (p2s, p2e) = parts
                w1, w2 = p1e - p1s, p2e - p2s
                min_valid_w = max(2, int(0.28 * median_w))
                if w1 >= min_valid_w and w2 >= min_valid_w:
                    next_boxes.append((p1s, y, w1, h))
                    next_boxes.append((p2s, y, w2, h))
                    changed = True
                    continue

            next_boxes.append((x, y, w, h))

        refined = sorted(next_boxes, key=lambda b: b[0])
        if not changed:
            break

    return refined

def segment_digits(img_thresh):
    """
    Cắt rời chữ số (Bước 3).
    Đầu vào: Ảnh nhị phân chữ trắng nền đen.
    Đầu ra: Danh sách các ảnh chữ số đã được cắt.
    """
    # 1. Phép Đóng (Morphological Closing - Chương 2)
    # Nối các nét đứt gãy bên trong từng chữ nhưng kernel nhỏ để tránh dính 2 chữ
    kernel_close = np.ones((2, 2), np.uint8)
    img_closed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel_close)
    img_clean = remove_barcode_bar_artifacts(img_closed)
    img_clean = keep_main_digit_band(img_clean)
    
    # 2. Vertical Projection (Chiếu dọc) - PASS 1: Lấy tất cả segments thô
    col_sums = np.sum(img_clean, axis=0)
    gap_threshold = np.max(col_sums) * 0.05 if np.max(col_sums) > 0 else 1
    is_gap = col_sums <= gap_threshold
    
    raw_segments = []
    in_char = False
    x_start = 0
    for col_idx in range(len(is_gap)):
        if not is_gap[col_idx] and not in_char:
            x_start = col_idx
            in_char = True
        elif is_gap[col_idx] and in_char:
            raw_segments.append((x_start, col_idx))
            in_char = False
    if in_char:
        raw_segments.append((x_start, len(is_gap)))
    
    if not raw_segments:
        return []
    
    # Lọc nhiễu: bỏ các đoạn quá mảnh (< 20% chiều rộng trung bình)
    all_ws = [e - s for s, e in raw_segments]
    median_w = int(np.median(all_ws))
    raw_segments = [(s, e) for s, e in raw_segments if (e - s) >= median_w * 0.2]
    
    # 3. PASS 2: Tách segment bất thường
    normal_avg_w = get_normal_avg_width(raw_segments)
    final_segments = []
    for x_s, x_e in raw_segments:
        seg_w = x_e - x_s
        seg_region = img_clean[:, x_s:x_e]

        # Ký tự có lỗ (0/6/8/9) dễ bị split sai, ưu tiên giữ nguyên.
        if seg_w > normal_avg_w * 1.55 and not has_inner_hole(seg_region):
            # Segment này nghi ngờ là 2 số dính → tách
            parts = split_one_segment(x_s, x_e, img_clean)
            if len(parts) == 1:
                # Fallback: thử tách bằng morphology nếu projection không tách được.
                parts = split_by_morph_fallback(x_s, x_e, img_clean)
            final_segments.extend(parts)
        else:
            final_segments.append((x_s, x_e))
    
    # 4. Từ segments, tính bounding box (thêm y top/bottom)
    final_boxes = []
    for x_s, x_e in final_segments:
        col_region = img_clean[:, x_s:x_e]
        row_sums = np.sum(col_region, axis=1)
        non_zero_rows = np.where(row_sums > 0)[0]
        if len(non_zero_rows) > 0:
            y_top = non_zero_rows[0]
            y_bot = non_zero_rows[-1]
            final_boxes.append((x_s, y_top, x_e - x_s, y_bot - y_top + 1))

    if not final_boxes:
        return []

    # Lọc box nhiễu quá mảnh (đặc biệt là bar dọc còn sót).
    widths = [w for (_, _, w, _) in final_boxes]
    median_w = max(1, int(np.median(widths)))
    filtered_boxes = []
    for x, y, w, h in final_boxes:
        region = img_clean[y:y+h, x:x+w]
        fill_ratio = (np.count_nonzero(region) / float(max(w * h, 1)))
        too_narrow = w < max(2, int(median_w * 0.25))
        looks_like_bar = (w <= 4 and h >= 18 and fill_ratio > 0.70)

        # Lọc thêm "gạch dọc giả ký tự":
        # nếu segment rất hẹp và profile mực theo hàng gần như phẳng (ít biến thiên),
        # thường đó là vạch barcode chứ không phải chữ số "1".
        row_profile = np.sum(region > 0, axis=1).astype(np.float32)
        profile_mean = float(np.mean(row_profile)) if row_profile.size > 0 else 0.0
        profile_std = float(np.std(row_profile)) if row_profile.size > 0 else 0.0
        profile_cv = (profile_std / profile_mean) if profile_mean > 1e-6 else 0.0
        flat_vertical_stroke = (w <= max(5, int(median_w * 0.45)) and profile_cv < 0.22 and fill_ratio > 0.45)

        if too_narrow or looks_like_bar or flat_vertical_stroke:
            continue
        filtered_boxes.append((x, y, w, h))

    if not filtered_boxes:
        return []

    # 5. Pass hoàn thiện: tách thêm các box rộng bất thường còn sót.
    filtered_boxes = refine_wide_boxes_iterative(filtered_boxes, img_clean, max_rounds=3)
    
    # Trích xuất từng ảnh nhỏ
    segmented_digits = []
    for (x, y, w, h) in filtered_boxes:
        pad = 2
        y1, y2 = max(0, y - pad), min(img_thresh.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(img_thresh.shape[1], x + w + pad)
        
        digit_img = img_thresh[y1:y2, x1:x2]
        segmented_digits.append(digit_img)
        
    return segmented_digits
