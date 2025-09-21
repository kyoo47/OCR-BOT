import json, os, re, glob
import cv2, pytesseract
from PIL import Image
import numpy as np
from playwright.sync_api import sync_playwright

# explicit path (safe even if PATH already set)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CONFIG_PATH = "config.json"
PAGE_URL = "http://instantcash.gaminglts.com:82/"
PICKS = ["pick2", "pick3", "pick4", "pick5"]
EXPECTED = {"pick2": 2, "pick3": 3, "pick4": 4, "pick5": 5}

def load_cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def screenshot_page(out_path="page.png"):
    cfg = load_cfg()
    vp = cfg.get("viewport", {})
    width  = int(vp.get("width", 1366))
    height = int(vp.get("height", 768))
    dpf    = float(vp.get("deviceScaleFactor", 1))
    with sync_playwright() as p:
        browser  = p.chromium.launch()
        context  = browser.new_context(viewport={"width": width, "height": height},
                                       device_scale_factor=dpf)
        page     = context.new_page()
        page.goto(PAGE_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(800)  # brief settle
        page.screenshot(path=out_path, full_page=True)
        browser.close()

def crop_with_pad(pil_img, roi, pad=8):
    x, y, w, h = roi
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = x + w + pad;     y1 = y + h + pad
    return pil_img.crop((x0, y0, x1, y1))

def upscale(pil_img, scale=3):
    arr = np.array(pil_img)
    up = cv2.resize(arr, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return up  # numpy RGB

def yellow_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 90, 120], np.uint8)
    upper = np.array([40, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask

def white_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 200], np.uint8)
    upper = np.array([180, 80, 255], np.uint8)
    return cv2.inRange(hsv, lower, upper)

def order_row_major(boxes):
    if not boxes:
        return []
    boxes_y = sorted(boxes, key=lambda b: b[1])
    avg_h = sum(b[3] for b in boxes_y) / len(boxes_y)
    row_thresh = max(6, avg_h * 0.6)
    rows, current = [], [boxes_y[0]]
    for b in boxes_y[1:]:
        if abs(b[1] - current[-1][1]) <= row_thresh:
            current.append(b)
        else:
            rows.append(sorted(current, key=lambda r: r[0]))
            current = [b]
    rows.append(sorted(current, key=lambda r: r[0]))
    return [b for row in rows for b in row]

def pick_digit_boxes(mask, want_n):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if w < 14 or h < 14 or area < 120:
            continue
        per = cv2.arcLength(c, True) + 1e-6
        circ = 4*np.pi*area/(per*per)
        if circ < 0.55:
            continue
        boxes.append((x,y,w,h))
    if not boxes:
        return []
    ordered = order_row_major(boxes)
    return ordered[:want_n]

def ocr_single_digit(img_bin):
    pil = Image.fromarray(img_bin)
    txt = pytesseract.image_to_string(
        pil, config=r'-c tessedit_char_whitelist=0123456789 --psm 10'
    )
    m = re.search(r"\d", txt or "")
    return m.group(0) if m else ""

def cleanup_images():
    patterns = ["page.png", "live_*_balls.png", "live_*_ymask.png", "live_*_digit*.png"]
    for pat in patterns:
        for path in glob.glob(pat):
            try:
                os.remove(path)
            except:
                pass

def detect_from_roi(base_pil, roi, want_n, label):
    """Detect digits, auto-expand ROI downward if fewer than expected are found."""
    base_np = np.array(base_pil)  # RGB
    H, W = base_np.shape[:2]

    def run_once(roi_local):
        row = crop_with_pad(base_pil, roi_local, pad=8)
        up_rgb = upscale(row, scale=3)
        up_bgr = cv2.cvtColor(up_rgb, cv2.COLOR_RGB2BGR)
        ymask = yellow_mask(up_bgr)
        boxes = pick_digit_boxes(ymask, want_n)
        digits = []
        for (x,y,w,h) in boxes:
            xi = x + int(w*0.18); yi = y + int(h*0.18)
            wi = max(1, int(w*0.64)); hi = max(1, int(h*0.64))
            ball = up_bgr[yi:yi+hi, xi:xi+wi]
            wmask = white_mask(ball)
            gray  = cv2.cvtColor(ball, cv2.COLOR_BGR2GRAY)
            glyph = cv2.bitwise_and(gray, gray, mask=wmask)
            _, bin_img = cv2.threshold(glyph, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            bin_img = cv2.bitwise_not(bin_img)
            d = ocr_single_digit(bin_img)
            digits.append(d if d else "?")
        return "".join(digits), len(boxes)

    # first try: original ROI
    out, n = run_once(roi)
    if n >= want_n:
        return out

    # fallback: expand ROI height downward to catch wrapped 2nd row
    x, y, w, h = roi
    new_h = int(h * 1.9)
    max_h = H - y - 1
    if new_h > max_h:
        new_h = max_h
    expanded = [x, y, w, new_h]
    out2, n2 = run_once(expanded)
    return out2

def main():
    # screenshot
    screenshot_page("page.png")
    base = Image.open("page.png").convert("RGB")
    cfg = load_cfg(); rois = cfg.get("roi", {})

    results = {}
    for label in PICKS:
        roi = rois.get(label)
        if not roi:
            results[label] = ""
            continue
        want_n = EXPECTED[label]
        results[label] = detect_from_roi(base, roi, want_n, label)

    for k in PICKS:
        print(f"{k}: {results.get(k,'')}")

    cleanup_images()

if __name__ == "__main__":
    main()
