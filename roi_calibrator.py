import cv2
import json
import os

CONFIG_PATH = "config.json"
PAGE_IMAGE  = "page.png"
LABELS = ["pick2", "pick3", "pick4", "pick5"]

def load_cfg():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {"roi": {"pick2": None, "pick3": None, "pick4": None, "pick5": None}}

def save_cfg(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def main():
    if not os.path.exists(PAGE_IMAGE):
        print("page.png not found. Run:  py screenshot_ocr.py")
        return

    img = cv2.imread(PAGE_IMAGE)
    if img is None:
        print("Could not open page.png")
        return

    cfg = load_cfg()
    rois = {}
    preview = img.copy()

    for label in LABELS:
        print(f"Draw ROI for {label}: click-drag, then press ENTER. Press ESC to skip.")
        r = cv2.selectROI("ROI Calibrator (page.png)", img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("ROI Calibrator (page.png)")
        if r == (0, 0, 0, 0):
            rois[label] = None
            continue
        x, y, w, h = map(int, r)
        rois[label] = [x, y, w, h]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(preview, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imwrite("rois_preview.png", preview)
    cfg["roi"] = rois
    save_cfg(cfg)
    print("Saved ROI boxes to", CONFIG_PATH)
    print("Preview saved to rois_preview.png")

if __name__ == "__main__":
    main()
