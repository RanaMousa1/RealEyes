"""
Screenshot Text & Image Extractor
===================================
Extracts text and visual regions from screenshots.
NO API needed — runs 100% offline.

Requirements:
    pip install easyocr opencv-python pillow
"""

import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from PIL import Image, ImageTk, ImageDraw
import threading
import json
import os
from pathlib import Path
import cv2
import numpy as np
import easyocr


# ── OCR & Detection ───────────────────────────────────────────────────────────

def is_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return any('\u0600' <= c <= '\u06FF' for c in text)


def reconstruct_lines(blocks: list) -> str:
    """
    Group text blocks by their vertical position (same line),
    then sort each line RTL if Arabic, LTR if English.
    Returns clean multi-line string.
    """
    if not blocks:
        return ""

    # Sort all blocks top-to-bottom first
    sorted_blocks = sorted(blocks, key=lambda b: b["bbox"][0][1])

    # Group into lines — blocks whose vertical center is within LINE_THRESHOLD px
    LINE_THRESHOLD = 15
    lines = []
    current_line = [sorted_blocks[0]]

    for block in sorted_blocks[1:]:
        cy_current = sum(p[1] for p in block["bbox"]) / 4
        cy_prev    = sum(p[1] for p in current_line[-1]["bbox"]) / 4
        if abs(cy_current - cy_prev) <= LINE_THRESHOLD:
            current_line.append(block)
        else:
            lines.append(current_line)
            current_line = [block]
    lines.append(current_line)

    result_lines = []
    for line in lines:
        # Detect direction by majority of blocks in this line
        arabic_count = sum(1 for b in line if is_arabic(b["text"]))
        is_rtl = arabic_count > len(line) / 2

        if is_rtl:
            # Sort RTL: rightmost block first (highest x = first word)
            line_sorted = sorted(line, key=lambda b: b["bbox"][0][0], reverse=True)
        else:
            # Sort LTR: leftmost block first
            line_sorted = sorted(line, key=lambda b: b["bbox"][0][0])

        result_lines.append(" ".join(b["text"] for b in line_sorted))

    return "\n".join(result_lines)


def extract_text(image_path: str) -> list:
    """Extract text with bounding boxes using EasyOCR."""
    reader = easyocr.Reader(['en', 'ar'], gpu=False) 
    results = reader.readtext(image_path)
    extracted = []
    for (bbox, text, confidence) in results:
        if confidence > 0.3:
            extracted.append({
                "text": text,
                "confidence": round(confidence, 2),
                "bbox": [[int(p[0]), int(p[1])] for p in bbox]
            })
    return extracted


def detect_image_regions(image_path: str, text_results: list = None) -> list:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # ── Step 1: mask text areas ───────────────────────────────────────────────
    text_mask = np.zeros((h, w), dtype=np.uint8)
    if text_results:
        padding = max(10, min(w, h) // 50)
        for item in text_results:
            pts = np.array(item["bbox"], dtype=np.int32)
            x1 = max(0, pts[:, 0].min() - padding)
            y1 = max(0, pts[:, 1].min() - padding)
            x2 = min(w, pts[:, 0].max() + padding)
            y2 = min(h, pts[:, 1].max() + padding)
            text_mask[y1:y2, x1:x2] = 255

    # ── Step 2: detect solid-color UI bars (headers/toolbars) ─────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row_std = gray.std(axis=1)
    UI_STD_THRESHOLD = 18

    ui_mask = np.zeros((h, w), dtype=np.uint8)
    in_band = False
    band_start = 0
    for row_idx, std in enumerate(row_std):
        if std < UI_STD_THRESHOLD and not in_band:
            in_band = True
            band_start = row_idx
        elif std >= UI_STD_THRESHOLD and in_band:
            in_band = False
            band_h = row_idx - band_start
            if band_h >= max(4, int(h * 0.015)):
                ui_mask[band_start:row_idx, :] = 255
    if in_band:
        band_h = h - band_start
        if band_h >= max(4, int(h * 0.015)):
            ui_mask[band_start:h, :] = 255

    combined_mask = cv2.bitwise_or(text_mask, ui_mask)

    # ── Step 3: variance map → photo pixels ───────────────────────────────────
    blurred      = cv2.GaussianBlur(img, (21, 21), 0).astype(np.float32)
    diff         = (img.astype(np.float32) - blurred) ** 2
    variance_map = diff.mean(axis=2)

    photo_mask = (variance_map > 8).astype(np.uint8) * 255
    photo_mask = cv2.bitwise_and(photo_mask, cv2.bitwise_not(combined_mask))

    # ── Step 4: morphological close/open ─────────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    closed = cv2.morphologyEx(photo_mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed,     cv2.MORPH_OPEN,  kernel)

    # ── Step 5: find contours ─────────────────────────────────────────────────
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (w * h) * 0.02
    max_area = (w * h) * 0.95

    regions = []
    seen    = set()
    for cnt in contours:
        x, y, rw, rh = cv2.boundingRect(cnt)
        area = rw * rh
        if area < min_area or area > max_area:
            continue

        region_center_y = y + rh // 2
        if ui_mask[region_center_y, w // 2] == 255:
            continue

        key = (x // 30, y // 30, rw // 30, rh // 30)
        if key in seen:
            continue
        seen.add(key)

        # ── NEW: expand the bounding box upward into low-variance photo content ──
        # Walk upward row-by-row from the detected top edge. Keep expanding as long
        # as the row is NOT a UI bar and has reasonable image content (not pure solid
        # background of the app, which tends to be very dark or very uniform).
        expanded_y = y
        for probe_y in range(y - 1, -1, -1):
            if ui_mask[probe_y, x + rw // 2] == 255:
                break  # hit a UI bar — stop

            row_slice = img[probe_y, x:x + rw]
            # Check if this row looks like app background:
            # App backgrounds are typically very dark (mean < 30) OR extremely uniform (std < 6)
            row_gray  = gray[probe_y, x:x + rw]
            row_mean  = float(row_gray.mean())
            row_s     = float(row_gray.std())

            if row_mean < 30 and row_s < 6:
                break  # pure dark app background — stop expanding
            if row_s < 4:
                break  # completely flat row — likely not photo content

            expanded_y = probe_y  # this row looks like it belongs to the photo

        # Also try expanding downward (same logic, catches bottom clips)
        expanded_y2 = y + rh
        for probe_y in range(y + rh, h):
            if ui_mask[probe_y, x + rw // 2] == 255:
                break
            row_gray = gray[probe_y, x:x + rw]
            row_mean = float(row_gray.mean())
            row_s    = float(row_gray.std())
            if row_mean < 30 and row_s < 6:
                break
            if row_s < 4:
                break
            expanded_y2 = probe_y + 1

        new_y  = expanded_y
        new_rh = expanded_y2 - expanded_y

        aspect = rw / new_rh if new_rh > 0 else 1
        if rw > w * 0.25 and new_rh > h * 0.25:
            region_type = "photo / main image"
        elif aspect > 3:
            region_type = "banner / horizontal bar"
        elif aspect < 0.4:
            region_type = "vertical image"
        else:
            region_type = "image block"

        regions.append({
            "id":      len(regions) + 1,
            "type":    region_type,
            "x": x,        "y": new_y,
            "width":   rw, "height": new_rh,
            "area_px": rw * new_rh,
        })

    regions.sort(key=lambda r: r["area_px"], reverse=True)
    return regions[:10]


def annotate_image(image_path: str, text_results: list, regions: list) -> Image.Image:
    """Draw bounding boxes: green for text, orange for image regions."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    for region in regions:
        x, y, rw, rh = region["x"], region["y"], region["width"], region["height"]
        draw.rectangle([x, y, x + rw, y + rh], outline=(255, 140, 0, 220), width=2)

    for item in text_results:
        pts = item["bbox"]
        flat = [coord for pt in pts for coord in pt]
        draw.polygon(flat, outline=(0, 220, 100, 230))

    return img


def classify_text(text_blocks: list, regions: list) -> tuple:
    """
    Split text_blocks into two lists:
    - outside_text: text that appears outside all image regions (UI / captions)
    - inside_text:  text that appears inside an image region (embedded in photo)
    """
    outside, inside = [], []
    for item in text_blocks:
        pts = item["bbox"]
        cx  = sum(p[0] for p in pts) / 4   # center x of text box
        cy  = sum(p[1] for p in pts) / 4   # center y of text box
        in_region = False
        for r in regions:
            if (r["x"] <= cx <= r["x"] + r["width"] and
                    r["y"] <= cy <= r["y"] + r["height"]):
                in_region = True
                break
        (inside if in_region else outside).append(item)
    return outside, inside


def run_extraction(image_path: str) -> dict:
    """Run full pipeline and return results dict."""
    text_results = extract_text(image_path)
    regions      = detect_image_regions(image_path, text_results)
    outside_text, inside_text = classify_text(text_results, regions)

    cropped_path = None
    if regions:
        cropped_path = save_cropped_region(image_path, regions[0])

    return {
        "source":       image_path,
        "text":         reconstruct_lines(outside_text),
        "text_inside":  reconstruct_lines(inside_text),
        "text_blocks":  text_results,
        "outside_text": outside_text,
        "inside_text":  inside_text,
        "image_regions":  regions,
        "cropped_image":  cropped_path,
        "summary": {
            "total_text_blocks":        len(text_results),
            "total_outside_text":       len(outside_text),
            "total_inside_text":        len(inside_text),
            "total_image_regions":      len(regions),
        }
    }


def save_cropped_region(image_path: str, region: dict) -> str:
    """Save the first detected image region as a PNG and return its path."""
    img = cv2.imread(image_path)
    x, y, w, h = region["x"], region["y"], region["width"], region["height"]
    cropped = img[y:y+h, x:x+w]
    out_path = str(Path(image_path).with_suffix("")) + "_cropped_region.png"
    cv2.imwrite(out_path, cropped)
    return out_path


def save_results(result: dict, image_path: str) -> str:
    base = Path(image_path).with_suffix("")
    out_path = str(base) + "_extraction.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return out_path


# ── GUI ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Screenshot Extractor  •  EasyOCR + OpenCV  •  Offline")
        self.geometry("960x700")
        self.configure(bg="#0f0f0f")
        self._image_path = None
        self._build_ui()

    def _build_ui(self):
        mono = ("Courier New", 10)
        bold = ("Courier New", 11, "bold")

        hdr = tk.Frame(self, bg="#0f0f0f")
        hdr.pack(fill="x", padx=20, pady=(16, 4))
        tk.Label(hdr, text="[ SCREENSHOT EXTRACTOR ]", font=("Courier New", 15, "bold"),
                 fg="#00bfff", bg="#0f0f0f").pack(side="left")
        tk.Label(hdr, text="offline · no api", font=("Courier New", 9),
                 fg="#444", bg="#0f0f0f").pack(side="left", padx=10)

        ctrl = tk.Frame(self, bg="#0f0f0f")
        ctrl.pack(fill="x", padx=20, pady=6)

        self.path_var = tk.StringVar(value="No file selected")
        tk.Label(ctrl, textvariable=self.path_var, font=mono,
                 fg="#aaa", bg="#1a1a1a", anchor="w", padx=8).pack(
            side="left", fill="x", expand=True)

        tk.Button(ctrl, text="Browse…", font=bold,
                  bg="#1e1e1e", fg="#00bfff", activebackground="#2a2a2a",
                  activeforeground="#00bfff", relief="flat", padx=14,
                  command=self._browse).pack(side="left", padx=(8, 0))

        self.run_btn = tk.Button(ctrl, text="Extract ▶", font=bold,
                                 bg="#00bfff", fg="#0f0f0f",
                                 activebackground="#0099cc", relief="flat",
                                 padx=14, state="disabled",
                                 command=self._run)
        self.run_btn.pack(side="left", padx=(8, 0))

        panes = tk.PanedWindow(self, orient="horizontal", bg="#0f0f0f", sashwidth=4)
        panes.pack(fill="both", expand=True, padx=20, pady=8)

        left = tk.Frame(panes, bg="#1a1a1a")
        tk.Label(left, text="PREVIEW  (green=text  orange=images)",
                 font=("Courier New", 8, "bold"), fg="#444", bg="#1a1a1a").pack(
            anchor="w", padx=6, pady=(4, 0))
        self.preview_label = tk.Label(left, bg="#1a1a1a", text="(no image)",
                                       fg="#333", font=mono)
        self.preview_label.pack(fill="both", expand=True, padx=6, pady=6)
        panes.add(left, minsize=260)

        right = tk.Frame(panes, bg="#0f0f0f")
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background="#0f0f0f", borderwidth=0)
        style.configure("TNotebook.Tab", background="#1a1a1a", foreground="#666",
                         font=("Courier New", 9, "bold"), padding=[12, 4])
        style.map("TNotebook.Tab",
                  background=[("selected", "#2a2a2a")],
                  foreground=[("selected", "#00bfff")])

        def make_tab(label):
            tab = tk.Frame(nb, bg="#1a1a1a")
            nb.add(tab, text=label)
            box = scrolledtext.ScrolledText(tab, font=mono, bg="#1a1a1a",
                                             fg="#e0e0e0", relief="flat",
                                             wrap="word", state="disabled")
            box.pack(fill="both", expand=True, padx=4, pady=4)
            return box

        self.text_box    = make_tab("TEXT (outside)")
        self.text_in_box = make_tab("TEXT (inside image)")
        self.region_box  = make_tab("IMAGE REGIONS")

        # ── CHANGE 1: Cropped images tab ──────────────────────────────────────
        crops_tab = tk.Frame(nb, bg="#1a1a1a")
        nb.add(crops_tab, text="CROPPED")
        crops_canvas = tk.Canvas(crops_tab, bg="#1a1a1a", highlightthickness=0)
        crops_scroll = tk.Scrollbar(crops_tab, orient="vertical", command=crops_canvas.yview)
        crops_canvas.configure(yscrollcommand=crops_scroll.set)
        crops_scroll.pack(side="right", fill="y")
        crops_canvas.pack(side="left", fill="both", expand=True)
        self.crops_inner = tk.Frame(crops_canvas, bg="#1a1a1a")
        self._crops_win = crops_canvas.create_window((0, 0), window=self.crops_inner, anchor="nw")
        def _on_crops_configure(e):
            crops_canvas.configure(scrollregion=crops_canvas.bbox("all"))
            crops_canvas.itemconfig(self._crops_win, width=crops_canvas.winfo_width())
        self.crops_inner.bind("<Configure>", _on_crops_configure)
        crops_canvas.bind("<Configure>", lambda e: crops_canvas.itemconfig(self._crops_win, width=e.width))
        self._crops_canvas = crops_canvas
        self._crop_photos = []
        # ─────────────────────────────────────────────────────────────────────

        self.json_box   = make_tab("JSON")
        panes.add(right, minsize=420)

        self.status_var = tk.StringVar(value="Ready — select a screenshot.")
        tk.Label(self, textvariable=self.status_var, font=("Courier New", 9),
                 fg="#555", bg="#0f0f0f", anchor="w").pack(
            fill="x", padx=20, pady=(0, 10))

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select screenshot",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"),
                       ("All files", "*.*")])
        if not path:
            return
        self._image_path = path
        self.path_var.set(Path(path).name)
        self.run_btn.config(state="normal")
        self._load_preview(path)
        self._status("File loaded — click Extract ▶")

    def _load_preview(self, path, annotated=None):
        try:
            img = annotated if annotated else Image.open(path)
            img.thumbnail((300, 400))
            photo = ImageTk.PhotoImage(img)
            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo
        except Exception as e:
            self.preview_label.config(text=f"Preview error:\n{e}", image="")

    def _run(self):
        self.run_btn.config(state="disabled")
        self._status("Running EasyOCR + OpenCV… (first run downloads model ~100MB)")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            result = run_extraction(self._image_path)
            annotated = annotate_image(self._image_path,
                                        result["text_blocks"],
                                        result["image_regions"])
            out_path = save_results(result, self._image_path)
            self.after(0, self._show_results, result, annotated, out_path)
        except Exception as e:
            self.after(0, self._show_error, str(e))

    def _show_results(self, result, annotated, out_path):
        # outside text
        outside = result.get("text") or "(no text found)"
        self._set(self.text_box, outside)

        # inside text
        inside = result.get("text_inside") or "(no text detected inside images)"
        self._set(self.text_in_box, inside)

        lines = []
        for r in result["image_regions"]:
            lines.append(f"[{r['id']}] {r['type']}")
            lines.append(f"     position: ({r['x']}, {r['y']})  size: {r['width']}x{r['height']}px")
            lines.append("")
        self._set(self.region_box, "\n".join(lines) if lines else "(no regions detected)")

        self._set(self.json_box, json.dumps(result, indent=2, ensure_ascii=False))
        self._load_preview(self._image_path, annotated)
        self._show_crops(self._image_path, result["image_regions"])  # ── CHANGE 2

        self._status(
            f"Done  {result['summary']['total_text_blocks']} text blocks, "
            f"{result['summary']['total_image_regions']} image regions  ->  {out_path}"
        )
        if result.get('cropped_image'):
            self._status(self.status_var.get() + f"  |  Cropped image: {result['cropped_image']}")
        self.run_btn.config(state="normal")

    # ── CHANGE 3: new method ──────────────────────────────────────────────────
    def _show_crops(self, image_path: str, regions: list):
        """Populate the CROPPED tab with clickable thumbnails — click to save."""
        for w in self.crops_inner.winfo_children():
            w.destroy()
        self._crop_photos.clear()
        # store full-res crops so we can save them on click
        self._crop_images = {}

        if not regions:
            tk.Label(self.crops_inner, text="(no regions detected)",
                     font=("Courier New", 10), fg="#555", bg="#1a1a1a").pack(pady=20)
            return

        # hint label at the top
        tk.Label(self.crops_inner, text="💾  Click any image to save it",
                 font=("Courier New", 9), fg="#00bfff", bg="#1a1a1a").grid(
            row=0, column=0, columnspan=3, pady=(6, 2))

        src = Image.open(image_path).convert("RGB")
        src_w, src_h = src.size
        mono = ("Courier New", 9)
        THUMB = 200
        COLS  = 3

        for idx, r in enumerate(regions):
            x, y, rw, rh = r["x"], r["y"], r["width"], r["height"]
            x1 = max(0, x);        y1 = max(0, y)
            x2 = min(src_w, x+rw); y2 = min(src_h, y+rh)
            if x2 <= x1 or y2 <= y1:
                continue

            # full-res crop (for saving)
            full_crop = src.crop((x1, y1, x2, y2))
            crop_id   = r["id"]
            self._crop_images[crop_id] = full_crop

            # thumbnail (for display)
            thumb = full_crop.copy()
            thumb.thumbnail((THUMB, THUMB))

            # grid position — offset by 1 row because of the hint label
            grid_row = (idx // COLS) * 2 + 1
            grid_col =  idx % COLS

            # hoverable cell frame
            cell = tk.Frame(self.crops_inner, bg="#222", bd=2, relief="solid",
                            cursor="hand2")
            cell.grid(row=grid_row, column=grid_col, padx=6, pady=(6, 2), sticky="n")

            photo = ImageTk.PhotoImage(thumb)
            self._crop_photos.append(photo)

            img_lbl = tk.Label(cell, image=photo, bg="#222", cursor="hand2")
            img_lbl.pack(padx=4, pady=(4, 2))

            # hover highlight
            def _enter(e, f=cell):  f.config(bg="#00bfff")
            def _leave(e, f=cell):  f.config(bg="#222")
            for w in (cell, img_lbl):
                w.bind("<Enter>", _enter)
                w.bind("<Leave>", _leave)
                w.bind("<Button-1>", lambda e, cid=crop_id: self._save_crop(cid))

            info = f"[{r['id']}] {r['type']}\n{rw}×{rh}px  •  click to save"
            tk.Label(self.crops_inner, text=info, font=mono,
                     fg="#aaa", bg="#1a1a1a", justify="center").grid(
                row=grid_row + 1, column=grid_col, pady=(0, 8))

    def _save_crop(self, crop_id: int):
        """Open save dialog and write the full-res crop to disk."""
        from tkinter import filedialog
        crop_img = self._crop_images.get(crop_id)
        if crop_img is None:
            return
        save_path = filedialog.asksaveasfilename(
            title=f"Save cropped image [{crop_id}]",
            defaultextension=".png",
            initialfile=f"crop_{crop_id}.png",
            filetypes=[("PNG image", "*.png"),
                       ("JPEG image", "*.jpg"),
                       ("All files", "*.*")])
        if not save_path:
            return
        crop_img.save(save_path)
        self._status(f"Saved crop [{crop_id}]  →  {save_path}")
    # ─────────────────────────────────────────────────────────────────────────

    def _show_error(self, msg):
        self._status(f"Error: {msg}")
        self.run_btn.config(state="normal")
        print("ERROR:", msg)

    def _set(self, widget, content):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.config(state="disabled")

    def _status(self, msg):
        self.status_var.set(msg)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()