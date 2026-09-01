from ultralytics import YOLO

model = YOLO("./manga_panel_detector_fp32.pt")
results = model.predict(r"D:\Project\manga-translation\open-mantra-dataset\images\balloon_dream\ja\010.jpg", conf=0.25)

for box in results[0].boxes:
    cls = int(box.cls)  # 0=panel, 1=text
    conf = float(box.conf)
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    label = "panel" if cls == 0 else "text"
    print(f"{label} ({conf:.2f}): [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")