import cv2
from ultralytics import YOLO
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# โหลดโมเดลที่ดาวน์โหลดมา
model = YOLO('D:/Helmet_Detection/python_flask/runs/best.pt')
model.to(device)

def detect_helmet_in_image(image_path):
    # อ่านรูปภาพด้วย OpenCV
    image = cv2.imread(image_path)
    results = model(image, device='cuda' if torch.cuda.is_available() else 'cpu')
    output_image_path = f"static/output/detected_{os.path.basename(image_path)}"
    results[0].save(output_image_path)  
    return output_image_path


def detect_helmet_in_video(video_path, scale_factor=0.5, process_interval=1):  # ปรับค่า process_interval
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = f"static/output/detected_{os.path.basename(video_path)}"
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))  # ปรับ fps เป็น 30

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % process_interval == 0:
            results = model(frame, device='cuda' if torch.cuda.is_available() else 'cpu')
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame

        annotated_frame = cv2.resize(annotated_frame, (frame_width, frame_height))
        out.write(annotated_frame)

        frame_count += 1

    cap.release()
    out.release()

    return output_video_path


def gen_frames():
    cap = cv2.VideoCapture(0)  # เปิดกล้องเว็บแคม

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # ตรวจจับหมวกกันน็อคในเฟรม (ถ้ามี)
            result = model(frame)
            annotated_frame = result[0].plot()

            # แปลงเฟรมเป็น JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # ส่งคืนเฟรมแบบสตรีม
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
