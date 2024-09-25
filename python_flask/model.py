import cv2
from ultralytics import YOLO
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# โหลดโมเดลที่ดาวน์โหลดมา
model = YOLO('D:/Helmet_Detection/python_flask/runs/F190.pt')
model.to(device)



# ฟังก์ชันตรวจจับหมวกกันน็อกในรูปภาพ
def detect_helmet_in_image(image_path):
    # อ่านรูปภาพด้วย OpenCV
    image = cv2.imread(image_path)

    # ตรวจจับหมวกกันน็อก
    results = model(image, device='cuda' if torch.cuda.is_available() else 'cpu')

    # วาดกรอบตรวจจับใหม่บนเฟรมต้นฉบับ
    for result in results[0].boxes:
        bbox = result.xyxy[0].cpu().numpy().astype(int)
        cls = int(result.cls)

        if cls == 0:
            color = (0, 255, 0)  # สีเขียวสำหรับ with-helmet
            label = 'with-helmet'
        else:
            color = (0, 0, 255)  # สีแดงสำหรับ without-helmet
            label = 'without-helmet'

        # วาดกรอบ
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # คำนวณขนาดของข้อความ
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)

        # เพิ่มพื้นหลังสี่เหลี่ยมให้กับข้อความ
        cv2.rectangle(image, (bbox[0], bbox[1] - text_height - 10), (bbox[0] + text_width, bbox[1]), color, -1)

        # วาดข้อความบนพื้นหลังสี่เหลี่ยม
        cv2.putText(image, label, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # บันทึกผลลัพธ์
    output_image_path = f"static/output/detected_{os.path.basename(image_path)}"
    cv2.imwrite(output_image_path, image)  # บันทึกเฟรมที่มีการวาดกรอบใหม่

    return output_image_path


def detect_helmet_in_video(video_path, process_interval=2):  # ปรับ process_interval
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # กำหนดอุปกรณ์เพียงครั้งเดียว

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = f"static/output/detected_{os.path.basename(video_path)}"
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))  # อัตราเฟรม 30 FPS

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ใช้ process_interval เพื่อเร่งความเร็ว
        if frame_count % process_interval == 0:
            results = model(frame, device=device)
            # ไม่วาดกรอบอะไรทั้งสิ้น
            # คุณอาจจะต้องการทำอะไรบางอย่างกับผลลัพธ์ แต่ไม่วาดกรอบ

        out.write(frame)  # เขียนเฟรมที่ไม่ได้มีการวาดกรอบ
        frame_count += 1

    # ปิดออบเจ็กต์เมื่อเสร็จสิ้น
    cap.release()
    out.release()

    return output_video_path

def gen_frames():
    cap = cv2.VideoCapture(0)  # เปิดกล้องเว็บแคม
    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("ไม่สามารถอ่านเฟรมจากกล้องได้")
            break

        # ตรวจจับหมวกกันน็อกในเฟรม
        results = model(frame)

        # วาดกรอบและ label ใหม่ในเฟรม
        for result in results[0].boxes:
            bbox = result.xyxy[0].cpu().numpy().astype(int)
            cls = int(result.cls)

        # แปลงเฟรมเป็น JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("ไม่สามารถแปลงเฟรมเป็น JPEG ได้")
            break

        frame = buffer.tobytes()

        # ส่งคืนเฟรมแบบสตรีม
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()