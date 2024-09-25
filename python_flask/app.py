from flask import Flask, Response, request, render_template, redirect, url_for
from model import detect_helmet_in_image, detect_helmet_in_video ,model
import os
import cv2
import torch

app = Flask(__name__)

# หน้าเว็บหลัก
@app.route('/')
def index():
    return render_template('index.html')

# อัปโหลดและตรวจจับจากรูปภาพ
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file', 400
    
    # บันทึกรูปภาพ
    image_path = os.path.join('static/output', file.filename)
    file.save(image_path)

    # ตรวจจับหมวกกันน็อคในรูปภาพ
    output_image_path = detect_helmet_in_image(image_path)

    return redirect(url_for('show_result', filename=os.path.basename(output_image_path)))

# อัปโหลดและตรวจจับจากวิดีโอ
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return 'ไม่มีไฟล์ที่อัปโหลด', 400

    file = request.files['file']
    
    if file.filename == '':
        return 'ไม่มีการเลือกไฟล์', 400
    
    # บันทึกไฟล์วิดีโอ
    video_path = os.path.join('static/output', file.filename)
    file.save(video_path)

    # สร้างเส้นทางสำหรับการสตรีมวิดีโอที่ตรวจจับแล้ว
    detected_video_path = detect_helmet_in_video(video_path)

    # เปลี่ยนเส้นทางไปยังหน้าสตรีมวิดีโอที่ตรวจจับแล้ว
    return redirect(url_for('stream_detected_video', filename=os.path.basename(detected_video_path)))


@app.route('/stream_detected_video/<filename>')
def stream_detected_video(filename):
    video_path = os.path.join('static/output', filename)
    
    def gen_frames():
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, device='cuda' if torch.cuda.is_available() else 'cpu')

            # วาดกรอบสำหรับแต่ละการตรวจจับ
            for result in results[0].boxes:
                bbox = result.xyxy[0].cpu().numpy().astype(int)  # กรอบ bounding box
                cls = int(result.cls)  # คลาสของวัตถุ

                # เรียกใช้ฟังก์ชันวาดกรอบ
                draw_bounding_box(frame, bbox, cls)

            ret, buffer = cv2.imencode('.jpg', frame)  # ใช้ frame ที่มีกรอบแล้ว
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# แสดงรูปภาพหรือวิดีโอที่ตรวจจับเสร็จแล้ว
@app.route('/result/<filename>')
def show_result(filename):
    if filename.endswith('.mp4'):
        return render_template('result_video.html', video_file=filename)
    else:
        return render_template('result_image.html', image_file=filename)

webcam_active = True

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

def gen_frames():
    cap = cv2.VideoCapture(0)  # เปิดกล้องเว็บแคม
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # ตรวจจับหมวกกันน็อคในเฟรม
            results = model(frame)
            annotated_frame = results[0].plot()

            # แปลงเฟรมเป็น JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # ส่งคืนเฟรมแบบสตรีม
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

webcam_active = True

def draw_bounding_box(frame, bbox, cls):
    if cls == 0:  # ตรวจจับคลาส with-helmet
        color = (0, 255, 0)  # สีเขียวสำหรับ with-helmet
        label = 'with-helmet'
    elif cls == 1:  # ตรวจจับคลาส without-helmet
        color = (0, 0, 255)  # สีแดงสำหรับ without-helmet
        label = 'without-helmet'
    else:
        # กรณีเจอคลาสอื่นนอกเหนือจาก 0 หรือ 1 ให้ข้ามการวาดกรอบ
        print(f"พบคลาสที่ไม่ใช่ 0 หรือ 1: {cls}, ข้ามการวาดกรอบ")
        return

    # วาดกรอบ bounding box รอบวัตถุที่ตรวจพบ
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    # คำนวณขนาดของข้อความที่จะใส่
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)

    # เพิ่มพื้นหลังสี่เหลี่ยมให้กับข้อความ
    cv2.rectangle(frame, (bbox[0], bbox[1] - text_height - 10), (bbox[0] + text_width, bbox[1]), color, -1)

    # วาดข้อความ label ใหม่บนเฟรมวิดีโอ
    cv2.putText(frame, label, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)  # ข้อความสีขาว

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
            draw_bounding_box(frame, bbox, cls)

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

@app.route('/stop_webcam')
def stop_webcam():
    global webcam_active
    webcam_active = False  # เปลี่ยนสถานะเพื่อหยุดการสตรีมเว็บแคม
    return redirect(url_for('index'))

            
if __name__ == '__main__':
    app.run(debug=True)