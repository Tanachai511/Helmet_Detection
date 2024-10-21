from flask import Flask, Response, request, render_template, redirect, url_for
from model import detect_helmet_in_image, detect_helmet_in_video ,model
from pymongo import MongoClient
import os
import cv2
import torch

app = Flask(__name__)

client = MongoClient('mongodb://localhost:27017/')
db = client['helmet_detection']  # Database name
issues_collection = db['issue_reports']  # Collection name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_report', methods=['POST'])
def submit_report():
    image_file = request.form.get('image_file')
    is_incorrect = request.form.get('is_incorrect') == 'on'
    comment = request.form.get('comment')

    issue_report = {
        'image_file': image_file,
        'is_incorrect': is_incorrect,
        'comment': comment
    }

    # เพิ่มข้อมูลลงในคอลเลกชัน 'issues'
    issues_collection.insert_one(issue_report)

    return redirect(url_for('show_result', filename=image_file))

#อัปโหลดและตรวจจับจากรูปภาพ
from flask import render_template

# ในฟังก์ชันตรวจจับหมวกกันน็อก
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    image_path = os.path.join('static/output', file.filename)
    file.save(image_path)

    output_image_path, helmet_count, without_helmet_count = detect_helmet_in_image(image_path)

    return render_template('result_image.html', 
                           image_file=os.path.basename(output_image_path), 
                           helmet_count=helmet_count, 
                           without_helmet_count=without_helmet_count)

# อัปโหลดและตรวจจับจากวิดีโอ
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return 'ไม่มีไฟล์ที่อัปโหลด', 400

    file = request.files['file']
    
    if file.filename == '':
        return 'ไม่มีการเลือกไฟล์', 400
    
    video_path = os.path.join('static/output', file.filename)
    file.save(video_path)
    detected_video_path = detect_helmet_in_video(video_path)

    return redirect(url_for('stream_detected_video', filename=os.path.basename(detected_video_path)))


@app.route('/stream_detected_video/<filename>')
def stream_detected_video(filename, process_interval=2):
    video_path = os.path.join('static/output', filename)
    
    def gen_frames():
        cap = cv2.VideoCapture(video_path)
        frame_count = 0 
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % process_interval == 0:
                results = model(frame, device='cuda' if torch.cuda.is_available() else 'cpu')

                for result in results[0].boxes:
                    bbox = result.xyxy[0].cpu().numpy().astype(int)
                    cls = int(result.cls) 
                    draw_bounding_box(frame, bbox, cls)

            ret, buffer = cv2.imencode('.jpg', frame) 
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            frame_count += 1

        cap.release()

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/result/<filename>')
def show_result(filename):
    helmet_count = request.args.get('helmet_count', type=int)
    without_helmet_count = request.args.get('without_helmet_count', type=int)

    if filename.endswith('.mp4'):
        return render_template('result_video.html', video_file=filename)
    else:
        return render_template('result_image.html', image_file=filename, 
                               helmet_count=helmet_count, without_helmet_count=without_helmet_count)


webcam_active = True

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model(frame)
            annotated_frame = results[0].plot()

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

webcam_active = True

def draw_bounding_box(frame, bbox, cls):
    if cls == 0:
        color = (0, 255, 0)  
        label = 'with-helmet'
    elif cls == 1:
        color = (0, 0, 255) 
        label = 'without-helmet'
    else:
        print(f"พบคลาสที่ไม่ใช่ 0 หรือ 1: {cls}, ข้ามการวาดกรอบ")
        return

    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)

    cv2.rectangle(frame, (bbox[0], bbox[1] - text_height - 10), (bbox[0] + text_width, bbox[1]), color, -1)

    cv2.putText(frame, label, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)  # ข้อความสีขาว

def gen_frames():
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("ไม่สามารถอ่านเฟรมจากกล้องได้")
            break

        results = model(frame)

        for result in results[0].boxes:
            bbox = result.xyxy[0].cpu().numpy().astype(int)
            cls = int(result.cls)
            draw_bounding_box(frame, bbox, cls)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("ไม่สามารถแปลงเฟรมเป็น JPEG ได้")
            break

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/stop_webcam')
def stop_webcam():
    global webcam_active
    webcam_active = False 
    return redirect(url_for('index'))

            
if __name__ == '__main__':
    app.run(debug=True)