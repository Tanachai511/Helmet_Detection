from ultralytics import YOLO

# เป็นการสร้างโมเดลใหม่ขึ้นมา
model = YOLO('yolov8n.yaml')

# โหลด pretrained model มาเพื่อให้เราไม่ต้องเทรนใหม่ทั้งหมดตั้งแต่เริ่ม
model = YOLO('yolov8n.pt')

path = 'D:/dataset/data.yaml'
results = model.train(data=path, epochs=20)

# เซฟโมเดลโดยให้โมเดลอยู่ใน ONNX format
success = model.export(format='onnx')