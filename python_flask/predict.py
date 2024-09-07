from ultralytics import YOLO
import cv2

class HELMET_DETECTION :
    def __init__(self,model_path):
        # นำโมเดลเข้ามา
        self.model = YOLO(model_path)
        
    def __call__(self,input_image, output_path):
        # อ่านรูปที่ต้องการที่จะ detect
        img = cv2.imread(input_image)
        # detect เเละเก็บผลลัพธ์ไว้ที่ตัวแปร results
        results = self.model(input_image)[0]
        # ใช้ for เพื่อรองรับการเจอ object มากกว่า 1 
        for i in range(len(results.boxes.data)):
            # นำค่าพิกัดbounding box,ค่าความมั่นใจ(confidet ratio),class 
            # มาเก็บไว้ที่ตัวแปร boxes
            boxes = results.boxes.data[i].numpy().tolist()
            #สร้าง bounding box
            cv2.rectangle(img,(int(boxes[0]),int(boxes[1])),
                         (int(boxes[2]),int(boxes[3])),[0,255,0],2)
            #เพิ่ม Text ที่บอก class เเละ confident ratio
            cv2.putText(img,
                        f'{results.names[int(boxes[5])]}:{int(boxes[4]*100)}%', 
                        (int(boxes[0]), int(boxes[1] - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, 
                        [225, 0, 0],
                        thickness=2)
        # บันทึกภาพลงที่ output_path
        cv2.imwrite(output_path, img)