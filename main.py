import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time

# تحميل النموذج - سيقوم بتحميله تلقائياً عند أول تشغيل
model = YOLO('yolov8n.pt') 

def draw_futuristic_overlay(frame, detections):
    h, w, _ = frame.shape
    center = (w // 2, h // 2)
    color = (0, 255, 0)  # اللون الأخضر الراداري
    
    # 1. رسم دوائر الرادار المركزية
    for r in [100, 200, 300]:
        cv2.circle(frame, center, r, color, 1)
        cv2.putText(frame, f"{r}m", (center[0] + 5, center[1] - r - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 2. رسم خطوط المسح (Scanning Lines)
    cv2.line(frame, (center[0]-350, center[1]), (center[0]+350, center[1]), color, 1)
    cv2.line(frame, (center[0], center[1]-350), (center[0], center[1]+350), color, 1)

    # 3. معالجة الأهداف المكتشفة
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        conf = det.conf[0]
        cls = int(det.cls[0])
        label = model.names[cls]

        # تصفية النتائج للتركيز على الطائرات أو الأجسام الطائرة (حسب نموذج YOLO)
        if label in ['bird', 'airplane', 'drone']: 
            # رسم مربع استهداف متطور (Corner Brackets)
            length = 20
            cv2.line(frame, (x1, y1), (x1 + length, y1), color, 2)
            cv2.line(frame, (x1, y1), (x1, y1 + length), color, 2)
            cv2.line(frame, (x2, y1), (x2 - length, y1), color, 2)
            cv2.line(frame, (x2, y1), (x2, y1 + length), color, 2)
            
            # حساب بيانات وهمية للسرعة والارتفاع (للمحاكاة)
            speed = np.random.randint(40, 80)
            alt = np.random.randint(100, 300)
            
            # عرض البيانات بجانب الهدف
            info_text = f"TARGET: {label.upper()} | SPD: {speed}km/h | ALT: {alt}m"
            cv2.putText(frame, info_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 4. لوحة المعلومات السفلية
    cv2.rectangle(frame, (0, h - 60), (w, h), (0, 40, 0), -1) # شريط أسفل داكن
    cv2.putText(frame, "RADAR SYSTEM ACTIVE - SCANNING SECTOR: NORTH", (20, h - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame

# تشغيل الكاميرا
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # إجراء التوقع بالذكاء الاصطناعي
    results = model(frame, conf=0.3, verbose=False)
    
    # رسم الواجهة الرادارية
    frame = draw_futuristic_overlay(frame, results[0].boxes)

    # عرض النتيجة
    cv2.imshow('AI Drone Radar System', frame)

    # الخروج عند الضغط على مفتاح 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
