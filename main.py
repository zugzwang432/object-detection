import torch
import cv2
import numpy as np
from PIL import Image

def load_model():
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detect_objects(model, image_path):
    # Read image
    img = Image.open(image_path)
    
    # Perform inference
    results = model(img)
    
    # Convert image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Process results
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if conf > 0.5:  # Confidence threshold
            label = f"{results.names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img_cv, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return img_cv

def main():
    model = load_model()
    image_path = "/home/user/object-detection/image.jpg"  # Replace with your image path
    result_image = detect_objects(model, image_path)
    
    # Display result
    cv2.imshow("Object Detection Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally, save the result
    # cv2.imwrite("result.jpg", result_image)

if __name__ == "__main__":
    main()
