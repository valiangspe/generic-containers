from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
from ultralytics import YOLO
import io

app = Flask(__name__)

# Load the YOLO model
model = YOLO("yolov8n.pt")

def base64_to_image(base64_string):
    """Converts base64 string to OpenCV image."""
    # Remove header if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64 string
    img_data = base64.b64decode(base64_string)
    
    # Convert to numpy array
    nparr = np.frombuffer(img_data, np.uint8)
    
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    try:
        # Get base64 image from request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert base64 to image
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Perform object detection
        results = model(image)
        
        # Extract detection results
        boxes = []
        confidences = []
        class_ids = []
        
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(float, result.xyxy[0])  # Convert to float for JSON serialization
            boxes.append([x1, y1, x2, y2])
            confidences.append(float(result.conf[0]))
            class_ids.append(int(result.cls[0]))
        
        response = {
            'boxes': boxes,
            'confidences': confidences,
            'class_ids': class_ids
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2006)