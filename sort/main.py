from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import supervision as sv
from typing import List

app = Flask(__name__)
# Initialize ByteTrack tracker
tracker = sv.ByteTrack()


def decode_base64_image(base64_str):
    """Decodes base64 image data into an OpenCV image format."""
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


"""
{
    "image": "<base64_encoded_image_string>",
    "boxes": [[x1, y1, x2, y2], ...],
    "confidences": [0.9, 0.8, ...],
    "class_ids": [1, 2, ...]
}
"""


@app.route("/track_objects", methods=["POST"])
def track_objects():
    data = request.json
    base64_image = data.get("image")
    boxes = data.get("boxes")  # Expected format: [[x1, y1, x2, y2], ...]
    confidences = data.get("confidences")  # Expected format: [0.9, 0.8, ...]
    class_ids = data.get("class_ids")  # Expected format: [1, 2, ...]

    # Ensure all required fields are present
    if not base64_image or not boxes or not confidences or not class_ids:
        return jsonify({"error": "Invalid input, required fields missing"}), 400

    # Decode the image (for visualization or further processing if needed)
    image = decode_base64_image(base64_image)

    # Convert inputs to numpy arrays
    boxes_np = np.array(boxes)
    confidences_np = np.array(confidences)
    class_ids_np = np.array(class_ids)

    # Create detections object
    detections = sv.Detections(
        xyxy=boxes_np, confidence=confidences_np, class_id=class_ids_np
    )

    # Update tracker with current detections using update_with_detections method
    tracked_detections = tracker.update_with_detections(detections=detections)

    # Prepare the JSON response
    tracked_data = [
        {
            "id": int(track_id),  # Track ID assigned by ByteTrack
            "box": [
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            ],  # Convert to float for JSON serialization
            "class_id": int(class_id),
            "confidence": float(conf),
        }
        for track_id, (x1, y1, x2, y2), conf, class_id in zip(
            tracked_detections.tracker_id,
            tracked_detections.xyxy,
            tracked_detections.confidence,
            tracked_detections.class_id,
        )
    ]

    return jsonify({"tracked_objects": tracked_data})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=2009)
