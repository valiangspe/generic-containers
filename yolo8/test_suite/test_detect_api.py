import pprint
import cv2
import base64
import json
import requests
import numpy as np


def read_image_as_base64(filepath):
    """Reads an image file and encodes it in base64 format."""
    with open(filepath, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    return base64_str


def draw_tracked_objects(image, tracked_objects):
    """Draws bounding boxes and IDs for tracked objects on the image."""
    # Make a copy of the image to avoid modifying the original
    output_image = image.copy()

    for obj in tracked_objects:
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, obj["box"])
        track_id = obj["id"]
        class_id = obj["class_id"]
        confidence = obj["confidence"]

        # Draw bounding box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Prepare label text with ID, class, and confidence
        label = f"ID:{track_id} Class:{class_id} Conf:{confidence:.2f}"

        # Calculate label position
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_label = max(y1 - 10, label_size[1])

        # Draw white background for text
        cv2.rectangle(
            output_image,
            (x1, y_label - label_size[1]),
            (x1 + label_size[0], y_label + baseline),
            (255, 255, 255),
            cv2.FILLED,
        )

        # Draw text
        cv2.putText(
            output_image,
            label,
            (x1, y_label),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    return output_image


def main():
    # Step 1: Read the image and get base64 encoding
    image_path = "traffic.png"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    base64_image = read_image_as_base64(image_path)

    # Step 2: Send image to object detection API
    detection_response = requests.post(
        "http://127.0.0.1:2006/detect_objects", json={"image": base64_image}
    )

    if detection_response.status_code != 200:
        raise Exception(f"Detection API error: {detection_response.text}")

    detection_results = detection_response.json()

    # Step 3: Prepare the data for the tracking API
    data = {
        "image": base64_image,
        "boxes": detection_results["boxes"],
        "confidences": detection_results["confidences"],
        "class_ids": detection_results["class_ids"],
    }

    # Step 4: Send data to the tracking API
    tracking_response = requests.post("http://127.0.0.1:2009/track_objects", json=data)
    tracked_objects = tracking_response.json().get("tracked_objects", [])

    print("Tracked objects:")
    pprint.pprint(tracked_objects)

    # Step 5: Draw tracked objects on the image
    output_image = draw_tracked_objects(image, tracked_objects)

    # Step 6: Display and save the results
    cv2.imshow("Tracked Objects", output_image)
    cv2.imwrite("tracked_objects.jpg", output_image)  # Save the result
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
