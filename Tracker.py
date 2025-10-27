import cv2
import numpy as np
import tensorflow as tf
from norfair import Detection, Tracker

# Custom metric
@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (150, 150))
    normalized_frame = resized_frame / 255.0
    return normalized_frame

# Utility function: scaling bbox
def scale_bbox(bbox, input_size, output_size):
    x, y, w, h = bbox
    scale_x = output_size[0] / input_size[0]
    scale_y = output_size[1] / input_size[1]
    return [
        x * scale_x,
        y * scale_y,
        w * scale_x,
        h * scale_y
    ]

# Tracker setup
def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

tracker = Tracker(distance_function=euclidean_distance, distance_threshold=30)

# Load model
model = tf.keras.models.load_model(
    "C:/Users/shans/OneDrive/Skrivebord/Vision AI/Video_CNN_IoU.h5",
    custom_objects={"mse": mse}
)

def scale_bbox(bbox, input_size, output_size):
    """
    Scale a bbox from model-input size to original frame-size.
    bbox: [x, y, w, h] i input_size coordinates
    input_size: (w_in, h_in) fx (150,150)
    output_size: (w_out, h_out) fx (1920,1080)
    """
    x, y, w, h = bbox
    scale_x = output_size[0] / input_size[0]
    scale_y = output_size[1] / input_size[1]
    return [
        x * scale_x,
        y * scale_y,
        w * scale_x,
        h * scale_y
    ]

# Load video
cap = cv2.VideoCapture("C:/Users/shans/OneDrive/Skrivebord/Vision AI/CNN test_2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]

    # Preprocess to model
    preprocessed_frame = preprocess_frame(frame)
    frame_expanded = np.expand_dims(preprocessed_frame, axis=0)

    # Model prediction
    predicted_bboxes, predicted_classes, predicted_scores = model.predict(frame_expanded, verbose=0)

    # Extract outputs
    bbox = predicted_bboxes[0]        # (4,)
    cls_vector = predicted_classes[0] # (num_classes,)
    score = predicted_scores[0][0]    # scalar

    # Scale bbox to original frame-size
    xmin, ymin, width, height = scale_bbox(bbox, (150,150), (orig_w, orig_h))

    detections = []
    if score >= 0.3:  # threshold
        center_x = xmin + width / 2
        center_y = ymin + height / 2
        detections.append(
            Detection(
                points=np.array([[center_x, center_y]]),
                scores=np.array([score]),
                label=int(np.argmax(cls_vector))
            )
        )

    # Opdate tracker
    tracked_objects = tracker.update(detections)

    # Draw results
    for tobj in tracked_objects:
        cx, cy = tobj.estimate[0]

        # Draw centerpoint
        cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)

        # Draw bbox (from latest detection)
        cv2.rectangle(frame,
                      (int(xmin), int(ymin)),
                      (int(xmin + width), int(ymin + height)),
                      (0, 255, 0), 2)

        label = tobj.label if hasattr(tobj, "label") else -1
        cv2.putText(frame,
                    f"cls {label}, conf {score:.2f}",
                    (int(xmin), int(ymin) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Frame with Bounding Box", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
