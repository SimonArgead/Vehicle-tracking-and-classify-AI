import cv2
import numpy as np
import tensorflow as tf
from norfair import Detection, Tracker, draw_tracked_objects

'''
 Issue: currently not actually tracking objects as it should.
 Hypothesis:
 1. The trained model and the dataset of only 200 images is to blame.
 2. Try either a transformer based model instead of CNN-LSTM, or try conv3D model instead of conv2D and lstm.
'''

# Registrer mse som en brugerdefineret metrik
@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (150, 150))
    normalized_frame = resized_frame / 255.0
    return normalized_frame

# Initialiser Norfair tracker
tracker = Tracker(distance_function=lambda det, trk: np.linalg.norm(det.points - trk.estimate), distance_threshold=30)

# Indlæs den gemte model
model = tf.keras.models.load_model('C:/Users/shans/OneDrive/Skrivebord/Vision AI/Video_CNN_IoU.h5', custom_objects={'mse': mse})

# Load video
cap = cv2.VideoCapture('C:/Users/shans/OneDrive/Skrivebord/Vision AI/CNN test_2.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Forbered frame til model input
    frame = cv2.resize(frame, (800, 800))
    preprocessed_frame = preprocess_frame(frame)
    frame_expanded = np.expand_dims(preprocessed_frame, axis=0)  # Tilføj batch dimension

    # Forudsig bounding boxes, klasser og scores
    predicted_bboxes, predicted_classes, predicted_scores = model.predict(frame_expanded)

    # Debug print statements
    #print("Predicted bounding boxes:", predicted_bboxes)
    #print("Predicted bounding boxes shape:", predicted_bboxes.shape)
    #print("Predicted classes:", predicted_classes)
    #print("Predicted scores:", predicted_scores)

    # Kontroller og juster formen på predicted_bboxes
    if predicted_bboxes.shape == (1, 4):
        predicted_bboxes = np.expand_dims(predicted_bboxes, axis=0)

    # Debug print statements
    #print("Adjusted predicted bounding boxes shape:", predicted_bboxes.shape)

    # Konverter bounding boxes fra Pascal VOC-format til det format, Norfair forventer
    detections = []
    for bbox, cls, score in zip(predicted_bboxes[0], predicted_classes[0], predicted_scores[0]):
        #print("Current bbox:", bbox)
        #print("Current class:", cls)
        #print("Current score:", score)
        if isinstance(bbox, np.ndarray) and bbox.shape == (4,):
            xmin, ymin, width, height = bbox  # Pascal VOC-format
            xmax = xmin + width
            ymax = ymin + height
            if score >= 0.05:  # Sænk threshold til 0.05
                center_x = xmin + width / 2
                center_y = ymin + height / 2
                detections.append(Detection(points=np.array([center_x, center_y]), scores=np.array([score]), label=int(np.argmax(cls))))
                #print(f"Detection added: {xmin, ymin, xmax, ymax, score, cls}")

    # Debug print statements
    #print("Detections:", detections)

    # Opdater tracker med de nye detektioner
    tracked_objects = tracker.update(detections)

    # Debug print statements
    #print("Tracked objects:", tracked_objects)

    # Tegn bounding boxes på frame
    draw_tracked_objects(frame, tracked_objects)

    # Vis frame
    cv2.imshow('Frame with Bounding Box', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
