import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Loss

# --- IoU metric og loss ---
def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # [x, y, w, h]
    x1 = tf.maximum(y_true[..., 0], y_pred[..., 0])
    y1 = tf.maximum(y_true[..., 1], y_pred[..., 1])
    x2 = tf.minimum(y_true[..., 0] + y_true[..., 2], y_pred[..., 0] + y_pred[..., 2])
    y2 = tf.minimum(y_true[..., 1] + y_true[..., 3], y_pred[..., 1] + y_pred[..., 3])

    intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    area_true = y_true[..., 2] * y_true[..., 3]
    area_pred = y_pred[..., 2] * y_pred[..., 3]
    union = area_true + area_pred - intersection

    return intersection / (union + tf.keras.backend.epsilon())

class IoULoss(Loss):
    def call(self, y_true, y_pred):
        return 1.0 - iou_metric(y_true, y_pred)

# --- CNN model ---
def build_cnn_detector(input_shape=(150,150,3), num_classes=4):
    inputs = Input(shape=input_shape)

    # Conv blok 1
    x = Conv2D(16, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    # Conv blok 2
    x = Conv2D(32, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    # Conv blok 3
    x = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    # Conv blok 4
    x = Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    # Dense lag
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)

    # Output heads
    bbox_output = Dense(4, activation='linear', name='bbox')  # [x, y, w, h]
    class_output = Dense(num_classes, activation='softmax', name='class')
    score_output = Dense(1, activation='sigmoid', name='score')

    outputs = [bbox_output(x), class_output(x), score_output(x)]
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss={
            'bbox': IoULoss(),
            'class': 'categorical_crossentropy',
            'score': 'binary_crossentropy'
        },
        metrics={
            'bbox': [iou_metric],
            'class': 'accuracy'
        }
    )

    return model

# --- Brug modellen ---
num_classes = 4
model = build_cnn_detector(input_shape=(150,150,3), num_classes=num_classes)
model.summary()
