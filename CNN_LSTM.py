import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Reshape
from tensorflow.keras.losses import Loss
from tensorflow.keras.regularizers import l2
import os
import xml.etree.ElementTree as ET
import random

def load_dataset(image_dir, xml_dir, classes):
    images = []
    bboxes = []
    class_labels = []
    scores = []  # Add scores list
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_dir, filename)
            images.append(img_path)
            
            # Load corresponding XML
            xml_filename = os.path.splitext(filename)[0] + '.xml'
            xml_path = os.path.join(xml_dir, xml_filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract bounding boxes and labels
            image_bboxes = []
            image_class_labels = []
            image_scores = []  # Add image scores
            for obj in root.findall('object'):
                label = obj.find('name').text
                if label in classes:
                    class_id = classes.index(label)
                    image_class_labels.append(class_id)
                    
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    image_bboxes.append([xmin, ymin, xmax - xmin, ymax - ymin])  # Convert to [x, y, width, height]
                    
                    score = random.uniform(0.5, 1.0)  # Generate a random score between 0.5 and 1.0
                    image_scores.append(score)
                else:
                    print(f"Warning: Label '{label}' not in defined classes.")
            bboxes.append(image_bboxes)
            class_labels.append(image_class_labels)
            scores.append(image_scores)  # Append scores
    return images, {'bbox': bboxes, 'class': class_labels, 'score': scores}  # Include scores in return

# Preprocessing function
def preprocess(image_path, labels, num_classes):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [150, 150])
    image = image / 255.0
    labels['bbox'] = tf.cast(labels['bbox'], tf.float32)  # Ensure bounding boxes are float32
    labels['class'] = tf.one_hot(labels['class'], depth=num_classes)  # Convert class labels to one-hot encoding
    labels['class'] = tf.squeeze(labels['class'])  # Remove the extra dimension
    labels['score'] = tf.cast(labels['score'], tf.float32)  # Ensure scores are float32
    return image, labels

# Define IoU metric
def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Compute intersection
    x1 = tf.maximum(y_true[..., 0], y_pred[..., 0])
    y1 = tf.maximum(y_true[..., 1], y_pred[..., 1])
    x2 = tf.minimum(y_true[..., 0] + y_true[..., 2], y_pred[..., 0] + y_pred[..., 2])
    y2 = tf.minimum(y_true[..., 1] + y_true[..., 3], y_pred[..., 1] + y_pred[..., 3])
    
    intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    
    # Compute union
    area_true = y_true[..., 2] * y_true[..., 3]
    area_pred = y_pred[..., 2] * y_pred[..., 3]
    union = area_true + area_pred - intersection
    
    iou = intersection / (union + tf.keras.backend.epsilon())
    return iou

# Define custom IoU loss function
class IoULoss(Loss):
    def call(self, y_true, y_pred):
        return 1 - iou_metric(y_true, y_pred)

# Example usage
classes = ['Car', 'lorry', 'SUV', 'Autocamper']
num_classes = 4  # Antal klasser

train_images, train_labels = load_dataset('C:/Users/shans/OneDrive/Skrivebord/Vision AI/vehicles/Train/images', 'C:/Users/shans/OneDrive/Skrivebord/Vision AI/vehicles/Train/Labels', classes)
test_images, test_labels = load_dataset('C:/Users/shans/OneDrive/Skrivebord/Vision AI/vehicles/Validate/images', 'C:/Users/shans/OneDrive/Skrivebord/Vision AI/vehicles/Validate/Labels', classes)

# Data augmentation and normalization
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Create tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, {'bbox': train_labels['bbox'], 'class': train_labels['class'], 'score': train_labels['score']}))
train_dataset = train_dataset.map(lambda x, y: (tf.convert_to_tensor(x, dtype=tf.string), y))
train_dataset = train_dataset.map(lambda x, y: preprocess(x, y, num_classes)).batch(32).prefetch(tf.data.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_tensor_slices((test_images, {'bbox': test_labels['bbox'], 'class': test_labels['class'], 'score': test_labels['score']}))
validation_dataset = validation_dataset.map(lambda x, y: (tf.convert_to_tensor(x, dtype=tf.string), y))
validation_dataset = validation_dataset.map(lambda x, y: preprocess(x, y, num_classes)).batch(32).prefetch(tf.data.AUTOTUNE)

# Define model architecture
input_shape = (150, 150, 3)

inputs = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.3)(x)

x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.3)(x)

x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.3)(x)

x = Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.3)(x)

x = Flatten()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)

# Reshape layer to add the timesteps dimension
x = Reshape((1, 256))(x)

# Add LSTM layer
x = LSTM(128, return_sequences=True)(x)
x = LSTM(128)(x)

x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)

# Output layers
bbox_output = Dense(4, activation='linear', name='bbox')(x)  # 4 for [x, y, width, height]
class_output = Dense(num_classes, activation='softmax', name='class')(x)
score_output = Dense(1, activation='sigmoid', name='score')(x)  # Score output

model = Model(inputs=inputs, outputs=[bbox_output, class_output, score_output])
model.compile(
    optimizer='adam',
    loss={
        'bbox': 'mse',  # Mean Squared Error for bounding boxes
        'class': 'categorical_crossentropy',  # Categorical Crossentropy for classification
        'score': 'binary_crossentropy'  # Binary Crossentropy for score
    },
    metrics={
        'bbox': [iou_metric],  # MeanIoU metric for bounding boxes
        'class': 'accuracy'  # Accuracy for classification
    }
)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print(model.summary())

# Train the model
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=[early_stopping]
)

# Save the model
model.save('C:/Users/shans/OneDrive/Skrivebord/Vision AI/Video_CNN_IoU.h5')
