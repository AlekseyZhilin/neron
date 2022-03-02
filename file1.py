import numpy as np
import os
import shutil
from imageai.Detection.Custom import CustomObjectDetection
from imageai.Detection.Custom import DetectionModelTrainer
from pathlib import Path

os.makedirs('imageai/data/train/images', exist_ok=True)
os.makedirs('imageai/data/train/annotations', exist_ok=True)
os.makedirs('imageai/data/validation/images', exist_ok=True)
os.makedirs('imageai/data/validation/annotations', exist_ok=True)
os.makedirs('imageai/data/test/images', exist_ok=True)
os.makedirs('imageai/data/test/annotations', exist_ok=True)

root_annots_path = 'annotations'
root_images_path = 'images'

annots_path = sorted([i for i in Path(root_annots_path).glob('*.xml')])
images_path = sorted([i for i in Path(root_images_path).glob('*.png')])
classes = np.array(["black-king", "white-king",
                    "black-pawn", "white-pawn",
                    "white-knight", "black-knight",
                    "black-bishop", "white-bishop",
                    "white-rook", "black-rook",
                    "black-queen", "white-queen"])

n_imgs = len(images_path)
n_split = n_imgs // 6
print(f'Количество изображений = {n_imgs}')

# Создание папок с изображениями
for i, (annot_path, img_path) in enumerate(zip(annots_path, images_path)):
    if i > n_imgs:
        break
    # train-val-test split
    if i < n_split:
        shutil.copy(img_path, 'imageai/data/test/images/' + img_path.parts[-1])
        shutil.copy(annot_path, 'imageai/data/test/annotations/' + annot_path.parts[-1])
    elif n_split <= i < n_split*2:
        shutil.copy(img_path, 'imageai/data/validation/images/' + img_path.parts[-1])
        shutil.copy(annot_path, 'imageai/data/validation/annotations/' + annot_path.parts[-1])
    else:
        shutil.copy(img_path, 'imageai/data/train/images/' + img_path.parts[-1])
        shutil.copy(annot_path, 'imageai/data/train/annotations/' + annot_path.parts[-1])


# Training model
trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="imageai/data/")
trainer.setTrainConfig(object_names_array=classes,
                       batch_size=8,
                       num_experiments=10,
                       train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()


# Evaluating model
model_path = sorted(list(Path('imageai/data/models/').iterdir()))[-1]

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="imageai/data/")
metrics = trainer.evaluateModel(model_path=model_path,
                                json_path="imageai/data/json/detection_config.json",
                                iou_threshold=0.2,
                                object_threshold=0.3,
                                nms_threshold=0.5)

# Testing model
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
detector.setJsonPath("imageai/data/json/detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(minimum_percentage_probability=60,
                                             input_image="imageai/data/test/images/chess16.png",
                                             output_image_path="detected.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])


# Epoch 1/10
# 56/56 [==============================] - ETA: 0s - loss: 142.9094 - yolo_layer_loss: 22.1859 - yolo_layer_1_loss: 30.9418 - yolo_layer_2_loss: 78.2075 WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
# 56/56 [==============================] - 637s 11s/step - loss: 142.9094 - yolo_layer_loss: 22.1859 - yolo_layer_1_loss: 30.9418 - yolo_layer_2_loss: 78.2075 - lr: 1.0000e-04
# Epoch 2/10
# 56/56 [==============================] - ETA: 0s - loss: 70.6401 - yolo_layer_loss: 12.3148 - yolo_layer_1_loss: 8.8943 - yolo_layer_2_loss: 37.8555 WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
# 56/56 [==============================] - 625s 11s/step - loss: 70.6401 - yolo_layer_loss: 12.3148 - yolo_layer_1_loss: 8.8943 - yolo_layer_2_loss: 37.8555 - lr: 1.0000e-04
# Epoch 3/10
# 56/56 [==============================] - ETA: 0s - loss: 65.1396 - yolo_layer_loss: 13.6245 - yolo_layer_1_loss: 6.3524 - yolo_layer_2_loss: 33.5950 WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
# 56/56 [==============================] - 872s 16s/step - loss: 65.1396 - yolo_layer_loss: 13.6245 - yolo_layer_1_loss: 6.3524 - yolo_layer_2_loss: 33.5950 - lr: 1.0000e-04
# Epoch 4/10
# 56/56 [==============================] - ETA: 0s - loss: 55.9030 - yolo_layer_loss: 11.3787 - yolo_layer_1_loss: 4.9419 - yolo_layer_2_loss: 28.0319 WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
# 56/56 [==============================] - 663s 12s/step - loss: 55.9030 - yolo_layer_loss: 11.3787 - yolo_layer_1_loss: 4.9419 - yolo_layer_2_loss: 28.0319 - lr: 1.0000e-04
# Epoch 5/10
# 56/56 [==============================] - ETA: 0s - loss: 54.6526 - yolo_layer_loss: 10.6260 - yolo_layer_1_loss: 5.5759 - yolo_layer_2_loss: 26.9273 WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
# 56/56 [==============================] - 793s 14s/step - loss: 54.6526 - yolo_layer_loss: 10.6260 - yolo_layer_1_loss: 5.5759 - yolo_layer_2_loss: 26.9273 - lr: 1.0000e-04
# Epoch 6/10
# 56/56 [==============================] - ETA: 0s - loss: 51.7159 - yolo_layer_loss: 10.6214 - yolo_layer_1_loss: 5.4667 - yolo_layer_2_loss: 24.1372 WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
# 56/56 [==============================] - 724s 13s/step - loss: 51.7159 - yolo_layer_loss: 10.6214 - yolo_layer_1_loss: 5.4667 - yolo_layer_2_loss: 24.1372 - lr: 1.0000e-04
# Epoch 7/10
# 56/56 [==============================] - ETA: 0s - loss: 48.5552 - yolo_layer_loss: 9.2440 - yolo_layer_1_loss: 5.2144 - yolo_layer_2_loss: 22.6582 WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
# 56/56 [==============================] - 699s 13s/step - loss: 48.5552 - yolo_layer_loss: 9.2440 - yolo_layer_1_loss: 5.2144 - yolo_layer_2_loss: 22.6582 - lr: 1.0000e-04
# Epoch 8/10
# 56/56 [==============================] - ETA: 0s - loss: 45.4307 - yolo_layer_loss: 8.2942 - yolo_layer_1_loss: 4.3146 - yolo_layer_2_loss: 21.4567 WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
# 56/56 [==============================] - 615s 11s/step - loss: 45.4307 - yolo_layer_loss: 8.2942 - yolo_layer_1_loss: 4.3146 - yolo_layer_2_loss: 21.4567 - lr: 1.0000e-04
# Epoch 9/10
# 56/56 [==============================] - ETA: 0s - loss: 44.3524 - yolo_layer_loss: 8.1406 - yolo_layer_1_loss: 4.6560 - yolo_layer_2_loss: 20.2825 WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
# 56/56 [==============================] - 659s 12s/step - loss: 44.3524 - yolo_layer_loss: 8.1406 - yolo_layer_1_loss: 4.6560 - yolo_layer_2_loss: 20.2825 - lr: 1.0000e-04
# Epoch 10/10
# 56/56 [==============================] - ETA: 0s - loss: 43.9061 - yolo_layer_loss: 8.6329 - yolo_layer_1_loss: 5.3756 - yolo_layer_2_loss: 18.7118 WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
# 56/56 [==============================] - 847s 15s/step - loss: 43.9061 - yolo_layer_loss: 8.6329 - yolo_layer_1_loss: 5.3756 - yolo_layer_2_loss: 18.7118 - lr: 1.0000e-04
# Starting Model evaluation....
# Evaluating over 0 samples taken from imageai/data/validation
# Training over 55 samples  given at imageai/data/train
# Validation samples were not provided.
# Please, check your validation samples are correctly provided:
# 	Annotations: imageai/data/validation\annotations
# 	Images: imageai/data/validation\images
# Model File:  imageai\data\models\detection_model-ex-010--loss-0043.906.h5
#
# Evaluation samples:  0
# Using IoU:  0.2
# Using Object Threshold:  0.3
# Using Non-Maximum Suppression:  0.5
# black-bishop: 0.0000
# black-king: 0.0000
# black-knight: 0.0000
# black-pawn: 0.0000
# black-queen: 0.0000
# black-rook: 0.0000
# white-bishop: 0.0000
# white-king: 0.0000
# white-knight: 0.0000
# white-pawn: 0.0000
# white-queen: 0.0000
# white-rook: 0.0000
# mAP: 0.0000
# ===============================
# WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
# white-knight  :  67.04927682876587  :  [131, 75, 210, 105]
# white-knight  :  62.29153275489807  :  [10, 93, 78, 125]
# white-knight  :  62.64277696609497  :  [319, 128, 385, 159]
# white-knight  :  63.68297338485718  :  [87, 53, 120, 70]
# white-pawn  :  88.6900007724762  :  [87, 53, 120, 70]
# white-pawn  :  69.80196237564087  :  [312, 46, 357, 78]
# white-pawn  :  66.72058701515198  :  [151, 69, 191, 98]
# white-knight  :  62.30002045631409  :  [157, 76, 186, 98]
# black-pawn  :  60.9305739402771  :  [198, 108, 240, 154]
# black-rook  :  66.41905903816223  :  [198, 108, 240, 154]
# white-pawn  :  61.484986543655396  :  [255, 122, 274, 149]
# white-pawn  :  85.81804633140564  :  [161, 178, 185, 197]
# white-knight  :  68.39227676391602  :  [160, 180, 185, 202]
#
# Process finished with exit code 0