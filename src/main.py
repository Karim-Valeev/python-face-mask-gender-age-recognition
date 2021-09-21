import os
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import datetime
from src.utils.GenderAndAgePredictor import GenderAndAgePredictor
from src.utils.Loader import Loader

if __name__ == '__main__':
    loader = Loader()
    images = loader.load_images_from_folder("input")

    predictor = GenderAndAgePredictor(
        gender_model_path=os.path.abspath("models/gender_model.model"),
        age_model_path=os.path.abspath("models/age_model.model")
    )

    prototxt = os.path.sep.join([os.path.abspath("models"), "deploy.prototxt"])
    weights = os.path.sep.join([os.path.abspath("models"), "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxt, weights)

    MaskModel = load_model(os.path.abspath("models/mask_detector.model"))

    for img in images:
        orig = img.copy()
        (h, w) = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        confidence = detections[0, 0, 0, 2]
        if confidence > 0.5:
            dt_string = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = img[startY:endY, startX:endX]
            try:
                face_for_mask = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            except BaseException as e:
                print(e)
                continue

            face_for_mask = cv2.resize(face_for_mask, (224, 224))
            face_for_mask = img_to_array(face_for_mask)
            face_for_mask = preprocess_input(face_for_mask)
            face_for_mask = np.expand_dims(face_for_mask, axis=0)

            (mask, withoutMask) = MaskModel.predict(face_for_mask)[0]
            label = "Mask" if mask > withoutMask else "Intruder"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            if mask < withoutMask:
                face_for_age_gender = cv2.resize(face, (48, 48))
                face_for_age_gender = cv2.cvtColor(face_for_age_gender, cv2.COLOR_BGR2GRAY)
                data = ''
                for i in range(48):
                    for j in range(48):
                        data += " " + str(face_for_age_gender[i][j])
                x = data[1:]
                img_arr = np.array(x.split(), dtype="float32") / 255
                X = img_arr.reshape(1, 48, 48, 1)
                gender, age = predictor.predict_gender_and_age(X)
                label += f"Gender: {gender} Age: {age}"
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                cv2.putText(img, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 3)
                cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
                cv2.imwrite(os.path.join(os.path.abspath("intruder"), f"intruder_{gender}_{int(age)}.jpg"), img)
            else:
                cv2.putText(img, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 3)
                cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
                cv2.imwrite(os.path.join(os.path.abspath("in_mask"), f"in_mask_{dt_string}.jpg"), img)
