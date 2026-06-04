import os
import cv2
import numpy as np
import pandas as pd

def load_dataset(dataset_path):
    images = []
    ages = []
    genders = []

    for file in os.listdir(dataset_path):
        try:
            age = int(file.split("_")[0])
            gender = int(file.split("_")[1])

            img_path = os.path.join(
                dataset_path,
                file
            )

            img = cv2.imread(img_path)

            if img is None:
                continue
            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            img = cv2.resize(
                img,
                (224,224)
            )

            images.append(img)
            ages.append(age)
            genders.append(gender)

        except:
            pass
    
    X = np.array(images)/255.0

    return (
        X,
        np.array(ages),
        np.array(genders)
    )