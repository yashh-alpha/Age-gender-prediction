import os
import pandas as pd

def create_dataframe(dataset_path):

    data = []

    for file in os.listdir(dataset_path):

        try:
            age = int(file.split("_")[0])
            gender = int(file.split("_")[1])

            data.append({
                "image_path": os.path.join(dataset_path, file),
                "age": age,
                "gender": gender
            })

        except:
            pass

    return pd.DataFrame(data)