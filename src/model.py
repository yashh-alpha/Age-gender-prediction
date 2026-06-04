from keras.models import Model
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout

def Build_model():
    
    base_model = MobileNetV2(
        weights = 'imagenet',
        include_top = False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(128,activation = 'relu')(x)
    x = Dropout(0.3)(x)

    age_output = Dense(1,activation = 'linear', name = 'age_output')(x)

    gender_output = Dense(1,activation = 'sigmoid', name = 'gender_output')(x)


    model = Model(inputs= base_model.input,outputs = [age_output,gender_output])

    return model
