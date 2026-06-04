from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from data_loader import load_dataset
from model import Build_model

X, y_age, y_gender = load_dataset('data\raw\UTKFace')

X_train,X_val,age_train,age_val,gender_train,gender_val = train_test_split(X,y_age,y_gender,test_size=0.2,random_state=42)

model = Build_model()

model.compile(
    optimizer='adam',
    loss ={'age_output':'mae',
           'gender_ouput':'binary_crossentropy'},
    metrics = {'gender_ouput':['accuracy']}
)

model.fit(X_train,
          {'age_output':age_train,'gender_output':gender_train},epochs = 10,
          batch_size = 32,
          validation_data=(X_val,{'age_output':age_val,'gender_output':gender_val})
        )

model.save("models/age_gender_model.keras")