from sklearn.model_selection import train_test_split

from data_loader import create_dataframe
from dataset import create_dataset
from model import Build_model


print("Loading labels...")

df = create_dataframe(
    "/content/Age-gender-prediction/data/UTKFace"
)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

print("Creating datasets...")

train_ds = create_dataset(
    train_df
)

val_ds = create_dataset(
    val_df
)

print("Building model...")

model = Build_model()

model.compile(
    optimizer="adam",

    loss={
        "age_output":"mae",
        "gender_output":"binary_crossentropy"
    },

    metrics={
        "gender_output":["accuracy"]
    }
)

print("Starting training...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

model.save("/content/Age-gender-prediction/models/age_gender_model.keras")