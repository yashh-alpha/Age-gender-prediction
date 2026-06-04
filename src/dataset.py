import tensorflow as tf

IMG_SIZE = 224


def load_image(path, age, gender):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = image / 255.0

    return image, {
        "age_output": age,
        "gender_output": gender
    }


def create_dataset(df, batch_size=32):

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            df["image_path"].values,
            df["age"].values,
            df["gender"].values
        )
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset