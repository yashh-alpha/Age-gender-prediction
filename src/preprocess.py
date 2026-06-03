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