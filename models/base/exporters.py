import numpy as np
import tensorflow as tf


def tf_summary_image(tensor):
    import io
    from PIL import Image

    tensor = tensor.astype(np.uint8)
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(
        height=height, width=width, colorspace=channel, encoded_image_string=image_string)
