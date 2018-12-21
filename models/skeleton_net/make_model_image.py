from tensorflow.python.keras.utils.vis_utils import plot_model
from .model import make_model

model = make_model(input_shape=(112, 112, 3), n_parallel_paths=2, n_classes=48)
model.summary(line_length=100)
plot_model(model, to_file='model.png', show_shapes=True)