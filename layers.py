
import keras
from keras.engine.topology import Layer


class LearnScale(Layer):

    def __init__(self, initial_value=1.0, **kwargs):
        self.initial_value = initial_value
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.scale = self.add_weight(
            name='scale',
            shape=[1],
            initializer=keras.initializers.Constant(self.initial_value),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        return x * self.scale

    def compute_output_shape(self, input_shape):
        return input_shape


