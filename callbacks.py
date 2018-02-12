
import keras
from pathlib import Path


class ModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self, path, model, *args, **kwargs):
        self.passed_model = model
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        path = str(path / 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
        super().__init__(path, *args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        self.set_model(self.passed_model)
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_intializer'):
                layer.kernel_intializer = None
        super().on_epoch_end(epoch, logs)
