import json
from pathlib import Path

import numpy as np
import keras
from evaluation import evaluate_model_dubrovnik, evaluate_pte_dubrovnik


def _np_to_float(x):
    if isinstance(x, np.float32):
        return float(x)
    return x


def _numpy_vals_to_floats(d):
    return {k: _np_to_float(v) for k, v in d.items()}


class DubrovnikEvalCallback(keras.callbacks.Callback):
    def __init__(self, eval_model, pte=None):
        self.eval_model = eval_model
        self.pte = pte
    def on_epoch_end(self, epoch, logs=None):
        if self.pte is None:
            logs['dubrovnik_results'] = \
                evaluate_model_dubrovnik(self.eval_model)
        else:
            logs['dubrovnik_results'] = \
                evaluate_pte_dubrovnik(self.pte)
        print()
        print(logs, flush=True)
        print()


class CSVLoggingCallback(keras.callbacks.Callback):
    def __init__(self, path, log_frequency=5):
        self.log_file = path.open('w')
        self.log_frequency = log_frequency

    def record(self, logs):
        logs = _numpy_vals_to_floats(logs)
        print(json.dumps(logs), file=self.log_file, flush=True)

    def on_epoch_begin(self, epoch_n, logs=None):
        self.epoch_n = epoch_n

    def on_epoch_end(self, epoch_n, logs=None):
        logs['epoch_n'] = epoch_n
        self.record(logs)

    def on_batch_end(self, batch_n, logs=None):
        if batch_n % self.log_frequency == 0:
            logs['epoch_n'] = self.epoch_n
            logs['batch_n'] = batch_n
            self.record(logs)


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
