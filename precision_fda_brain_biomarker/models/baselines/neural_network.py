"""
Copyright (C) 2019  F.Hoffmann-La Roche Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import os
import json
import numpy as np
from keras.callbacks import *
from keras.models import load_model
from sklearn.base import BaseEstimator, ClassifierMixin
from precision_fda_brain_biomarker.models.model_builder import ModelBuilder
from precision_fda_brain_biomarker.models.baselines.base_model import BaseModel, HyperparamMixin


class NeuralNetwork(BaseModel, BaseEstimator, ClassifierMixin, HyperparamMixin):
    def __init__(self, l2_weight=0.0,
                 num_units=36, learning_rate=0.0001,
                 p_dropout=0.0, num_layers=1, with_bn=False,
                 activation="relu",
                 monitor="val_loss",
                 num_epochs=300,
                 early_stopping_patience=12,
                 verbose=2,
                 batch_size=32,
                 best_model_path="best_model.h5"):
        self.l2_weight = l2_weight
        self.num_units = num_units
        self.learning_rate = learning_rate
        self.p_dropout = p_dropout
        self.num_layers = num_layers
        self.with_bn = with_bn
        self.activation = activation
        self.monitor = monitor
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose
        self.batch_size = batch_size
        self.best_model_path = best_model_path
        self.model = None

    @staticmethod
    def get_hyperparameter_ranges():
        ranges = {
            "dropout": [0.0, 0.35],
            "num_layers": [1, 3],
            "l2_weight": (0.0, 0.0001, 0.00001),
            "batch_size": (8, 16, 32, 64, 128),
            "num_units": (16, 32, 64, 128, 256),
            "learning_rate": (0.0001, 0.001, 0.01),
            "with_bn": (True, False),
            "activation": ("relu", "selu", "elu")
        }
        return ranges

    def fit(self, x, y, validation_data=None):
        model_params = {
            "input_dim": x.shape[-1],
            "output_dim": 1 if len(y.shape) == 1 else y.shape[-1]
        }
        model_params.update(self.get_params())
        self.model = ModelBuilder.build_mlp(**model_params)

        history = self.model.fit(x=x, y=y, batch_size=self.batch_size,
                                 epochs=self.num_epochs,
                                 validation_data=validation_data,
                                 verbose=self.verbose,
                                 callbacks=[
                                     EarlyStopping(monitor=self.monitor,
                                                   patience=self.early_stopping_patience),
                                     ModelCheckpoint(self.best_model_path,
                                                     monitor=self.monitor,
                                                     save_weights_only=True),
                                     ReduceLROnPlateau(monitor=self.monitor, factor=np.sqrt(0.1),
                                                       cooldown=0, patience=9, min_lr=1e-5, verbose=1)
                                 ])

        # Reset to best encountered weights.
        self.model.load_weights(self.best_model_path)
        return history

    def fit_generator(self, train_generator, train_steps, val_generator, val_steps, input_dim, output_dim):
        model_params = {
            "input_dim": input_dim,
            "output_dim": output_dim
        }
        model_params.update(self.get_params())
        self.model = ModelBuilder.build_mlp(**model_params)

        history = self.model.fit_generator(generator=train_generator,
                                           steps_per_epoch=train_steps,
                                           epochs=self.num_epochs,
                                           validation_data=val_generator,
                                           validation_steps=val_steps,
                                           verbose=self.verbose,
                                           callbacks=[
                                               EarlyStopping(monitor=self.monitor,
                                                             patience=self.early_stopping_patience),
                                               ModelCheckpoint(self.best_model_path,
                                                               monitor=self.monitor,
                                                               save_weights_only=True),
                                               ReduceLROnPlateau(monitor=self.monitor, factor=np.sqrt(0.1),
                                                                 cooldown=0, patience=9, min_lr=1e-5, verbose=1)
                                           ])

        # Reset to best encountered weights.
        self.model.load_weights(self.best_model_path)
        return history

    def predict(self, x):
        if self.model is None:
            raise AssertionError("Model must be fit before calling predict.")

        return self.model.predict(x)

    def predict_proba(self, x):
        y_pred = self.predict(x)
        return np.concatenate([1-y_pred, y_pred], axis=-1)

    def get_config(self):
        config = {
            "l2_weight": self.l2_weight,
            "num_units": self.num_units,
            "learning_rate": self.learning_rate,
            "p_dropout": self.p_dropout,
            "num_layers": self.num_layers,
            "with_bn": self.with_bn,
            "activation": self.activation,
            "monitor": self.monitor,
            "num_epochs": self.num_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "verbose": self.verbose,
            "batch_size": self.batch_size,
            "best_model_path": self.best_model_path,
        }
        return config

    @staticmethod
    def get_config_file_name():
        return "config.json"

    @staticmethod
    def load(file_path):
        directory_path = os.path.dirname(os.path.abspath(file_path))
        config_file_name = NeuralNetwork.get_config_file_name()
        config_file_path = os.path.join(directory_path, config_file_name)
        with open(config_file_path, "r") as fp:
            config = json.load(fp)

        l2_weight = config["l2_weight"]
        num_units = config["num_units"]
        learning_rate = config["learning_rate"]
        p_dropout = config["p_dropout"]
        num_layers = config["num_layers"]
        with_bn = config["with_bn"]
        activation = config["activation"]
        monitor = config["monitor"]
        num_epochs = config["num_epochs"]
        early_stopping_patience = config["early_stopping_patience"]
        verbose = config["verbose"]
        batch_size = config["batch_size"]
        best_model_path = config["best_model_path"]

        model = load_model(file_path)
        instance = NeuralNetwork(l2_weight=l2_weight,
                                 num_units=num_units,
                                 learning_rate=learning_rate,
                                 p_dropout=p_dropout,
                                 num_layers=num_layers,
                                 with_bn=with_bn,
                                 activation=activation,
                                 monitor=monitor,
                                 num_epochs=num_epochs,
                                 early_stopping_patience=early_stopping_patience,
                                 verbose=verbose,
                                 batch_size=batch_size,
                                 best_model_path=best_model_path)

        instance.model = model
        return instance

    def save(self, file_path, overwrite=True):
        BaseModel.save_config(file_path, self.get_config(), self.get_config_file_name(), overwrite, NeuralNetwork)
        self.model.save(file_path)

    @staticmethod
    def get_save_file_type():
        return ".h5"
