"""
Copyright (C) 2019  F.Hoffmann-La Roche Ltd
Copyright (C) 2019  Patrick Schwab, ETH Zurich

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
from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.regularizers import L1L2
from keras.layers import Dense, Input, Dropout
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.normalization import BatchNormalization


class ModelBuilder(object):
    @staticmethod
    def compile_model(model, learning_rate, optimizer="adam", loss_weights=list([1.0, 1.0, 1.0, 1.0]),
                      main_loss="mse", extra_loss=None, metrics={}, gradient_clipping_threshold=-1):

        losses = main_loss

        if loss_weights is not None:
            losses = [losses] * len(loss_weights)

        if extra_loss is not None:
            if isinstance(extra_loss, list):
                for i in range(1, 1 + len(extra_loss)):
                    losses[i] = extra_loss[i - 1]
            else:
                losses[1] = extra_loss

        if optimizer == "rmsprop":
            opt = RMSprop(lr=learning_rate, clipvalue=gradient_clipping_threshold)
        elif optimizer == "sgd":
            opt = SGD(lr=learning_rate, nesterov=True, momentum=0.9, clipvalue=gradient_clipping_threshold)
        else:
            opt = Adam(lr=learning_rate, clipvalue=gradient_clipping_threshold)

        model.compile(loss=losses,
                      loss_weights=loss_weights,
                      optimizer=opt,
                      metrics=metrics)
        return model

    @staticmethod
    def build_mlp(input_dim, output_dim, l2_weight=0.0,
                  num_units=36, learning_rate=0.0001,
                  p_dropout=0.0, num_layers=1, with_bn=False,
                  activation="relu",
                  **kwargs):
        if isinstance(input_dim, list):
            tasks_input = Input(shape=input_dim[0])
        else:
            tasks_input = Input(shape=(input_dim,))

        last_layer = tasks_input
        for i in range(num_layers):

            last_layer = Dense(num_units,
                               activation=activation,
                               kernel_regularizer=L1L2(l2=l2_weight),
                               bias_regularizer=L1L2(l2=l2_weight))(last_layer)

            if with_bn:
                last_layer = BatchNormalization(beta_regularizer=L1L2(l2=l2_weight),
                                                gamma_regularizer=L1L2(l2=l2_weight))(last_layer)

            if not np.isclose(p_dropout, 0.0):
                last_layer = Dropout(p_dropout)(last_layer)

        input_layer = tasks_input

        last_layer = Dense(output_dim,
                           activation="sigmoid",
                           kernel_regularizer=L1L2(l2=l2_weight),
                           bias_regularizer=L1L2(l2=l2_weight))(last_layer)

        model = Model(inputs=input_layer, outputs=last_layer)
        model.summary()
        return ModelBuilder.compile_model(model,
                                          learning_rate=learning_rate,
                                          main_loss="binary_crossentropy",
                                          loss_weights=[1.0],
                                          metrics=["accuracy"])
