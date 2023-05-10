import keras.backend as K
import tensorflow as tf
from tensorflow.keras import initializers, regularizers

from tensorflow.keras import layers
# from keras.layers.convolutional import Convolution1D
import tensorflow.keras as keras
from tensorflow.keras.layers import Flatten, GlobalMaxPooling1D, Dense, Convolution1D, Dropout,\
    GlobalAveragePooling1D, Concatenate, Layer, Add
# from keras.engine.topology import Layer
import numpy as np
import sys


class BaseLayer(keras.layers.Layer):
    def build_layers(self, input_shape):
        shape = input_shape
        for layer in self.layers:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)


class MultiHeadAttention(Layer):

    def __init__(self, heads, head_size, output_dim=None, **kwargs):
        self.heads = heads
        self.head_size = head_size
        self.output_dim = output_dim or heads * head_size
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(
                                          3, input_shape[2], self.head_size),  # shape is 3 *  400 * 100
                                      initializer='uniform',
                                      trainable=True)
        self.dense = self.add_weight(name='dense',
                                     # shape is 400 * 400
                                     shape=(input_shape[2], self.output_dim),
                                     initializer='uniform',
                                     trainable=True)

        super(MultiHeadAttention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        out = []
        for i in range(self.heads):
            WQ = K.dot(x, self.kernel[0])  # shape of wq = 50 * 100
            WK = K.dot(x, self.kernel[1])  # shape of wk= 50 *100
            WV = K.dot(x, self.kernel[2])  # shaep of wv= 50 *100

            # print("WQ.shape",WQ.shape)
            # print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)

            QK = K.batch_dot(WQ, K.permute_dimensions(
                WK, [0, 2, 1]))  # shape of qk = 50 * 50
            QK = QK / (100**0.5)
            QK = K.softmax(QK)

            # print("QK.shape",QK.shape)

            V = K.batch_dot(QK, WV)  # shape of v= 50 *100
            out.append(V)
        out = Concatenate(axis=-1)(out)  # output shape is 50 * 400
        # output shape is (50 * 400) * (400 * 400)
        out = K.dot(out, self.dense)
        return out  # shape is batch * 50 * 400

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


# class DCNNEmbedding(tf.keras.Model):

#     def __init__(self,
#                  nb_filters=50,
#                  FFN_units=512,
#                  nb_classes=2,
#                  dropout_rate=0.1,
#                  name="dcnn"):
#         super(DCNNEmbedding, self).__init__(name=name)

#         self.bigram = layers.Conv1D(filters=nb_filters,
#                                     kernel_size=2,
#                                     padding='valid',
#                                     activation='relu')
#         self.trigram = layers.Conv1D(filters=nb_filters,
#                                      kernel_size=3,
#                                      padding='valid',
#                                      activation='relu')
#         self.fourgram = layers.Conv1D(filters=nb_filters,
#                                       kernel_size=4,
#                                       padding='valid',
#                                       activation='relu')
#         self.pool = layers.GlobalMaxPooling1D()

#         self.dense_1 = layers.Dense(units=FFN_units, activation='relu')
#         self.last_dense = layers.Dense(units=nb_classes, activation='softmax')

#     def call(self, inputs, training):
#         x = self.embed_with_bert(inputs)

#         x_1 = self.bigram(x)
#         x_1 = self.pool(x_1)
#         x_2 = self.trigram(x)
#         x_2 = self.pool(x_2)
#         x_3 = self.fourgram(x)
#         x_3 = self.pool(x_3)

#         merged = tf.concat([x_1, x_2, x_3], axis=-1)
#         merged = self.dense_1(merged)
#         merged = self.dropout(merged, training)
#         output = self.last_dense(merged)

#         return output


class ExpertModule_trm(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.conv_layers = []
        self.pooling_layers = []
        self.shapes = []
        # self.filter_size=[5,3]
        self.filters = 128
        self.layers = []
        self.nb_filters = 50
        self.FFN_units = 400
        self.bigram = layers.Conv1D(filters=self.nb_filters,
                                    kernel_size=2,
                                    padding='valid',
                                    activation='relu')
        self.trigram = layers.Conv1D(filters=self.nb_filters,
                                     kernel_size=3,
                                     padding='valid',
                                     activation='relu')
        self.fourgram = layers.Conv1D(filters=self.nb_filters,
                                      kernel_size=4,
                                      padding='valid',
                                      activation='relu')
        self.pool = layers.GlobalMaxPooling1D()
        self.dense_1 = layers.Dense(units=self.FFN_units, activation='relu')
        self.last_dense = layers.Dense(units=150, activation='relu')
        super(ExpertModule_trm, self).__init__(**kwargs)

    def build(self, input_shape):
        # output shape is batch * 50 *400
        self.layers.append(MultiHeadAttention(4, 100))  # 1
        self.layers.append(Dense(400, activation="relu"))  # 2
        self.layers.append(Dropout(0.5))  # 3
        # 4 # output shape is batch * 50
        self.layers.append(GlobalMaxPooling1D())
        # output shape is batch * 50
        self.layers.append(GlobalAveragePooling1D())  # 5
        self.layers.append(Concatenate())  # 6  # output shape is batch * 100
        self.layers.append(Dropout(0.1))  # 7
        # output is  batch * self.units[0]
        self.layers.append(Dense(self.units[0], activation='relu'))  # 8
        # output is batch * self.units[1]
        self.layers.append(Dense(self.units[1], activation='relu'))
        self.layers.append(Dropout(0.1))

        super(ExpertModule_trm, self).build(input_shape)

    def call(self, inputs):
        xs = self.layers[0](inputs)
        x_1 = self.bigram(xs)   
        x_1 = self.pool(x_1)  # b * 50
        x_2 = self.trigram(xs)  # b * 48 * 50
        x_2 = self.pool(x_2)  # b * 50
        x_3 = self.fourgram(xs)  # b * 47 * 50
        x_3 = self.pool(x_3)  # b * 50
        # output shape is batch_size * 150
        merged = tf.concat([x_1, x_2, x_3], axis=-1)
        xs = self.dense_1(merged)  # output shape is batch * 400
        xs = self.layers[2](xs)
        xs = self.last_dense(xs)  # output shape is batch * 150
        # xs = self.layers[2](xs)
        # xs_max = self.layers[3](xs)
        # xs_avg = self.layers[4](xs)
        # xs = self.layers[5]([xs_max, xs_avg])
        # for layer in self.layers[6:]:
        #     xs = layer(xs)
        return xs

    def compute_output_shape(self, input_shape):
        return input_shape[0]+[self.units[1]]  # output shape is batch *150


class GateModule(BaseLayer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.conv_layers = []
        self.pooling_layers = []
        self.layers = []
        super(GateModule, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layers.append(MultiHeadAttention(4, 100))
        self.layers.append(Dropout(0.1))
        self.layers.append(Dense(400, activation='relu'))
        self.layers.append(GlobalMaxPooling1D())
        self.layers.append(GlobalAveragePooling1D())
        self.layers.append(Concatenate())
        self.layers.append(Dropout(0.1))
        self.layers.append(Dense(self.units[0], activation='relu'))
        self.layers.append(Dense(self.units[0], activation='relu'))
        self.layers.append(Dropout(0.1))
        self.layers.append(Dense(self.units[1], activation='softmax'))

        super(GateModule, self).build(input_shape)

    def call(self, inputs):
        xs = self.layers[0](inputs)
        xs = self.layers[1](xs)
        xs = self.layers[2](xs)
        xs_max = self.layers[3](xs)
        xs_avg = self.layers[4](xs)
        xs = self.layers[5]([xs_max, xs_avg])
        for layer in self.layers[6:]:
            xs = layer(xs)
        return xs

    def compute_output_shape(self, input_shape):
        # (batch 3 for each task)
        return input_shape[0]+[self.units[-1]]


class HSMMBottom(BaseLayer):
    # Hate Speech Mixture Model
    def __init__(self,
                 model_type,
                 non_gate,
                 expert_units,
                 gate_unit=100,
                 task_num=2, expert_num=3,
                 **kwargs):
        self.model_type = model_type
        self.non_gate = non_gate
        self.gate_unit = gate_unit
        self.expert_units = expert_units
        self.task_num = task_num
        self.expert_num = expert_num
        self.experts = []
        self.gates = []
        super(HSMMBottom, self).__init__(**kwargs)

    def build(self, input_shape):
        for i in range(self.expert_num):
            expert = ExpertModule_trm(units=self.expert_units)
            expert.build(input_shape)
            self.experts.append(expert)
        # for i in range(self.task_num):
        #     gate = GateModule(units=[self.gate_unit, self.expert_num])
        #     gate.build(input_shape)
        #     self.gates.append(gate)
        super(HSMMBottom, self).build(input_shape)

    def call(self, inputs):
        # 构建多个expert
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(inputs))

        # 构建多个gate，用来加权expert
        gate_outputs = []
        # if self.non_gate:
        print('111111111111111111111111')
        # batch_size, expert_num, expert_out_dim
        self.expert_output = tf.stack(
            expert_outputs, axis=1)  # batch * 3 * 150
        m1 = tf.reduce_mean(self.expert_output, axis=1)  # batch * 150
        outputs = tf.stack([m1, m1], axis=1)  # batch * 2 * 150
        return outputs

        # else:
        #     for gate in self.gates:
        #         gate_outputs.append(gate(inputs))
        #     # 使用gate对expert进行加权平均
        #     # batch_size, expert_num, expert_out_dim
        #     self.expert_output = tf.stack(expert_outputs, axis=1)
        #     # batch_size, task_num, expert_num
        #     self.gate_output = tf.stack(gate_outputs, axis=1)
        #     # batch_size,task_num,expert_out_dim
        #     outputs = tf.matmul(self.gate_output, self.expert_output)
        #     return outputs

    def compute_output_shape(self, input_shape):
        # batch * 2 * 150
        return [input_shape[0], self.task_num, self.expert_units[-1]]


class HSMMTower(BaseLayer):
    # Hate Speech Mixture Model Tower
    def __init__(self,
                 units,
                 **kwargs):
        self.units = units
        self.layers = []
        super(HSMMTower, self).__init__(**kwargs)

    def build(self, input_shape):
        # for unit in self.units[:-1]:# b * 512
        self.layers.append(Dense(128,kernel_regularizer=regularizers.l2(0.01),activation="relu")) # b* 150
        self.layers.append(Dense(self.units[-1], activation='softmax')) # b * 2
        super(HSMMTower, self).build(input_shape)

    def call(self, inputs):  # b * 2
        xs = self.layers[0](inputs)
        xs = self.layers[1](xs)
        return xs

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.units[-1]]  # output = batch *2
