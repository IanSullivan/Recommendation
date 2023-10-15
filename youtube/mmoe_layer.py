import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow import tensordot, add, multiply, matmul, reduce_sum
from tensorflow import float32, expand_dims, map_fn


class MMoE(Layer):

    def __init__(self, units, num_experts, num_tasks):

        self.units = units
        self.n_experts = num_experts
        self.n_tasks = num_tasks

        self.gate_weights = None
        self.gate_bias = None

        self.experts_weights = None
        self.experts_bias = None

        self.gate_out = None
        self.experts_out = None

        super(MMoE, self).__init__()

    def build(self, input_shape):

        assert input_shape is not None and len(input_shape) >= 2

        input_dimension = input_shape[-1]

        self.gate_weights = self.add_variable('gate_weight', shape=[self.n_tasks, input_dimension, self.n_experts],
                                              initializer=tf.keras.initializers.VarianceScaling)

        self.gate_bias = self.add_variable('gate_bias', shape=[self.n_tasks, self.n_experts],
                                           initializer=tf.keras.initializers.VarianceScaling)

        self.experts_weights = self.add_variable('experts_weights', shape=[input_dimension, self.units, self.n_experts],
                                                 initializer=tf.keras.initializers.VarianceScaling)

        self.experts_bias = self.add_variable('experts_bias', shape=[self.n_experts],
                                              initializer=tf.keras.initializers.VarianceScaling)

        self.l2 = tf.keras.layers.Dense(self.n_experts)

        super(MMoE, self).build(input_shape)

    def call(self, inputs):
        self.experts_out = self.expert_layer(inputs)
        # self.experts_out2 = self.l2(self.experts_out)
        self.gate_out = self.gate_in(inputs)
        return self.mmoe_out(self.experts_out)

    @tf.function
    def expert_layer(self, inputs):
        experts_out = tf.tensordot(inputs, self.experts_weights, axes=1)
        experts_out = tf.add(experts_out, self.experts_bias)
        return tf.nn.leaky_relu(experts_out)

    def gate_in(self, inputs):
        def layer(index):
            weight_out = tf.matmul(inputs, self.gate_weights[index])
            bias_add = tf.add(weight_out, self.gate_bias[index])
            return tf.nn.softmax(bias_add)

        t_range = tf.range(self.n_tasks)
        return tf.map_fn(layer, t_range, dtype=tf.float32)

    def mmoe_out(self, experts_out):

        def _mmoeout(gate_out_i):
            expanded_gate_output = tf.expand_dims(gate_out_i, axis=1)
            repeated_expanded_gate = tf.keras.backend.repeat_elements(expanded_gate_output, self.units, axis=1)
            weighted_expert_output = tf.multiply(experts_out, repeated_expanded_gate)
            return tf.reduce_sum(weighted_expert_output, -1)

        return tf.map_fn(_mmoeout, self.gate_out)
