import tensorflow as tf


class VideoEmbedder(tf.keras.layers.Layer):

    def __init__(self, data_set_size=0, embedding_dim=32):

        super(VideoEmbedder, self).__init__()
        self.data_set_size = data_set_size
        self.embedding_dim = embedding_dim

        self.embedding = None
        # self.embed = None

    def build(self, input_shape):
        self.embedding = self.add_variable('embedding_matrix', shape=[self.data_set_size, self.embedding_dim],
                                              initializer=tf.keras.initializers.VarianceScaling)

    def call(self, inputs):
        [123, 124, 674, 234, 1]
        embedded_group = self.embedded_group(inputs)
        return tf.reduce_mean(embedded_group, -1)  # -1 is the last axis

    @tf.function
    def embedded_group(self, inputs):
        @tf.function
        def emmbed_lookup_loop(index):
            return tf.nn.embedding_lookup(self.embedding, index)

        return tf.map_fn(emmbed_lookup_loop, inputs, dtype=tf.float32)


























