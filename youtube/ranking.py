import numpy as np
import math
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow import constant, matmul
from tensorflow import float32, int32, unstack
from tensorflow import data, sigmoid, argsort

from mmoe_layer import MMoE

def user_engagement(mmoe_out):
    return Dense(3, activation='relu', name='user_engagement')(mmoe_out)

def user_satisfaction(mmoe_out):
    return Dense(4, activation='sigmoid', name='user_satisfaction')(mmoe_out)

batch_size = 32
data_size = 200
video_data_size = 5
user_data_size = 6
n_tasks = 2

video_data = np.random.rand(data_size, video_data_size)
user_data = np.random.rand(data_size, user_data_size)

label = np.random.rand(data_size, 3)
label2 = np.random.rand(data_size, 4)

dataset = data.Dataset.from_tensor_slices(({'input_1': video_data, 'input_2': user_data},
                                           {'user_engagement': label, 'user_satisfaction': label2}))
dataset = dataset.batch(batch_size).repeat()

videos_embedded_input = Input(shape=(video_data_size,))
user_data_input = Input(shape=(user_data_size,))

full_input = concatenate([videos_embedded_input, user_data_input])
# Tensor("strided_slice_1:0", shape=(None, 4), dtype=float32)
mmoe_layers = MMoE(
        units=4,
        num_experts=8,
        num_tasks=2)(full_input)

tower_1_out = user_satisfaction(mmoe_layers[0])
tower_2_out = user_engagement(mmoe_layers[1])

model = Model([videos_embedded_input, user_data_input], [tower_1_out, tower_2_out])

model.compile(optimizer='adam', loss=['binary_crossentropy', 'mse'], experimental_run_tf_function=False)

model.fit(dataset, epochs=1, steps_per_epoch=data_size//batch_size)

final_out = concatenate([tower_1_out, tower_2_out], dtype=float32)

w = constant(np.array([[12], [.2], [5], [12], [3], [4], [1]]), dtype=float32)
rank = sigmoid(matmul(final_out, w))

sorted = argsort(rank)




