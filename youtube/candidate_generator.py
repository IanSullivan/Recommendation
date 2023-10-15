import numpy as np
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow import float32, int32
from tensorflow import data

from Video_Embedder import VideoEmbedder

batch_size = 32
data_size = 10000
history_size = 5
user_data_size = 8

videos_array = np.random.randint(data_size, size=(data_size, history_size))
user_data = np.random.rand(data_size, user_data_size)
target_video = np.random.randint(data_size, size=(data_size, 1))
one_hot_target = to_categorical(target_video, num_classes=data_size)

dataset = data.Dataset.from_tensor_slices(({"input_1": videos_array, "input_2": user_data}, one_hot_target))
dataset = dataset.batch(batch_size).repeat()

videos_input = Input(shape=(history_size,), dtype=int32)
user_data_input = Input(shape=(user_data_size,))

embeded_video_history = VideoEmbedder(data_set_size=data_size)(videos_input)

full_input = concatenate([embeded_video_history, user_data_input])

hidden1 = Dense(512, activation='relu')(full_input)

out = Dense(data_size, activation='softmax')(hidden1)

model = Model([videos_input, user_data_input], out)

model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit(dataset, epochs=10, steps_per_epoch=data_size//batch_size)

model.get_layer('video_embedder').embedding.numpy()










