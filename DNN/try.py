from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply, Add
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.models import Model

# Define input layer
inputs = Input(shape=(150,))

# Define three parallel dense layers for processing the input features from the three experts
dense_1 = Dense(400, activation=relu)(inputs)
dense_2 = Dense(400, activation=relu)(inputs)
dense_3 = Dense(400, activation=relu)(inputs)

# Define three parallel dense layers for evaluating the contribution of each expert
score_1 = Dense(1, activation=sigmoid)(dense_1)
score_2 = Dense(1, activation=sigmoid)(dense_2)
score_3 = Dense(1, activation=sigmoid)(dense_3)

# Concatenate the contribution scores of the three experts
scores = Concatenate(axis=1)([score_1, score_2, score_3])

# Multiply the contribution scores by the expert outputs element-wise
weighted_1 = Multiply()([dense_1, scores[:, 0]])
weighted_2 = Multiply()([dense_2, scores[:, 1]])
weighted_3 = Multiply()([dense_3, scores[:, 2]])

# Add the weighted outputs to get the final output tensor
weighted_sum = Add()([weighted_1, weighted_2, weighted_3])

# Apply a final dense layer to get the final output tensor
outputs = Dense(2)(weighted_sum)

# Define the model
model = Model(inputs=inputs, outputs=outputs)
