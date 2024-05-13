import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import keras.backend as K

# Problem creation
def test_knapsack(x_weights, x_prices, x_capacity, picks):
    total_price = np.dot(x_prices, picks)
    total_weight = np.dot(x_weights, picks)
    return total_price, max(total_weight - x_capacity, 0)

def brute_force_knapsack(x_weights, x_prices, x_capacity):
    picks_space = 2 ** x_weights.shape[0]
    best_price = 0
    best_picks = None
    for p in range(picks_space):
        picks = np.zeros((x_weights.shape[0]))
        for i, c in enumerate("{0:b}".format(p)[::-1]):
            picks[i] = int(c)
        price, violation = test_knapsack(x_weights, x_prices, x_capacity, picks)
        if violation == 0:
            if price > best_price:
                best_price = price
                best_picks = picks
    return best_price, best_picks

def create_knapsack(item_count=5):
    x_weights = np.random.randint(1, 15, item_count)
    x_prices = np.random.randint(1, 10, item_count)
    x_capacity = np.random.randint(15, 50)
    _, y_best_picks = brute_force_knapsack(x_weights, x_prices, x_capacity)
    return x_weights, x_prices, x_capacity, y_best_picks

# Model creation
def create_knapsack_model(item_count=5):
    inputs = Input(shape=(item_count,))
    outputs = Dense(item_count, activation='sigmoid')(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='mse')
    return model

# Dataset creation
def create_knapsack_dataset(count, item_count=5):
    x = []
    y = []
    for _ in range(count):
        p = create_knapsack(item_count)
        x.append(p[0])  # Weights
        y.append(p[3])  # Best picks
    return np.array(x), np.array(y)

# Train the model
def train_knapsack_model(model, train_x, train_y, epochs=100, batch_size=32):
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.1)
    return history

# Evaluation metrics
def overpricing_metric(y_true, y_pred):
    return K.mean(K.sum(y_pred, axis=-1) - K.sum(y_true, axis=-1))

# Main
item_count = 5
train_x, train_y = create_knapsack_dataset(10000, item_count)
test_x, test_y = create_knapsack_dataset(200, item_count)

model = create_knapsack_model(item_count)
history = train_knapsack_model(model, train_x, train_y)

# Evaluate the model
train_loss = model.evaluate(train_x, train_y)
test_loss = model.evaluate(test_x, test_y)
print("Train Loss:", train_loss)
print("Test Loss:", test_loss)

# Print learning progress
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Progress')
plt.legend()
plt.show()
