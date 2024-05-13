import numpy as np
from keras.models import Model
from keras.layers import Dense, Activation, Lambda, Input, Concatenate, Multiply
from keras.metrics import binary_accuracy
from keras.losses import binary_crossentropy
import keras.backend as K

# problem creation

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
            picks[i] = c
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

x_weights, x_prices, x_capacity, y_best_picks = create_knapsack()
print("Weights:", x_weights)
print("Prices:", x_prices)
print("Capacity:", x_capacity)
print("Best picks:", y_best_picks)


def metric_overprice(input_prices):
    def overpricing(y_true, y_pred):
        y_pred = K.round(y_pred)
        return K.mean(K.batch_dot(y_pred, input_prices, 1) - K.batch_dot(y_true, input_prices, 1))

    return overpricing


def metric_space_violation(input_weights, input_capacity):
    def space_violation(y_true, y_pred):
        y_pred = K.round(y_pred)
        return K.mean(K.maximum(K.batch_dot(y_pred, input_weights, 1) - input_capacity, 0))

    return space_violation

def metric_pick_count():
    def pick_count(y_true, y_pred):
        y_pred = K.round(y_pred)
        return K.mean(K.sum(y_pred, -1) - K.sum(y_true, -1))

    return pick_count


# main part

def create_knapsack_dataset(count, item_count=5):
    x = [[], [], []]
    y = [[]]
    for _ in range(count):
        p = create_knapsack(item_count)
        x[0].append(p[0])
        x[1].append(p[1])
        x[2].append(p[2])
        y[0].append(p[3])
    return x, y

# train_x, train_y = create_knapsack_dataset(1)   # 10000 - examples for learning
# test_x, test_y = create_knapsack_dataset(1)   # 200 - for texting 

# def train_knapsack(model):
#     from keras.callbacks import ModelCheckpoint
#     import os
#     if os.path.exists("best_model.h5"): os.remove("best_model.h5")
#     model.fit(train_x, train_y, epochs=96, verbose=0, callbacks=[ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True, save_weights_only=True)])
#     model.load_weights("best_model.h5")
#     train_results = model.evaluate(train_x, train_y, 64, 0)
#     test_results = model.evaluate(test_x, test_y, 64, 0)
#     print("Model results(Train/Test):")
#     print(f"Loss:               {train_results[0]:.2f} / {test_results[0]:.2f}")
#     print(f"Binary accuracy:    {train_results[1]:.2f} / {test_results[1]:.2f}")
#     print(f"Space violation:    {train_results[2]:.2f} / {test_results[2]:.2f}")
#     print(f"Overpricing:        {train_results[3]:.2f} / {test_results[3]:.2f}")
#     print(f"Pick count:         {train_results[4]:.2f} / {test_results[4]:.2f}")

# # control of solution

# def supervised_continues_knapsack(item_count=5):
#     input_weights = Input((item_count,))
#     input_prices = Input((item_count,))
#     input_capacity = Input((1,))
#     inputs_concat = Concatenate()([input_weights, input_prices, input_capacity])
#     picks = Dense(item_count, use_bias=False, activation="sigmoid")(inputs_concat)
#     model = Model(inputs=[input_weights, input_prices, input_capacity], outputs=[picks])
#     model.compile("sgd",
#                   binary_crossentropy,
#                   metrics=[binary_accuracy, metric_space_violation(input_weights, input_capacity),
#                            metric_overprice(input_prices), metric_pick_count()])
#     return model
# model = supervised_continues_knapsack()
# train_knapsack(model)



# new vers (of train_knapsack)

# def train_knapsack(model):
#     from keras.callbacks import ModelCheckpoint
#     import os
#     if os.path.exists("best.weights.h5"): os.remove("best.weights.h5")
#     model.fit(train_x, train_y, epochs=96, verbose=0, callbacks=[ModelCheckpoint("best.weights.h5", monitor="loss", save_best_only=True, save_weights_only=True)])
#     model.load_weights("best.weights.h5")
#     train_results = model.evaluate(train_x, train_y, 64, 0)
#     test_results = model.evaluate(test_x, test_y, 64, 0)
#     print("Model results(Train/Test):")
#     print(f"Loss:               {train_results[0]:.2f} / {test_results[0]:.2f}")
#     print(f"Binary accuracy:    {train_results[1]:.2f} / {test_results[1]:.2f}")
#     print(f"Space violation:    {train_results[2]:.2f} / {test_results[2]:.2f}")
#     print(f"Overpricing:        {train_results[3]:.2f} / {test_results[3]:.2f}")
#     print(f"Pick count:         {train_results[4]:.2f} / {test_results[4]:.2f}")
