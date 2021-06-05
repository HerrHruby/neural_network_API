import numpy as np
import NN_api
import matplotlib.pyplot as plt


def main():

    # load in data and apply appropriate transformations
    training_data = np.load('data/fashion-train-imgs.npz')
    training_labels = np.load('data/fashion-train-labels.npz')
    validation_data = np.load('data/fashion-dev-imgs.npz')
    validation_labels = np.load('data/fashion-dev-labels.npz')

    # reshape input data as required
    training_data = np.reshape(training_data, (784, 12000)).T
    training_labels = np.reshape(training_labels, (-1, 1))
    validation_data = np.reshape(validation_data, (784, 1000)).T
    validation_labels = np.reshape(validation_labels, (-1, 1))

    # train a simple neural network
    model = NN_api.Dense()  # initialise network
    model.add_layer('relu', 784, 256)  # add layers
    model.add_layer('sigmoid', 256, 1)
    model.set_name('my_network')
    # set optimiser and fit to the training data
    model.compile(optimiser='momentum', loss='MSE', metric='binary_accuracy',
                  learning_rate=0.01, decay_rate=0.95)
    model.fit(input_data=training_data, labels=training_labels, epochs=100, batch_size=120, validate=True,
              validation_data=validation_data, validation_labels=validation_labels, save_data=True)

    # automatic hyperparameter tuning
    tuner = NN_api.GridSearchTuner()  # create tuner object
    tuner.add_layer('relu', 784, 256)  # add layers
    tuner.add_layer('sigmoid', 256, 1)
    tuner.compile(optimiser='momentum', loss='MSE', metric='binary_accuracy')  # set optimiser
    tuner.learning_rate_grid([0.00005, 0.0001, 0.0005, 0.001])  # provide grid for learning rate
    tuner.batch_size_grid([1, 120, 1200, 12000])  # provide grid for batch size
    tuner.decay_rates_grid([0.8, 0.85, 0.9, 0.95])  # provide grid for momentum decay rates
    # perform grid search!
    tuner.search(training_data=training_data, labels=training_labels, validation_data=validation_data,
                 validation_labels=validation_labels, epochs=20)


if __name__ == '__main__':
    main()
