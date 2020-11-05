import numpy as np
import NN_api
import matplotlib.pyplot as plt


def main():

    # load in data and apply appropriate transformations
    training_data = np.load('data/fashion-train-imgs.npz')
    training_labels = np.load('data/fashion-train-labels.npz')
    validation_data = np.load('data/fashion-dev-imgs.npz')
    validation_labels = np.load('data/fashion-dev-labels.npz')

    training_data = np.reshape(training_data, (784, 12000)).T
    training_labels = np.reshape(training_labels, (-1, 1))
    validation_data = np.reshape(validation_data, (784, 1000)).T
    validation_labels = np.reshape(validation_labels, (-1, 1))

    # plt.imshow(training_data[8].reshape((28, 28)), cmap='gray')
    # plt.show()

    model = NN_api.Dense()  # initialise network
    model.add_layer('relu', 784, 256)
    model.add_layer('sigmoid', 256, 1)
    # compile and then fit to training data
    model.set_name('optimal_mom')
    model.compile(optimiser='momentum', loss='MSE', metric='binary_accuracy',
                  learning_rate=0.01, decay_rate=0.95)
    results = model.fit(input_data=training_data, labels=training_labels, epochs=100, batch_size=120, validate=True,
                        validation_data=validation_data, validation_labels=validation_labels, save_data=True)

    # code for hyperparameter tuning
    tune_model = NN_api.Dense()  # initialise network and add layers
    tune_model.add_layer('relu', 784, 256)
    tune_model.add_layer('sigmoid', 256, 1)

    tuner = NN_api.GridSearchTuner()  # create tuner object
    tuner.set_model(tune_model)  # set model for tuner object
    tuner.learning_rate_grid([0.00005, 0.0001, 0.0005, 0.001])  # provide grid for learning rate
    tuner.batch_size_grid([1, 120, 1200, 12000])  # provide grid for batch size
    tuner.decay_rates_grid([0.8, 0.85, 0.9, 0.95])
    # perform grid search!
    tuner.search(training_data=training_data, labels=training_labels, validation_data=validation_data,
                 validation_labels=validation_labels, epochs=20, optimiser='momentum', loss='MSE',
                 metric='binary_accuracy')
    print(tuner.get_best_params())


if __name__ == '__main__':
    main()
