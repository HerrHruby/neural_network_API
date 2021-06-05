# neural_network_API
A simple API for feedforward deep neural networks, written in Python

The NN classes are held in NN_api.py. An example
neural network is constructed in main.py (used on 
a binary classification MNIST fashion dataset)

Currently supports the following:
- Loss functions: cross entropy, binary cross entropy, MSE
- Activations: sigmoid, ReLU, Linear, softmax
- Optimisers: vanilla gradient descent, momentum
- Hyperparameter Tuning: Automatic Tuner
    
## Example

To train a simple Neural Network:

```python
    import NN_api

    model = NN_api.Dense()  # initialise network
    model.add_layer('relu', 784, 256)  # add layers
    model.add_layer('sigmoid', 256, 1)
    model.set_name('my_network')
    # set optimiser and fit to the training data
    model.compile(optimiser='momentum', loss='MSE', metric='binary_accuracy',
                  learning_rate=0.01, decay_rate=0.95)
    model.fit(input_data=training_data, labels=training_labels, epochs=100, batch_size=120, validate=True,
              validation_data=validation_data, validation_labels=validation_labels, save_data=True)
```

To use the automatic tuner:

```python
    import NN_api
    
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
```


          
