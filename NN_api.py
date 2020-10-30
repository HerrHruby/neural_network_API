"""
A simple Neural Network API. Provides a clean interface for building Deep Feedforward Neural Networks.
The network itself is represented by the Dense class, which provides a reference to the first (input) layer. Also
provides a selection of methods to implement forwards and backwards backpropagation, fit data and make predictions.
The layers in the network are represented by the Layer class, which are joined-up in a Linked List by the Dense object.

TODO: more loss functions (cross-entropy, absolute error etc.), more activations (softmax, tanh etc.), more optimisers
(adam, RMSProp etc.), "matrix style" propagation (rather than propagation datapoint by datapoint) for efficiency gains

Ian Wu
29/10/2020
"""

import numpy as np


class Layer:
    """Layer Class for deep feedforward networks.
        Attributes:
            input_dim: dimensions of the previous layer
            output_dim: dimensions of the next layer
            w: weight matrix associated with the layer
            w_gradient: weight gradients associated with the layer
            w_momentum: weight momentum associated with the layer
            b: bias vector associated with the layer
            b_gradient: bias gradients associated with the layer
            b_momentum: bias momentum associated with the layer
            activation: activation associated with the layer
            data: data (post activation) contained in the layer
            cache: data (pre-activiation) contained in the layer
            state: layer type (input, hidden or output)
            next_layer: points to the next layer
            previous_layer: points to the previous layer
    """

    def __init__(self, activation, input_dim, output_dim):
        """Initialise the Layer object
            activation: activation function
            input_dim: dimensions of the previous layer
            output_dim: dimensions of the next layer
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        if self.input_dim and self.output_dim:
            self.w = np.random.normal(0, (1/self.input_dim), size=(self.output_dim, self.input_dim))  # "Xavier" Init.
            self.w_gradient = np.zeros((self.output_dim, self.input_dim))
            self.w_momentum = np.zeros((self.output_dim, self.input_dim))
        if self.output_dim:
            self.b = np.zeros((self.output_dim,))
            self.b_gradient = np.zeros((self.output_dim,))
            self.b_momentum = np.zeros((self.output_dim,))
        self.activation = activation
        self.next_layer = None
        self.previous_layer = None
        self.data = None
        self.cache = None
        self.state = None

    def set_data(self, data):
        """Set the data"""
        self.data = data

    def get_data(self):
        """Get the data"""
        return self.data

    def get_w(self):
        """Get weight matrix"""
        return self.w

    def set_w(self, w):
        """Set weight matrix"""
        self.w = w

    def get_b(self):
        """Get bias vector"""
        return self.b

    def set_b(self, b):
        """Set bias vector"""
        self.b = b

    def get_w_gradient(self):
        """Get weight gradient"""
        return self.w_gradient

    def set_w_gradient(self, w_grad):
        """Set weight gradient"""
        self.w_gradient = w_grad

    def get_b_gradient(self):
        """Get bias gradient"""
        return self.b_gradient

    def set_b_gradient(self, b_grad):
        """Set bias gradient"""
        self.b_gradient = b_grad

    def get_w_momentum(self):
        """Get weight momentum"""
        return self.w_momentum

    def set_w_momentum(self, w_mom):
        """Set weight momentum"""
        self.w_momentum = w_mom

    def get_b_momentum(self):
        """Get bias momentum"""
        return self.b_momentum

    def set_b_momentum(self, b_mom):
        """Set bias momentum"""
        self.b_momentum = b_mom

    def get_cache(self):
        """Get the cache"""
        return self.cache

    def get_activation(self):
        """Get the activation"""
        return self.activation

    def get_next_layer(self):
        """Get the next layer"""
        return self.next_layer

    def set_next_layer(self, layer):
        """Set the next layer"""
        self.next_layer = layer

    def set_prev_layer(self, layer):
        """Set the previous layer"""
        self.previous_layer = layer

    def get_prev_layer(self):
        """Get the previous layer"""
        return self.previous_layer

    def set_state(self, state):
        """Set the state"""
        self.state = state

    def get_state(self):
        """Get the state"""
        return self.state

    def compute_data(self):
        """Transform the input into the layer (affine and then activation)"""
        data = self.get_prev_layer().get_data()  # get the data from the layer before
        data = np.matmul(self.w, data) + self.b  # affine transformation
        self.cache = data  # save the affine transform in self.cache
        self.data = self.activate(data)  # save the activated data in self.data

    def activate(self, z):
        """Activate data"""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'relu':
            return np.maximum(0, z)


class Dense:
    """Dense Class for neural network. Represents the network. Implements a doubly-linked list of layer
        Attributes:
            head: the first layer of the network
            tail: the last layer of the network
            loss: loss function for network
            metric: accuracy metric
            optimiser: optimiser for parameters
    """

    def __init__(self):
        """Initialise the Dense object and the input layer"""

        self.head = Layer(activation=None, input_dim=None, output_dim=None)  # set the input layer on net initialisation
        self.head.set_state('input')
        self.tail = self.head
        self.loss = None
        self.metric = None
        self.optimiser = None

    def add_layer(self, activation, input_dim, output_dim):
        """Add a layer to the neural network"""
        layer = Layer(activation, input_dim, output_dim)  # initialise the layer
        layer.set_state('output')
        layer.set_prev_layer(self.tail)  # connect the layer to the tail of the network
        if layer.get_prev_layer().get_state() == 'output':  # set the new states
            layer.get_prev_layer().set_state('hidden')
        self.tail.set_next_layer(layer)  # connect the tail of the network to the layer
        self.tail = layer  # set the new layer as the tail

    def forward_propagate(self, input_data):
        """Perform a single pass of forward propagation"""
        self.head.set_data(input_data)  # pass input data into the network
        current_layer = self.head.get_next_layer()
        # step through the network and compute the transformations on the data
        while current_layer:
            current_layer.compute_data()
            current_layer = current_layer.get_next_layer()

        return self.tail.get_data()  # return the final transformed data

    def backward_propagate(self, outputs, labels):
        """Perform a single pass of backward propagation"""
        current_layer = self.tail  # start at the tail (output layer L)
        activation = current_layer.get_activation()
        del_l = (outputs - labels) * self.activation_prime(current_layer.get_cache(), activation)
        # derivative of the loss wrt the affine transformed data of the output layer L
        d_cost_b = del_l  # derivative of cost wrt bias of output layer L
        d_cost_w = np.outer(del_l, current_layer.get_prev_layer().get_data())
        # derivative of cost wrt weights of output layer L
        current_layer.set_b_gradient(d_cost_b + current_layer.get_b_gradient())  # update the weight gradient
        current_layer.set_w_gradient(d_cost_w + current_layer.get_w_gradient())  # update the bias gradient
        current_layer = current_layer.get_prev_layer()
        # iterate through all layers l = L-1, L-2,...,3, 2
        while current_layer.get_state() != 'input':
            prev_layer = current_layer.get_prev_layer()
            activation = current_layer.get_activation()
            del_l = np.matmul(current_layer.get_next_layer().get_w().T, del_l) * \
                    self.activation_prime(current_layer.get_cache(), activation)
            # derivative of loss wrt affine transformation of lth layer
            d_cost_b = del_l  # derivative of loss wrt lth later bias
            d_cost_w = np.outer(del_l, prev_layer.get_data())  # derivative of loss wrt lth layer weights
            current_layer.set_b_gradient(d_cost_b + current_layer.get_b_gradient())  # update bias gradients
            current_layer.set_w_gradient(d_cost_w + current_layer.get_w_gradient())  # update weight gradients
            current_layer = prev_layer  # step into layer l-1

    def gradient_descent(self, learning_rate, batch_size):
        """Performs a step of gradient descent for all layers, for weights and biases
            Parameters:
                learning_rate: the learning rate for gradient descent
                batch_size: the batch size of data
        """
        current_layer = self.head  # start at input layer
        # step through all layers
        while current_layer.get_next_layer():
            current_layer = current_layer.get_next_layer()
            new_w = current_layer.get_w() - (learning_rate * (1/batch_size) * current_layer.get_w_gradient())
            current_layer.set_w(new_w)  # update w as w := w - (learning_rate/batch_size) * grad_w
            new_b = current_layer.get_b() - (learning_rate * (1/batch_size) * current_layer.get_b_gradient())
            current_layer.set_b(new_b)  # update b as b := b - (learning_rate/batch_size) * grad_b
            # zero all the gradients
            current_layer.set_w_gradient(np.zeros((current_layer.output_dim, current_layer.input_dim)))
            current_layer.set_b_gradient(np.zeros((current_layer.output_dim, )))

    def momentum(self, learning_rate, batch_size, decay_rate):
        """Performs a step of momentum gradient descent for all layers, for weights and biases
            Parameters:
                learning_rate: the learning rate for gradient descent
                batch_size: the batch size of data
                decay_rate: the decay rate for momentum
        """
        current_layer = self.head  # start at input layer
        # step through all layers
        while current_layer.get_next_layer():
            current_layer = current_layer.get_next_layer()
            update_w = learning_rate * (1 / batch_size) * current_layer.get_w_gradient()
            update_b = learning_rate * (1 / batch_size) * current_layer.get_b_gradient()
            current_layer.set_w_momentum(decay_rate * current_layer.get_w_momentum() +
                                         update_w)  # update weight momentum
            current_layer.set_b_momentum(decay_rate * current_layer.get_b_momentum() +
                                         update_b)  # update bias momentum
            new_w = current_layer.get_w() - current_layer.get_w_momentum()
            current_layer.set_w(new_w)  # update w := w - (learning_rate/batch_size) * grad_w + (decay_rate * momentum)
            new_b = current_layer.get_b() - current_layer.get_b_momentum()
            current_layer.set_b(new_b)  # update b := b - (learning_rate/batch_size) * grad_b + (decay_rate * momentum)
            current_layer.set_w_gradient(np.zeros((current_layer.output_dim, current_layer.input_dim)))  # zero grad_w
            current_layer.set_b_gradient(np.zeros((current_layer.output_dim,)))  # zero grad_b

    def compile(self, optimiser, loss, metric, learning_rate=0.005, decay_rate=0.9):
        """Compile the neural network. Set the optimiser, loss, metric, learning and decay rates
            Parameters:
                optimiser: optimiser for gradient descent ('gradient_descent'/'momentum')
                loss: loss function ('MSE')
                metric: (optional) accuracy metric ('binary_accuracy)
                learning_rate: learning rate for gradient descent (default=0.005)
                decay_rate: decay rate for momentum gradient descent (default=0.9
        """
        optimiser = (optimiser, learning_rate, decay_rate)  # group the optimiser and its parameters
        self.optimiser = optimiser
        self.loss = loss
        self.metric = metric

    def fit(self, input_data, labels, epochs, shuffle_input=True, shuffle_validate=True, batch_size=None,
            validate=False, validation_data=None, validation_labels=None):
        """Train the neural network on the training data. Optionally, perform validation after every epoch
            Parameters:
                input_data: training data
                labels: training data labels
                epochs: the number of epochs (complete passes through the data)
                shuffle_input: shuffle the training data every epoch (default=True)
                shuffle_validate: shuffle the validation data every epoch (default=True)
                batch_size: batch size for training
                validate: perform validation at the end of each epoch (True/False, default=False)
                validation_data: validation data
                validation_labels: validation data labels
        """
        epoch = 0
        optimiser_type = self.optimiser[0]
        learning_rate = self.optimiser[1]
        results = [None, None]  # contains the final losses and metrics
        while epoch < epochs:
            print('--------------------------------------------')
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            epoch_loss = 0
            epoch_metric = 0
            # shuffle the dataset and labels together if shuffle_input=True
            if shuffle_input:
                indices = np.arange(input_data.shape[0])
                np.random.shuffle(indices)
                input_data = input_data[indices]
                labels = labels[indices]
            cycle = 1
            last_optimise_step = 0  # track the cycle of the last optimisation step
            # if batch size not specified, assume full-batch
            if not batch_size:
                batch_size = len(input_data)
            # iterate through all the data points and labels
            for datapoint, label in zip(input_data, labels):
                # do forward prop and update metric and losses
                predictions = self.forward_propagate(datapoint)
                if self.metric == 'binary_accuracy':
                    if round(predictions[0]) == label[0]:
                        epoch_metric += 1
                epoch_loss += self.compute_loss(predictions, label, self.loss)
                self.backward_propagate(predictions, label)  # back prop
                # perform optimisation if mini-batch is complete. If last cycle reached, optimise as well
                if cycle % batch_size == 0 or cycle == len(input_data):
                    if optimiser_type == 'gradient_descent':
                        # standard gradient descent. Batch size = no. of data points since last optimisation step
                        self.gradient_descent(learning_rate, cycle - last_optimise_step)
                    elif optimiser_type == 'momentum':
                        decay_rate = self.optimiser[2]
                        # gradient descent w. momentum. Batch size = no. of data points since last optimisation step
                        self.momentum(learning_rate, cycle - last_optimise_step, decay_rate=decay_rate)
                    last_optimise_step = cycle
                cycle += 1
            epoch_loss = epoch_loss/len(input_data)  # compute average loss for this epoch
            print('Epoch {} Complete'.format(epoch + 1))
            print('Epoch Training Loss: {}'.format(epoch_loss))
            epoch_metric = float(epoch_metric/len(input_data))  # compute average metric for this epoch
            if self.metric == 'binary_accuracy':
                print('Epoch Training Accuracy: {}'.format(epoch_metric))
            results[0] = (epoch_loss, epoch_metric)  # store the final results
            epoch += 1

            if validate:  # do validation step after every epoch if validate=True
                validation_loss = 0
                validation_metric = 0
                # shuffle the validation data and labels if shuffle_validate=True
                if shuffle_validate:
                    indices = np.arange(validation_data.shape[0])
                    np.random.shuffle(indices)
                    validation_data = validation_data[indices]
                    validation_labels = validation_labels[indices]
                # iterate through validation data and labels
                for datapoint, label in zip(validation_data, validation_labels):
                    predictions = self.predict(datapoint)  # make prediction
                    validation_loss += self.compute_loss(predictions, label, self.loss)  # get loss
                    if self.metric == 'binary_accuracy':  # get metric
                        if round(predictions[0]) == label[0]:
                            validation_metric += 1
                validation_loss = validation_loss/len(validation_data)  # average validation loss for epoch
                validation_metric = float(validation_metric/len(validation_data))  # average validation metric for epoch
                print('Epoch Validation Loss: {}'.format(validation_loss))
                if self.metric == 'binary_accuracy':
                    print('Epoch Validation Accuracy: {}'.format(validation_metric))
                results[1] = (validation_loss, validation_metric)  # store results
            print('--------------------------------------------')
        print('Final Training Loss/Accuracy: {}'.format(results[0]))
        print('Final Validation Loss/Accuracy: {}'.format(results[1]))

        return results

    def predict(self, input_data):
        """Predict on input data using the neural network"""
        prediction = self.forward_propagate(input_data)  # propagate data forward to make prediction
        return prediction

    @staticmethod
    def compute_loss(outputs, labels, loss):
        """Compute the loss using a specified loss function"""
        if loss == 'MSE':
            return 0.5 * np.sum((outputs - labels) ** 2)

    @staticmethod
    def activation_prime(z, activation):
        """Evaluate the derivative of the activation"""
        if activation == 'sigmoid':
            sigma = 1 / (1 + np.exp(-z))
            return sigma * (1 - sigma)
        elif activation == 'relu':
            q = z.copy()
            q[z > 0] = 1
            q[z <= 0] = 0
            return q


class GridSearchTuner:
    """Hyperparameter tuning class, using grid-search. Tunes the learning rate, batch-size and decay-rate
        Attributes:
            learning_rates: list of learning rates to try
            batch_sizes: list of batch-sizes to try
            decay_rates: list of decay_rates to try
            model: model to tune (pre-compilation)
            best_params: list of the best parameters
    """

    def __init__(self):
        """Initialise the GridSearchTuner object"""
        self.learning_rates = [0.01]
        self.batch_sizes = [1]
        self.decay_rates = [0]
        self.model = None
        self.best_params = None

    def learning_rate_grid(self, grid):
        """Set the list of learning rates to try"""
        self.learning_rates = grid

    def batch_size_grid(self, grid):
        """Set the list of batch sizes to try"""
        self.batch_sizes = grid

    def decay_rates_grid(self, grid):
        """Set the list of decay rates to try"""
        self.decay_rates = grid

    def set_model(self, model):
        """Set the model to tune (pre-compiled model)"""
        self.model = model

    def get_best_params(self):
        """Get the best parameters found from tuning"""
        return self.best_params

    def search(self, training_data, labels, validation_data, validation_labels, epochs, optimiser, loss, metric):
        """Perform hyperparameter tuning on the selected model. Sets the best_params attribute"""
        optimal_loss = float('inf')
        best_params = [0, 0, 0]
        # search through the grid by training using all combinations of params on grid
        for learning_rate in self.learning_rates:
            for batch_size in self.batch_sizes:
                for decay_rate in self.decay_rates:
                    print('Testing Hyperparameters: Learning Rate = {}, Batch Size = {}, Momentum = {}'
                          .format(learning_rate, batch_size, decay_rate))
                    self.model.compile(optimiser=optimiser, loss=loss, metric=metric,
                                       learning_rate=learning_rate, decay_rate=decay_rate)
                    results = self.model.fit(input_data=training_data, labels=labels, epochs=epochs,
                                             batch_size=batch_size, validation_data=validation_data,
                                             validation_labels=validation_labels, validate=True)
                    if results[1][0] < optimal_loss:  # store the params if they yield the best results so far
                        optimal_loss = results[1][0]
                        best_params[0] = learning_rate
                        best_params[1] = batch_size
                        best_params[2] = decay_rate

        self.best_params = best_params  # save the best params as an attribute of GridSearchTuner



