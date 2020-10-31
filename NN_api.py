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
import pickle


class Layer:
    """Layer Class for deep feedforward networks.
        Attributes:
            input_dim: dimensions of the previous layer
            output_dim: dimensions of the next layer
            w: weight matrix associated with the layer
            w_momentum: weight momentum associated with the layer
            b: bias vector associated with the layer
            b_momentum: bias momentum associated with the layer
            activation: activation associated with the layer
            data: data (post activation) contained in the layer
            cache: data (pre-activation) contained in the layer
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
            self.w_momentum = np.zeros((self.output_dim, self.input_dim))
        if self.output_dim:
            self.b = np.zeros((self.output_dim, ))
            self.b_momentum = np.zeros((self.output_dim, ))
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
        data = np.matmul(self.w, data) + self.b.reshape(-1, 1)  # affine transformation
        self.cache = data  # save the affine transform in self.cache
        self.data = self.activate(data)  # save the activated data in self.data

    def activate(self, z):
        """Activate data"""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'relu':
            return np.maximum(0, z)

    def gradient_descent(self, learning_rate, batch_size, d_cost_b, d_cost_w):
        """Performs a step of gradient descent for all layers, for weights and biases
            Parameters:
                learning_rate: the learning rate for gradient descent
                batch_size: the batch size of data
                d_cost_b: grad b
                d_cost_w: grad w
        """
        new_w = self.get_w() - (learning_rate * (1/batch_size) * d_cost_w)
        self.set_w(new_w)  # update w as w := w - (learning_rate/batch_size) * grad_w
        new_b = self.get_b() - (learning_rate * (1/batch_size) * d_cost_b)
        self.set_b(new_b)  # update b as b := b - (learning_rate/batch_size) * grad_b

    def momentum(self, learning_rate, batch_size, decay_rate, d_cost_b, d_cost_w):
        """Performs a step of momentum gradient descent for all layers, for weights and biases
            Parameters:
                learning_rate: the learning rate for gradient descent
                batch_size: the batch size of data
                decay_rate: the decay rate for momentum
                d_cost_b: grad b
                d_cost_w: grad w

        """
        grad_desc_w = learning_rate * (1/batch_size) * d_cost_w
        grad_desc_b = learning_rate * (1/batch_size) * d_cost_b
        delta_w = decay_rate * self.get_w_momentum() - grad_desc_w
        delta_b = decay_rate * self.get_b_momentum() - grad_desc_b
        self.set_w(self.get_w() + delta_w)
        self.set_b(self.get_b() + delta_b)
        self.set_w_momentum(delta_w)
        self.set_b_momentum(delta_b)


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
        self.name = None

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

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

    def backward_propagate(self, outputs, labels, batch_size):
        """Perform a single pass of backward propagation"""
        current_layer = self.tail  # start at the tail (output layer L)
        activation = current_layer.get_activation()
        del_l = None
        if self.loss == 'MSE':
            del_l = (outputs - labels) * self.activation_prime(current_layer.get_cache(), activation)
        # derivative of the loss wrt the affine transformed data of the output layer L
        d_cost_b = np.sum(del_l, axis=1)  # derivative of cost wrt bias of output layer L
        d_cost_w = np.matmul(del_l, current_layer.get_prev_layer().get_data().T)
        # derivative of cost wrt weights of output layer L
        if self.optimiser[0] == 'gradient_descent':
            current_layer.gradient_descent(learning_rate=self.optimiser[1], batch_size=batch_size, d_cost_w=d_cost_w,
                                           d_cost_b=d_cost_b)
        elif self.optimiser[0] == 'momentum':
            current_layer.momentum(learning_rate=self.optimiser[1], batch_size=batch_size, d_cost_w=d_cost_w,
                                   d_cost_b=d_cost_b, decay_rate=self.optimiser[2])
        current_layer = current_layer.get_prev_layer()
        while current_layer.get_state() != 'input':
            prev_layer = current_layer.get_prev_layer()
            activation = current_layer.get_activation()
            if self.loss == 'MSE':
                del_l = np.matmul(current_layer.get_next_layer().get_w().T, del_l) * \
                        self.activation_prime(current_layer.get_cache(), activation)
            # derivative of loss wrt affine transformation of lth layer
            d_cost_b = np.sum(del_l, axis=1)  # derivative of loss wrt lth layer bias
            d_cost_w = np.matmul(del_l, prev_layer.get_data().T)  # derivative of loss wrt lth layer weights
            if self.optimiser[0] == 'gradient_descent':
                current_layer.gradient_descent(learning_rate=self.optimiser[1], batch_size=batch_size, d_cost_w=d_cost_w,
                                               d_cost_b=d_cost_b)
            elif self.optimiser[0] == 'momentum':
                current_layer.momentum(learning_rate=self.optimiser[1], batch_size=batch_size, d_cost_w=d_cost_w,
                                       d_cost_b=d_cost_b, decay_rate=self.optimiser[2])
            current_layer = prev_layer

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
        print(self.optimiser)

    def fit(self, input_data, labels, epochs, shuffle_input=True, shuffle_validate=True, batch_size=None,
            validate=False, validation_data=None, validation_labels=None, save_data=False):
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
                save_data: save the training and validation losses and metrics
        """
        epoch = 0
        final_results = [None, None]  # contains the final losses and metrics
        save_train_results = []
        save_val_results = []
        while epoch < epochs:
            print('--------------------------------------------')
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            epoch_loss = 0
            epoch_metric = 0
            # if batch size not specified, assume full-batch
            if not batch_size:
                batch_size = len(input_data)
            # iterate through all the data points and labels
            for batch in self.generate_batch(input_data, labels, batch_size, shuffle_input=shuffle_input):
                # do forward prop and update metric and losses
                data_point = batch[0]
                label = batch[1]
                predictions = self.forward_propagate(data_point)
                if self.metric == 'binary_accuracy':
                    epoch_metric += np.sum(np.round(predictions[0]) == label[0])
                epoch_loss += self.compute_loss(predictions, label, self.loss)
                self.backward_propagate(predictions, label, batch_size=len(data_point[0]))  # back prop
            epoch_loss = epoch_loss/len(input_data)  # compute average loss for this epoch
            print('Epoch {} Complete'.format(epoch + 1))
            print('Epoch Training Loss: {}'.format(epoch_loss))
            epoch_metric = float(epoch_metric/len(input_data))  # compute average metric for this epoch
            if self.metric == 'binary_accuracy':
                print('Epoch Training Accuracy: {}'.format(epoch_metric))
            final_results[0] = (epoch_loss, epoch_metric)  # store the final results
            if save_data:
                save_train_results.append((epoch_loss, epoch_metric))

            epoch += 1

            if validate:  # do validation step after every epoch if validate=True
                validation_loss = 0
                validation_metric = 0
                # iterate through validation data and labels
                for batch in self.generate_batch(validation_data, validation_labels, batch_size=len(validation_data),
                                                 shuffle_input=shuffle_validate):
                    data_point = batch[0]
                    label = batch[1]
                    predictions = self.predict(data_point)  # make prediction
                    validation_loss += self.compute_loss(predictions, label, self.loss)  # get loss
                    if self.metric == 'binary_accuracy':  # get metric
                        validation_metric += np.sum(np.round(predictions[0]) == label[0])
                validation_loss = validation_loss/len(validation_data)  # average validation loss for epoch
                validation_metric = validation_metric/len(validation_data)  # average validation metric for epoch
                print('Epoch Validation Loss: {}'.format(validation_loss))
                if self.metric == 'binary_accuracy':
                    print('Epoch Validation Accuracy: {}'.format(validation_metric))
                final_results[1] = (validation_loss, validation_metric)  # store results
                if save_data:
                    save_val_results.append((validation_loss, validation_metric))
            print('--------------------------------------------')
        print('Final Training Loss/Accuracy: {}'.format(final_results[0]))
        print('Final Validation Loss/Accuracy: {}'.format(final_results[1]))

        if save_data:
            training_name = self.get_name() + '_training'
            with open(training_name, 'wb') as fp:
                pickle.dump(save_train_results, fp)
            if validate:
                val_name = self.get_name() + '_validation'
                with open(val_name, 'wb') as fp:
                    pickle.dump(save_val_results, fp)

        return final_results

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
    def generate_batch(input_data, labels, batch_size, shuffle_input):
        """Generate mini-batches of training data and labels"""
        indices = None
        # shuffle the data and labels simultaneously if shuffle_input=True
        if shuffle_input:
            indices = np.arange(input_data.shape[0])
            np.random.shuffle(indices)
        # generator for the mini-batches
        for index in range(0, input_data.shape[0] - batch_size + 1, batch_size):
            end_index = min(index + batch_size, input_data.shape[0])
            if shuffle_input:
                data_slice = indices[index:end_index]
            else:
                data_slice = slice(index, end_index)
            yield input_data[data_slice].T, labels[data_slice].T

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



