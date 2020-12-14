"""
A simple Neural Network API. Provides a clean interface for building Deep Feedforward Neural Networks.
The network itself is represented by the Dense class, which provides a reference to the first (input) layer. Also
provides a selection of methods to implement forwards and backwards backpropagation, fit data and make predictions.
The layers in the network are represented by the Layer class, which are joined up in a Linked List by the Dense object.

Currently Supported:
    Loss functions: MSE, binary cross entropy, cross entropy
    Activations: ReLU, Sigmoid, Softmax, Linear
    Optimisers: standard gradient descent, momentum
    Tuning: Grid Search

TODO: more loss functions (absolute error etc.), more activations (tanh etc.), more optimisers
(adam, RMSProp etc.), dropout, improved tuner (to tune layer sizes etc.)

Ian Wu
14/12/2020
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

    def compute_data(self):
        """Transform the input into the layer (affine and then activation)"""
        data = self.previous_layer.data  # get the data from the layer before
        data = np.matmul(self.w, data) + self.b.reshape(-1, 1)  # affine transformation
        self.cache = data  # save the affine transform in self.cache
        self.data = self.activate(data)  # save the activated data in self.data

    def activate(self, z):
        """Activate data"""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'softmax':
            return np.exp(z) / np.sum(np.exp(z), axis=0)
        elif self.activation == 'linear':
            return z

    def gradient_descent(self, learning_rate, batch_size, d_cost_b, d_cost_w):
        """Performs a step of gradient descent for the layer, for weights and biases
            Parameters:
                learning_rate: the learning rate for gradient descent
                batch_size: the batch size of data
                d_cost_b: grad b
                d_cost_w: grad w
        """
        new_w = self.w - (learning_rate * (1/batch_size) * d_cost_w)
        self.w = new_w  # update w as w := w - (learning_rate/batch_size) * grad_w
        new_b = self.b - (learning_rate * (1/batch_size) * d_cost_b)
        self.b = new_b  # update b as b := b - (learning_rate/batch_size) * grad_b

    def momentum(self, learning_rate, batch_size, decay_rate, d_cost_b, d_cost_w):
        """Performs a step of momentum gradient descent for the layer, for weights and biases
            Parameters:
                learning_rate: the learning rate for gradient descent
                batch_size: the batch size of data
                decay_rate: the decay rate for momentum
                d_cost_b: grad b
                d_cost_w: grad w
        """
        grad_desc_w = learning_rate * (1/batch_size) * d_cost_w  # standard descent for w
        grad_desc_b = learning_rate * (1/batch_size) * d_cost_b  # standard descent for b
        delta_w = decay_rate * self.w_momentum - grad_desc_w  # compute new w momentum
        delta_b = decay_rate * self.b_momentum - grad_desc_b  # compute new b momentum
        self.w = self.w + delta_w  # update weights
        self.b = self.b + delta_b  # update biases
        self.w_momentum = delta_w  # update w momentum
        self.b_momentum = delta_b  # update b momentum


class Dense:
    """Dense Class for neural network. Represents the network. Implements a doubly-linked list of layer
        Attributes:
            head: the first layer of the network
            tail: the last layer of the network
            loss: loss function for network
            metric: accuracy metric
            optimiser: optimiser for parameters
            name: the name of the model (for saving)
    """

    def __init__(self):
        """Initialise the Dense object and the input layer"""

        self.head = Layer(activation=None, input_dim=None, output_dim=None)  # set the input layer on net initialisation
        self.head.state = 'input'
        self.tail = self.head
        self.loss = None
        self.metric = None
        self.optimiser = None
        self.name = None

    def set_name(self, name):
        """Set the name of the network"""
        self.name = name

    def add_layer(self, activation, input_dim, output_dim):
        """Add a layer to the neural network"""
        layer = Layer(activation, input_dim, output_dim)  # initialise the layer
        layer.state = 'output'
        layer.previous_layer = self.tail  # connect the layer to the tail of the network
        if layer.previous_layer.state == 'output':  # set the new states
            layer.previous_layer.state = 'hidden'
        self.tail.next_layer = layer  # connect the tail of the network to the layer
        self.tail = layer  # set the new layer as the tail

    def forward_propagate(self, input_data):
        """Perform a single pass of forward propagation"""
        self.head.data = input_data  # pass input data into the network
        current_layer = self.head.next_layer
        # step through the network and compute the transformations on the data
        while current_layer:
            current_layer.compute_data()
            current_layer = current_layer.next_layer

        return self.tail.data  # return the final transformed data

    def compute_out_grad(self, activation, current_layer, outputs, labels):
        """Compute the gradient of the loss wrt. the final affine transformed data, dJ/dz_L"""
        if self.loss == 'MSE' and activation == 'sigmoid':
            del_l = (outputs - labels) * self.activation_prime(current_layer.cache, activation)
        elif activation == 'linear':
            del_l = outputs - labels
        elif self.loss == 'binary_cross_entropy' and activation == 'sigmoid':
            outputs[outputs == 0] += 1e-15  # prevent instability of log(0)
            outputs[outputs == 1] -= 1e-15
            del_l = outputs - labels
        elif self.loss == 'cross_entropy' and activation == 'softmax':
            outputs[outputs == 0] += 1e-15  # prevent instability of log(0)
            outputs[outputs == 1] -= 1e-15
            del_l = outputs - labels
        else:
            # only the above combination of output layers and losses are supported
            raise ValueError('Loss/output activation combination unsupported')

        return del_l

    def backward_propagate(self, outputs, labels, batch_size):
        """Perform a single pass of backward propagation"""
        current_layer = self.tail  # start at the tail (output layer L)
        activation = current_layer.activation
        del_l = self.compute_out_grad(activation, current_layer, outputs, labels)  # compute dJ/dz_L
        d_cost_b = np.sum(del_l, axis=1)  # derivative of cost wrt bias of output layer L
        d_cost_w = np.matmul(del_l, current_layer.previous_layer.data.T)  # derivative of cost wrt
        if self.optimiser[0] == 'gradient_descent':
            current_layer.gradient_descent(learning_rate=self.optimiser[1], batch_size=batch_size, d_cost_w=d_cost_w,
                                           d_cost_b=d_cost_b)
        elif self.optimiser[0] == 'momentum':
            current_layer.momentum(learning_rate=self.optimiser[1], batch_size=batch_size, d_cost_w=d_cost_w,
                                   d_cost_b=d_cost_b, decay_rate=self.optimiser[2])
        current_layer = current_layer.previous_layer
        while current_layer.state != 'input':
            prev_layer = current_layer.previous_layer
            activation = current_layer.activation
            del_l = np.matmul(current_layer.next_layer.w.T, del_l) * \
                    self.activation_prime(current_layer.cache, activation)
            # derivative of loss wrt affine transformation of lth layer
            d_cost_b = np.sum(del_l, axis=1)  # derivative of loss wrt lth layer bias
            d_cost_w = np.matmul(del_l, prev_layer.data.T)  # derivative of loss wrt lth layer weights
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
        save_train_results = []  # for recording training loss/metric per epoch
        save_val_results = []  # for recording val loss/metric per epoch
        while epoch < epochs:
            print('--------------------------------------------')
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            epoch_loss = 0
            epoch_metric = 0
            # if batch size not specified, assume full-batch
            if not batch_size:
                batch_size = len(input_data)
            # get mini-batches from generator
            for batch in self.generate_batch(input_data, labels, batch_size, shuffle_input=shuffle_input):
                # do forward prop and update metric and losses
                data_point = batch[0]
                label = batch[1]
                predictions = self.forward_propagate(data_point)  # forward prop with batch training data
                epoch_metric += self.count_correct(predictions, label)
                epoch_loss += self.compute_loss(predictions, label, self.loss)  # get the loss
                self.backward_propagate(predictions, label, batch_size=len(data_point[0]))  # back prop + optimise
            epoch_loss = epoch_loss/len(input_data)  # compute average loss for this epoch
            print('Epoch {} Complete'.format(epoch + 1))
            print('Epoch Training Loss: {}'.format(epoch_loss))
            epoch_metric = float(100*epoch_metric/len(input_data))  # compute average metric for this epoch
            if self.metric:
                print('Epoch Training Accuracy: {}'.format(epoch_metric))
            final_results[0] = (epoch_loss, epoch_metric)  # store the final results
            if save_data:
                save_train_results.append((epoch_loss, epoch_metric))  # record training loss/metric per epoch
            epoch += 1

            if validate:  # do validation step after every epoch if validate=True
                validation_loss = 0
                validation_metric = 0
                # get full-batch from generator
                for batch in self.generate_batch(validation_data, validation_labels, batch_size=len(validation_data),
                                                 shuffle_input=shuffle_validate):
                    data_point = batch[0]
                    label = batch[1]
                    predictions = self.predict(data_point)  # make prediction
                    validation_loss += self.compute_loss(predictions, label, self.loss)  # get loss
                    validation_metric += self.count_correct(predictions, label)
                validation_loss = validation_loss/len(validation_data)  # average validation loss for epoch
                validation_metric = 100*validation_metric/len(validation_data)  # average validation metric for epoch
                print('Epoch Validation Loss: {}'.format(validation_loss))
                if self.metric:
                    print('Epoch Validation Accuracy: {}'.format(validation_metric))
                final_results[1] = (validation_loss, validation_metric)  # store results
                if save_data:
                    save_val_results.append((validation_loss, validation_metric))  # record val loss/metric per epoch
            print('--------------------------------------------')
        print('Final Training Loss/Accuracy: {}'.format(final_results[0]))
        print('Final Validation Loss/Accuracy: {}'.format(final_results[1]))

        # if save_data=True, dump all saved data to binary file
        if save_data:
            training_name = self.name + '_training'
            with open(training_name, 'wb') as fp:
                pickle.dump(save_train_results, fp)
            if validate:
                val_name = self.name + '_validation'
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
        elif loss == 'binary_cross_entropy':
            outputs[outputs == 0] += 1e-15  # prevent instability of log(0)
            outputs[outputs == 1] -= 1e-15
            return -np.sum(labels * np.log(outputs) + (1 - labels) * np.log(1 - outputs))
        elif loss == 'cross_entropy':
            outputs[outputs == 0] += 1e-15  # prevent instability of log(0)
            outputs[outputs == 1] -= 1e-15
            return -np.sum(labels * np.log(outputs))

    def count_correct(self, predictions, labels):
        """Compute the number of correct predictions in the batch"""
        correct = None
        if self.metric == 'binary_accuracy':
            correct = np.sum(np.round(predictions) == labels)
        elif self.metric == 'accuracy':
            prediction_max = np.argmax(predictions, axis=0)
            labels_max = np.argmax(labels, axis=0)
            correct = np.sum(prediction_max == labels_max)

        return correct

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
        elif activation == 'linear':
            return np.ones((z.shape[0], z.shape[1]))


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
        self.optimiser = None
        self.loss = None
        self.metric = None

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

    def compile(self, optimiser, loss, metric):
        """Compile the tuner. Set the optimiser, loss, metric, learning and decay rates
            Parameters:
                optimiser: optimiser for gradient descent ('gradient_descent'/'momentum')
                loss: loss function ('MSE')
                metric: (optional) accuracy metric ('binary_accuracy)
        """
        self.optimiser = optimiser
        self.loss = loss
        self.metric = metric

    def _cross_validate(self, cv, X, y, learning_rate, decay_rate, batch_size, epochs):
        """Perform K-fold cross-validation. Returns the average loss"""
        # shuffle the data and labels
        indices = [i for i in range(0, len(X))]
        shuffle_indices = np.arange(len(X))
        np.random.shuffle(shuffle_indices)
        shuffled_X = X[shuffle_indices]
        shuffled_y = y[shuffle_indices]
        n = len(X) // cv  # min. no. of samples in each fold
        remainder = len(X) % cv  # the remainder in the last fold
        val_loss_list = []
        for partition in range(0, cv):
            print('Cross-validating: fold {}...'.format(partition + 1))
            if partition != cv - 1:  # for the first K - 1 folds...
                val_indices = indices[partition * n: (partition + 1) * n]  # extract partition for val
                train_indices = indices[: partition * n] + indices[(partition + 1) * n:]  # extract partition for train
            else:
                # for the final fold, we include the remainder
                val_indices = indices[partition * n: (partition + 1) * n + remainder]  # extract partition for val
                train_indices = indices[: partition * n] + indices[(partition + 1) * n + remainder:]  # extract partition for train
            X_val = shuffled_X[val_indices]  # validation features for current fold
            y_val = shuffled_y[val_indices]  # validation labels for current fold
            X_train = shuffled_X[train_indices]  # training features for current fold
            y_train = shuffled_y[train_indices]  # training labels for current fold
            # build, train and validate neural network
            self.model.compile(optimiser=self.optimiser, loss=self.loss, metric=self.metric,
                               learning_rate=learning_rate, decay_rate=decay_rate)
            results = self.model.fit(input_data=X_train, labels=y_train, epochs=epochs,
                                     batch_size=batch_size, validation_data=X_val,
                                     validation_labels=y_val, validate=True)
            val_loss_list.append(results[1][0])

        return np.mean(val_loss_list)

    def search(self, training_data, labels, epochs, cv=None, validation_data=None, validation_labels=None):
        """Perform hyperparameter tuning on the selected model. Sets the best_params attribute"""
        optimal_loss = float('inf')
        best_params = [0, 0, 0]
        # search through the grid by training using all combinations of params on grid
        for learning_rate in self.learning_rates:
            for batch_size in self.batch_sizes:
                for decay_rate in self.decay_rates:
                    print('Testing Hyperparameters: Learning Rate = {}, Batch Size = {}, Momentum = {}'
                          .format(learning_rate, batch_size, decay_rate))
                    if cv:
                        # if cross-validation selected, find average val loss using k-fold cv on training data
                        val_loss = self._cross_validate(cv, training_data, labels, learning_rate, decay_rate,
                                                        batch_size, epochs)
                    else:
                        # if no cv, use the validation set for validation
                        self.model.compile(optimiser=self.optimiser, loss=self.loss, metric=self.metric,
                                           learning_rate=learning_rate, decay_rate=decay_rate)
                        results = self.model.fit(input_data=training_data, labels=labels, epochs=epochs,
                                                 batch_size=batch_size, validation_data=validation_data,
                                                 validation_labels=validation_labels, validate=True)
                        val_loss = results[1][0]
                    if val_loss < optimal_loss:  # store the params if they yield the best results so far
                        optimal_loss = val_loss
                        best_params[0] = learning_rate
                        best_params[1] = batch_size
                        best_params[2] = decay_rate

        self.best_params = best_params  # save the best params as an attribute of GridSearchTuner
        print('Best hyperparameters found: {}'.format(best_params))


