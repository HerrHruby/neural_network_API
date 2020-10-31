import matplotlib.pyplot as plt
import pickle


def plotter(train_name, validation_name):

    with open(train_name, 'rb') as fp:
        train_data = pickle.load(fp)

    with open(validation_name, 'rb') as fp:
        validation_data = pickle.load(fp)

    X = [i for i in range(0, len(train_data))]
    train_loss = [i[0] for i in train_data]
    train_acc = [i[1] for i in train_data]
    val_loss = [i[0] for i in validation_data]
    val_acc = [i[1] for i in validation_data]

    plt.scatter(X, train_loss, s=6, label='Training Loss')
    plt.scatter(X, val_loss, s=6, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.scatter(X, train_acc, s=6, label='Training Loss')
    plt.scatter(X, val_acc, s=6, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_file = 'test_name_training'
    validation_file = 'test_name_validation'
    plotter(train_file, validation_file)



