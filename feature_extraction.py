import pickle
import tensorflow as tf
import numpy as np
tf.python.control_flow_ops = tf

# TODO: import Keras layers you need here
from keras.layers import Input, Dense, Flatten
from keras.models import Model


flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
#additional flags
flags.DEFINE_integer('epoch', 50, "Number of Epochs")
flags.DEFINE_integer('batch_size', 256, "Batch Size")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print("Features train = ", X_train.shape, ". Labels train = ", y_train.shape)
    print("Features validation = ", X_val.shape, ". Labels validation = ", y_val.shape)

    n_classes = len(np.unique(y_train))
    
    print("Number of Classes = ", n_classes)
    
    

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    input_shape = X_train.shape[1:]
    inputs = Input(shape=input_shape)

    #Feature extraction... put an additional fully conected layer... flatten first...
    x = Flatten()(inputs)
    x = Dense(n_classes, activation='softmax')(x)
    
    #Now I define the model (inputs, outputs)
    model = Model(inputs, x)

    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
    model.fit(X_train, y_train, FLAGS.batch_size, FLAGS.epoch, validation_data=(X_val, y_val), shuffle=True)



# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
