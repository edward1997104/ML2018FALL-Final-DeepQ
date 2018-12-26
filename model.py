from keras.models import Sequential, load_model
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, InputLayer, LeakyReLU, BatchNormalization, Dropout, GlobalAveragePooling2D, Input, Conv2D, AvgPool2D, multiply, Lambda
import keras.applications
import keras.backend as K
from sklearn.metrics import roc_curve, auc, roc_auc_score
from dataset import Training_Generator, Testing_Generator, load_train_data, load_test_idxs
import tensorflow as tf
import numpy as np
from scipy import interp
import pandas as pd
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description='Enter Argument for model')

    # training flag
    parser.add_argument('--training', type = lambda x: (str(x).lower() == 'true'), default = True)

    # add position for saving
    parser.add_argument("--model", type=str, default = 'baseline.h5')
    parser.add_argument("--output", type=str, default = 'output.csv')


    # add model parms:
    parser.add_argument("--learning_rate", type=float, default = 1e-5)
    parser.add_argument("--epochs", type=int, default = 20)
    parser.add_argument("--drop_out", type=float, default = 0.5)
    parser.add_argument("--batch_size", type=int, default = 32)
    parser.add_argument("--activation", type=str, default = 'elu')


    return parser.parse_args()

args = arg_parser()

def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

class XRAY_model():
    
    def __init__(self, MODEL, preprocess_func = None, use_attn = True, input_dim = (150, 150, 3), output_dim = 14, learning_rate = 0.00001, epochs = 20, drop_out = 0.5, batch_size = 32, activation = 'elu'):

        # parms:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.drop_out = drop_out
        self.batch_size = batch_size


        inputs = Input(shape = input_dim)
        processed_inputs = inputs

        if preprocess_func:
            processed_inputs = Lambda(preprocess_func) (processed_inputs)

        pretrained_model = MODEL(weights='imagenet', include_top=False, input_shape = self.input_dim)
        
        # freeze the weights first
        pretrained_model.trainable = False

        model_output = pretrained_model(processed_inputs)

        if use_attn:
            pt_depth = model_output.shape[-1]
            bn_features = BatchNormalization()(model_output)
            attn_layer = Conv2D(128, kernel_size = (1,1), padding = 'same', activation = 'elu')(bn_features)
            attn_layer = Conv2D(32, kernel_size = (1,1), padding = 'same', activation = 'elu')(attn_layer)
            attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'elu')(attn_layer)
            attn_layer = AvgPool2D((2,2), strides = (1,1), padding = 'same')(attn_layer) # smooth results
            attn_layer = Conv2D(1, kernel_size = (1,1), padding = 'valid', activation = 'sigmoid')(attn_layer)

            # fan it out to all of the channels
            up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
                        activation = 'linear', use_bias = False, kernel_initializer = keras.initializers.Ones())
            up_c2.trainable = False
            attn_layer = up_c2(attn_layer)
            self.attn_layer = attn_layer

            # normalize the feature
            mask_features = multiply([attn_layer, bn_features])
            gap_features = GlobalAveragePooling2D()(mask_features)
            gap_mask = GlobalAveragePooling2D()(attn_layer)
            model_output = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
        else:
            model_output = GlobalAveragePooling2D()(model_output)
        
        # Dense Layers
        output = Dropout(self.drop_out) (model_output)
        output = Dense(128, activation = activation)(output)
        output = Dropout(self.drop_out) (output)
        output = Dense(self.output_dim, activation = 'sigmoid') (output)


        auc_roc = as_keras_metric(tf.metrics.auc)
        recall = as_keras_metric(tf.metrics.recall)
        precision = as_keras_metric(tf.metrics.precision)
        f1_measure = as_keras_metric(tf.contrib.metrics.f1_score)


        # Build Model
        self.model = Model(inputs = [inputs], outputs = [output])

        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['binary_accuracy', 'mae', auc_roc, recall, precision, f1_measure])
        self.model.summary()


        
    def fit(self, x, y, validation_ratio = 0.1):

        # fit the data
        print ("Start Training model")
        test_idx = int(len(x) * validation_ratio)
        X_train, y_train, X_test, y_test = x[test_idx:], y[test_idx:], x[:test_idx], y[:test_idx]
        training_gen = Training_Generator(X_train, y_train, self.batch_size, reshaped_size = self.input_dim[:-1])
        validation_gen = Training_Generator(X_test, y_test, self.batch_size, reshaped_size = self.input_dim[:-1])
        hist = self.model.fit_generator(
            training_gen,
            validation_data = validation_gen, epochs = self.epochs)
        
        print ("Done Training model")
        print ("AVG AUC:", self.score(X_test, y_test))
        return hist
    
    # data did preprocessing
    def inference(self, x):
        test_gen = Testing_Generator(x, self.batch_size, reshaped_size = self.input_dim[:-1])
        return self.model.predict_generator(test_gen, verbose = 1)

    def score(self, x, y):
        # make predicition
        y_pred = self.predict(x)
        return roc_auc_score(y, y_pred, average = "macro")

    # pain data without preprocessing
    def predict(self, x):
        return self.inference(x)

    def save_weight(self, path = 'baseline.h5'):
        print ("Start Saving model")
        self.model.save(path)
        print ("Done Saving model")
        return

    def load_weight(self, path = 'baseline.h5'):
        print ("Start Loading model")
        self.model.load_weights(path)
        print ("Done Loading model")
        return

def output_csv(pred, idx, disease_path = './data/ntu_final_2018/classname.txt', output_path = './output.csv'):
    
    disease_name = []

    with open(disease_path, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                disease_name.append(line[:-1])
            else:
                disease_name.append(line)
    
    # build dic
    dic = {}
    dic['Id'] = idx
    for i in range(len(disease_name)):
        dic[disease_name[i]] = pred[:,i]
    
    df = pd.DataFrame.from_dict(dic)
    df.to_csv(output_path, index = False)

    print('Done saving at', output_path)
    return

if __name__ == "__main__":
    
    model = XRAY_model(keras.applications.VGG16, 
                        preprocess_func = keras.applications.vgg16.preprocess_input,
                        input_dim = (224,224,3), use_attn = True, learning_rate = args.learning_rate,
                        epochs = args.epochs, drop_out = args.drop_out, batch_size = args.batch_size, activation = args.activation)
    
    if args.training:
        X_train, y_label, unlabelled = load_train_data()
        model.fit(X_train, y_label)
        model.save_weight(args.model)
    else:
        model.load_weight(args.model)

    test_idxs, output_idx = load_test_idxs()

    pred = model.predict(test_idxs)

    output_csv(pred, output_idx, output_path = args.output)