import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.activations import relu, softmax, tanh, sigmoid
from tensorflow.keras.applications import vgg19, resnet50
from typing import Tuple, List
from .ResNetBuilder import build_resnet_model


def compile_model_default(model: keras.Sequential) -> keras.Sequential:
    model.compile(optimizer=optimizers.Adam(), loss=losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[metrics.CategoricalAccuracy(name='accuracy')])
    return model


class WaveModels:

    @staticmethod
    def build_base_model(input_shape: Tuple[int, int], num_classes: int, compile_model: bool = True,
                         print_summary: bool = True):
        # Build model
        model = keras.Sequential(layers=[
            layers.InputLayer(input_shape=input_shape),
            layers.Conv1D(filters=8, kernel_size=64, padding='causal', activation=relu),
            layers.AveragePooling1D(pool_size=4),
            layers.Conv1D(filters=16, kernel_size=64, padding='causal', activation=relu),
            layers.AveragePooling1D(pool_size=4),
            layers.Conv1D(filters=32, kernel_size=64, padding='causal', activation=relu),
            layers.AveragePooling1D(pool_size=4),
            layers.Conv1D(filters=64, kernel_size=64, padding='causal', activation=relu),
            layers.AveragePooling1D(pool_size=4),
            layers.Conv1D(filters=128, kernel_size=64, padding='causal', activation=relu),
            layers.AveragePooling1D(pool_size=4),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(units=512, activation=relu),
            layers.Dense(units=num_classes, activation=softmax)
        ])
        # Print model summary if required
        if print_summary:
            print(model.summary())
        # If compile_model is set to True, use some default parameters for optimizer, loss and metrics
        if compile_model:
            model = compile_model_default(model)
        return model

    @staticmethod
    def build_parallel_cnn(input_shape: Tuple[int, int], num_classes: int, kernel_sizes: List[int], n_convs: int,
                           pool_size: int = 4, compile_model: bool = True, print_summary: bool = True) -> Model:
        input_layer = layers.Input(shape=input_shape, name='Input_Layer')
        concat_layers = []
        for kernel_size in kernel_sizes:
            n_filters = 16
            X = input_layer
            for i in range(n_convs):
                X = layers.Conv1D(filters=n_filters, kernel_size=kernel_size, padding='causal', activation=relu,
                                  name=f'Kernel_Size_{kernel_size}_Conv{i+1}')(X)
                X = layers.AveragePooling1D(pool_size=pool_size, name=f'Kernel_Size_{kernel_size}_Pool{i+1}')(X)
                n_filters *= 2
            concat_layers.append(X)
        X = layers.concatenate(inputs=concat_layers, axis=1, name='Concatenate_Layer')
        X = layers.Dropout(rate=0.5)(layers.Flatten(name='Flatten-Layer')(X))
        clf = layers.Dense(units=num_classes, activation=softmax, name='Classifier')(X)

        model = Model(inputs=input_layer, outputs=clf)

        # Print model summary if required
        if print_summary:
            print(model.summary())
        # If compile_model is set to True, use some default parameters for optimizer, loss and metrics
        if compile_model:
            model = compile_model_default(model)
        return model

    @staticmethod
    def build_wavenet_model(input_shape: Tuple[int, int], num_classes: int, k_layers: int, num_filters: int = 32,
                            compile_model: bool = True, print_summary: bool = True) -> Model:
        input_layer = layers.Input(shape=input_shape, name='Input_Layer')
        X = input_layer
        skip_connections = []
        for i in range(1, k_layers + 1):
            dilated_conv = layers.Conv1D(filters=num_filters, kernel_size=2, padding='causal', dilation_rate=2 ** i,
                                         name=f'Block_{i}_Dilated_Conv')(X)
            conv_dropout = layers.Dropout(rate=0.5, name=f'Block_{i}_Dropout')(dilated_conv)
            tanh_act = layers.Activation(activation=tanh, name=f'Block_{i}_Tanh_Activation')(conv_dropout)
            sigmoid_act = layers.Activation(activation=sigmoid, name=f'Block_{i}_Sigm_Activation')(conv_dropout)
            multiply = layers.multiply([tanh_act, sigmoid_act], name=f'Block_{i}_Multiply')
            conv = layers.Conv1D(filters=num_filters, kernel_size=1, name=f'Block_{i}_Convolution')(multiply)
            skip_connections.append(conv)
            X = layers.add([conv, X], name=f'Block_{i}_Add')
        add_skips = layers.add(skip_connections, name='Add_Skip_Connections')
        act = layers.Activation(activation=relu, name='ReLu_Activation')(add_skips)
        conv = layers.Conv1D(filters=num_filters, kernel_size=1, padding='causal', activation=relu,
                             name='Convolution')(act)
        pooling = layers.AveragePooling1D(pool_size=2, name='Pooling')(conv)
        flatten = layers.Flatten(name='Flatten')(pooling)
        dropout = layers.Dropout(rate=0.5, name='Dropout')(flatten)
        clf = layers.Dense(units=num_classes, activation=softmax, name='Classifier')(dropout)

        model = Model(inputs=input_layer, outputs=clf)

        # Print model summary if required
        if print_summary:
            print(model.summary())
        # If compile_model is set to True, use some default parameters for optimizer, loss and metrics
        if compile_model:
            model = compile_model_default(model)
        return model


class SpectralModels:

    @staticmethod
    def build_base_cnn(input_shape: Tuple[int, int, int], num_classes: int, compile_model: bool = True,
                       print_summary: bool = True) -> keras.Sequential:
        model = keras.Sequential(layers=[
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(filters=8, kernel_size=3, padding='same', activation=relu),
            layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=relu),
            layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=relu),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(rate=0.5),
            layers.Dense(units=64, activation=relu),
            layers.Dense(units=num_classes, activation=softmax)
        ])
        # Print model summary if required
        if print_summary:
            print(model.summary())
        # If compile_model is set to True, use some default parameters for optimizer, loss and metrics
        if compile_model:
            model = compile_model_default(model)
        return model

    @staticmethod
    def build_complex_cnn(input_shape: Tuple[int, int, int], num_classes: int, compile_model: bool = True,
                          print_summary: bool = True) -> keras.Sequential:
        model = keras.Sequential(layers=[
            layers.InputLayer(input_shape=input_shape),
            layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=relu),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=relu),
            layers.MaxPooling2D(),
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation=relu),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(rate=0.5),
            layers.Dense(units=128, activation=relu),
            layers.Dense(units=64, activation=relu),
            layers.Dense(units=num_classes, activation=softmax)
        ])
        # Print model summary if required
        if print_summary:
            print(model.summary())
        # If compile_model is set to True, use some default parameters for optimizer, loss and metrics
        if compile_model:
            model = compile_model_default(model)
        return model

    @staticmethod
    def build_residual_model(input_shape: Tuple[int, int, int], residual_blocks: int, num_classes: int,
                             compile_model: bool = True, print_summary: bool = True) -> Model:
        depth = 9 * residual_blocks + 2
        model = build_resnet_model(input_shape=input_shape, depth=depth, num_classes=num_classes)
        # Print model summary if required
        if print_summary:
            print(model.summary())
        # If compile_model is set to True, use some default parameters for optimizer, loss and metrics
        if compile_model:
            model = compile_model_default(model)
        return model

    @staticmethod
    def build_adapted_residual_model(input_shape: Tuple[int, int, int], residual_blocks: int, num_classes: int,
                                     compile_model: bool = True, print_summary: bool = True) -> Model:
        num_filters = 16
        input_layer = layers.Input(shape=input_shape)
        X = layers.Activation(activation=relu)(input_layer)
        for i in range(residual_blocks):
            # Build residual path
            res_X = layers.Conv2D(filters=num_filters*2, kernel_size=3, padding='same', activation=relu)(X)
            # Build Conv path
            conv_X = layers.Conv2D(filters=num_filters, kernel_size=3, padding='same', activation=relu)(X)
            conv_X = layers.BatchNormalization()(conv_X)
            conv_X = layers.Conv2D(filters=num_filters*2, kernel_size=3, padding='same', activation=relu)(conv_X)
            # Add paths
            X = layers.add([res_X, conv_X])
            X = layers.MaxPooling2D()(X)
            # Double the number of filters
            num_filters *= 2
        # Add classifier to network
        X = layers.Flatten()(X)
        X = layers.Dropout(0.5)(X)
        X = layers.Dense(units=512, activation=relu)(X)
        X = layers.Dropout(0.5)(X)
        y = layers.Dense(units=num_classes, activation=softmax)(X)

        # Build final model
        model = Model(inputs=input_layer, outputs=y)
        # Print model summary if required
        if print_summary:
            print(model.summary())
        # If compile_model is set to True, use some default parameters for optimizer, loss and metrics
        if compile_model:
            model = compile_model_default(model)
        return model


class ImageModels:

    @staticmethod
    def build_resnet50_transfer_net(input_shape: Tuple[int, int, int], num_classes: int, compile_model: bool = True,
                                    print_summary: bool = True) -> Model:
        input_layer = layers.Input(shape=input_shape)
        preprocess_layer = resnet50.preprocess_input(input_layer)
        base_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=preprocess_layer)
        base_model.trainable = False
        flatten_layer = layers.Flatten()(base_model.output)
        dropout_layer = layers.Dropout(rate=0.5)(flatten_layer)
        dense_layer = layers.Dense(units=256, activation=relu)(dropout_layer)
        dropout_layer = layers.Dropout(rate=0.5)(dense_layer)
        dense_layer = layers.Dense(units=128, activation=relu)(dropout_layer)
        dropout_layer = layers.Dropout(rate=0.5)(dense_layer)
        clf_layer = layers.Dense(units=num_classes, activation=softmax)(dropout_layer)

        # Create the final model
        model = Model(name='Classifier', inputs=[input_layer], outputs=[clf_layer])

        # Print model summary if required
        if print_summary:
            print(model.summary())
        # If compile_model is set to True, use some default parameters for optimizer, loss and metrics
        if compile_model:
            model = compile_model_default(model)
        return model

    @staticmethod
    def build_vgg19_feature_extractor(input_shape: Tuple[int, int, int], num_classes: int, compile_model: bool = True,
                                      print_summary: bool = True) -> Model:
        input_layer = layers.Input(shape=input_shape)
        preprocess_layer = vgg19.preprocess_input(input_layer)
        base_model = vgg19.VGG19(include_top=False, input_tensor=preprocess_layer, weights='imagenet')
        base_model.trainable = False
        flatten_layer = layers.Flatten()(base_model.output)
        dropout_layer = layers.Dropout(rate=0.5)(flatten_layer)
        dense_layer = layers.Dense(units=64, activation=relu)(dropout_layer)
        clf = layers.Dense(units=num_classes, activation=softmax)(dense_layer)

        model = Model(inputs=input_layer, outputs=clf)

        # Print model summary if required
        if print_summary:
            print(model.summary())
        # If compile_model is set to True, use some default parameters for optimizer, loss and metrics
        if compile_model:
            model = compile_model_default(model)
        return model
