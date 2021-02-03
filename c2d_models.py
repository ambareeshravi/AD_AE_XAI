from model_utils import *

class C2D_AE_128_3x3:
    def __init__(
        self,
        input_shape = 128,
        channels = 3,
        filters_count = [64,64,96,96,128],
        useACB = False
    ):
        self.input_shape = input_shape if isinstance(input_shape, tuple) else (input_shape, input_shape)
        self.channels = channels
        self.filters_count = filters_count
        self.useACB = useACB
        
        if self.useACB:
            self.conv_layer = C2D_ACB
            self.conv_transpose_layer = CT2D_ACB
            self.__name__ = "C2D_AE_128_3x3"
        else:
            self.conv_layer = C2D_BN_A
            self.conv_transpose_layer = CT2D_BN_A
            self.__name__ = "C2D_ACB_AE_128_3x3"
            
        self.model = self.get_model()
            
    def get_model(self,):
        inputs = Input(shape=tuple(list(self.input_shape) + [self.channels]))
        conv1 = self.conv_layer(filters = self.filters_count[0], kernel_size = 3, strides = 2, activation = 'relu', name = "conv1")(inputs)
        conv2 = self.conv_layer(filters = self.filters_count[1], kernel_size = 3, strides = 2, activation = 'relu', name = "conv2")(conv1)
        conv3 = self.conv_layer(filters = self.filters_count[2], kernel_size = 3, strides = 2, activation = 'relu', name = "conv3")(conv2)
        conv4 = self.conv_layer(filters = self.filters_count[3], kernel_size = 3, strides = 2, activation = 'relu', name = "conv4")(conv3)
        encodings = C2D_BN_A(filters = self.filters_count[4], kernel_size = 3, strides = 2, activation = 'relu', name = "encodings")(conv4)

        convt1 = CT2D_BN_A(filters = self.filters_count[3], kernel_size = 4 if self.useACB else 3, strides = 2, activation = 'relu', name = "convt1")(encodings)
        convt2 = self.conv_transpose_layer(filters = self.filters_count[2], kernel_size = 3, strides = 2, activation = 'relu', name = "convt2")(convt1)
        convt3 = self.conv_transpose_layer(filters = self.filters_count[1], kernel_size = 3, strides = 2, activation = 'relu', name = "convt3")(convt2)
        convt4 = self.conv_transpose_layer(filters = self.filters_count[0], kernel_size = 3, strides = 2, activation = 'relu', name = "convt4")(convt3)
        reconstructions = self.conv_transpose_layer(filters = self.channels, kernel_size = 4, strides = 2, activation = 'sigmoid', name = "reconstructions")(convt4)
        return Model(inputs, reconstructions)
    
    def __call__(self):
        return self.model