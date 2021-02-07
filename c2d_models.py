from model_utils import *

class C2D_AE_128_3x3(Model):
    def __init__(
        self,    
        input_shape = 128,
        channels = 3,
        filters_count = [64,64,96,96,128],
        useACB = False,
        max_value = 1,
        isTrain = True
    ):
        super(C2D_AE_128_3x3, self).__init__()
        self.i_shape = input_shape if isinstance(input_shape, tuple) else (input_shape, input_shape)
        self.c = channels
        self.filters_count = filters_count
        self.useACB = useACB
        self.isTrain = isTrain
        self.max_value = max_value
        
        if self.useACB:
            self.conv_layer = C2D_ACB
            self.conv_transpose_layer = CT2D_ACB
            self.__name__ = "C2D_ACB_AE_128_3x3"
        else:
            self.conv_layer = C2D_BN_A
            self.conv_transpose_layer = CT2D_BN_A
            self.__name__ = "C2D_AE_128_3x3"
            
        self.conv1 = self.conv_layer(filters = self.filters_count[0], kernel_size = 3, strides = 2, activation = 'relu', name = "conv1")
        self.conv2 = self.conv_layer(filters = self.filters_count[1], kernel_size = 3, strides = 2, activation = 'relu', name = "conv2")
        self.conv3 = self.conv_layer(filters = self.filters_count[2], kernel_size = 3, strides = 2, activation = 'relu', name = "conv3")
        self.conv4 = self.conv_layer(filters = self.filters_count[3], kernel_size = 3, strides = 2, activation = 'relu', name = "conv4")
        self.encoder_conv = C2D_BN_A(filters = self.filters_count[4], kernel_size = 3, strides = 2, activation = 'relu', name = "encodings")

        self.convt1 = CT2D_BN_A(filters = self.filters_count[3], kernel_size = 4 if self.useACB else 3, strides = 2, activation = 'relu', name = "convt1")
        self.convt2 = self.conv_transpose_layer(filters = self.filters_count[2], kernel_size = 3, strides = 2, activation = 'relu', name = "convt2")
        self.convt3 = self.conv_transpose_layer(filters = self.filters_count[1], kernel_size = 3, strides = 2, activation = 'relu', name = "convt3")
        self.convt4 = self.conv_transpose_layer(filters = self.filters_count[0], kernel_size = 3, strides = 2, activation = 'relu', name = "convt4")
        self.reconstructor_convt = self.conv_transpose_layer(filters = self.c, kernel_size = 4, strides = 2, activation = 'sigmoid', name = "reconstructions")
        
        self.loss_layer = Lambda(lambda x: (K.expand_dims(K.sum(K.square(x[0]-x[1]), axis = (-3,-2,-1)), axis = -1))/self.max_value)
        
        self.model = self.get_model()
    
    def encoder(self, x):
        conv1o = self.conv1(x)
        conv2o = self.conv2(conv1o)
        conv3o = self.conv3(conv2o)
        conv4o = self.conv4(conv3o)
        encodings = self.encoder_conv(conv4o)
        return encodings
    
    def decoder(self, x):
        convt1o = self.convt1(x)
        convt2o = self.convt2(convt1o)
        convt3o = self.convt3(convt2o)
        convt4o = self.convt4(convt3o)
        reconstructions = self.reconstructor_convt(convt4o)
        return reconstructions
    
    def call(self, x):
        encodings = self.encoder(x)
        reconstructions = self.decoder(encodings)
        if self.isTrain: return reconstructions
        else: return self.loss_layer([x, reconstructions])
    
    def get_model(self):
        x = Input(shape=tuple(list(self.i_shape) + [self.c]))
        return Model(inputs=[x], outputs=self.call(x))