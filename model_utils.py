from tensorflow.python.keras.models import Model, load_model, Sequential
try: from tensorflow.python.keras.callbacks.callbacks import ModelCheckpoint
except:  from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Lambda


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if "relu" in activation_type: return activations.relu
    elif "sigmoid" in activation_type: return activations.sigmoid

class C2D_BN_A:
    def __init__(self, filters, kernel_size, strides, padding = "valid", useBias = True, activation = "relu", name = ""):
        self.conv = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, use_bias=useBias, name = name + "_Conv2D")
        self.bn = BatchNormalization(name = name + "_BatchNorm2D")
        self.act = get_activation(activation)
        
    def __call__(self, inputs):
        conv_out = self.conv(inputs)
        bn_out = self.bn(conv_out)
        act_out = self.act(bn_out)
        return act_out

class CT2D_BN_A:
    def __init__(self, filters, kernel_size, strides, padding = "valid", useBias = True, activation = "relu", name = ""):
        self.conv = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, use_bias=useBias, name = name + "_ConvTranspose2D")
        self.bn = BatchNormalization(name = name + "_BatchNorm2D")
        self.act = get_activation(activation)
        
    def __call__(self, inputs):
        conv_out = self.conv(inputs)
        bn_out = self.bn(conv_out)
        act_out = self.act(bn_out)
        return act_out
    
class C2D_ACB:
    def __init__(self, filters, kernel_size = 3, strides = 2, padding = "same", useBias = True, activation = "relu", name = ""):
        self.conv_v = Conv2D(filters, kernel_size=(kernel_size, 1), strides = strides, padding = "same", name = name + "_Conv2D_V")
        self.conv_h = Conv2D(filters, kernel_size=(1, kernel_size), strides = strides, padding = "same", name = name + "_Conv2D_H")
        self.conv_m = Conv2D(filters, kernel_size=(1, kernel_size), strides = strides, padding = "same", name = name + "_Conv2D_M")
        self.activation = activation
        
    def __call__(self, inputs):
        v_out = self.conv_v(inputs)
        h_out = self.conv_h(inputs)
        m_out = self.conv_m(inputs)
        added_out = Add()([v_out, m_out, h_out])
        bn_out = BatchNormalization()(added_out)
        act_out = get_activation(self.activation)(bn_out)
        return act_out

class CT2D_ACB:
    def __init__(self, filters, kernel_size = 3, strides = 2, padding = "same", useBias = True, activation = "relu", name = ""):
        self.conv_v = Conv2DTranspose(filters, kernel_size=(kernel_size, 1), strides = strides, padding = "same", name = name + "_Conv2D_V")
        self.conv_h = Conv2DTranspose(filters, kernel_size=(1, kernel_size), strides = strides, padding = "same", name = name + "_Conv2D_H")
        self.conv_m = Conv2DTranspose(filters, kernel_size=(1, kernel_size), strides = strides, padding = "same", name = name + "_Conv2D_M")
        self.activation = activation
        
    def __call__(self, inputs):
        v_out = self.conv_v(inputs)
        h_out = self.conv_h(inputs)
        m_out = self.conv_m(inputs)
        added_out = Add()([v_out, m_out, h_out])
        bn_out = BatchNormalization()(added_out)
        act_out = get_activation(self.activation)(bn_out)
        return act_out

class ContractiveLoss:
    def __init__(self, model, enc_layer_name = 'encodings', lam = 1e-4):
        self.model = model
        self.lam = lam
        self.enc_layer_name = enc_layer_name
        self.first = True

    def contractive_loss(self, y_pred, y_true):
        mse = K.mean(K.square(y_true - y_pred), axis=1)
        if self.first: self.first = False; return mse
        W = K.variable(value=self.model.get_layer(self.enc_layer_name).get_weights()[0])  # N x N_hidden
        W = K.transpose(W) # N_hidden x N
        h = self.model.get_layer(self.enc_layer_name).output
        dh = h * (1 - h)  # N_batch x N_hidden
        contractive = self.lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)
        return mse + contractive