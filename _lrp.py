from keras_explain.lrp import LRP
from keras import backend as K
from keras.models import Model

class E_LRP:
    def __init__(self, model):
        self.model = model

    def explain(dataset, batch_size):
        with DeepExplain(session=K.get_session()) as de:
            input_tensor = self.model.layers[0].input
            de_model = Model(inputs=input_tensor, outputs = self.model.layers[-1].output)
            target_tensor = de_model(input_tensor)

            return de.explain(elrp, target_tensor, input_tensor, dataset, ys=dataset, batch_size=batch_size)
