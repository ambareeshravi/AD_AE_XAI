from deepexplain.tensorflow import DeepExplain
# from keras_explain.lrp import LRP
from keras import backend as K
from keras.models import Model

class E_LRP:
    def __init__(
        self,
        model_path,
        max_loss_value = 1000
    ):
        self.model_path = model_path
        self.max_loss_value = max_loss_value

    def explain(self, dataset, batch_size = 64):
        with DeepExplain(session=K.get_session()) as de:
            # Model should be loaded under the de scope
            self.model = get_model(self.model_path, rec = True, max_value=self.max_loss_value)
            input_tensor = self.model.layers[0].input
            de_model = Model(inputs=input_tensor, outputs = self.model.layers[-1].output)
            target_tensor = de_model(input_tensor)
            return de.explain("elrp", target_tensor, input_tensor, dataset, ys=dataset, batch_size=batch_size)
