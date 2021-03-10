from utils import *
from deepexplain.tensorflow import DeepExplain
from keras_explain.grad_cam import GradCam
from keras_explain.guided_bp import GuidedBP
# from keras_explain.lrp import LRP
from keras import backend as K
from keras.models import Model

class GradMethods:
    def __init__(
        self,
        model_path,
        de_output_layer = -1,
        debug = True
    ):
        self.model_path = model_path
        self.de_output_layer = de_output_layer
        self.debug = debug
        
    def run_ke(self, dataset):
        '''
        Runs methods from keras_explain library
        '''
        # Clear /tmp/ dir
        for fl in ["/tmp/checkpoint", "/tmp/gb_keras.h5"]  + glob("/tmp/guided_backprop_ckpt.*"):
            os.remove(fl)
        
        # Load model
        model = get_model(self.model_path, rec = True, max_value=self.max_loss_value)
        self.ke_methods = {
            "GuidedBP": GuidedBP(model),
            "GradCAM": GradCam(model, layer = -2),
        }
        
        ke_results = dict([(key, list()) for key in list(self.ke_methods.keys())])

        for (method, explainer) in self.ke_methods.items():
            if self.debug: print(method)
            for data in dataset:
                exp = explainer.explain(data, 1)
                ke_results[method].append(normalize(exp[0]))

        for key in ke_results.keys():
            ke_results[key] = np.array(ke_results[key])
        
        return ke_results
    
    def run_de(self, dataset):
        '''
        Runs method from deepexplain library
        '''
        self.de_methods = {
            "Gradient_Inputs": "grad*input",
            "Saliency": "saliency",
            "Integrated_Gradients": "intgrad",
#             "DeepLIFT": "deeplift",
            "e-LRP": "elrp",
            "Occlusion": "occlusion"
        }
        
        de_results = dict()
        
        with DeepExplain(session=K.get_session()) as de:
            # Model should be loaded under the de scope
            model = get_model(self.model_path, rec = True, max_value=self.max_loss_value)
            input_tensor = model.layers[0].input
            de_model = Model(inputs=input_tensor, outputs = model.layers[self.de_output_layer].output)
            target_tensor = de_model(input_tensor)
            
            for method_name, method_tag in self.de_methods.items():
            if self.debug: print(method_name)
                try: attributions = de.explain(method_tag, target_tensor, input_tensor, dataset, ys=dataset, batch_size = self.batch_size) # np.expand_dims(y_test, axis = -1))
                except Exception as e:
                    print("%s failed"%(method_name))
                    if self.debug: print("ERROR: %s"%(e))
                    continue
                de_results[method_name] = attributions
        return de_results

    def explain(dataset, max_loss_value = 1000, batch_size = 64):
        self.max_loss_value = max_loss_value
        self.batch_size = batch_size
        ke_results = self.run_ke(dataset)
        de_results = self.run_de(dataset)
        de_results.update(ke_results)
        return de_results