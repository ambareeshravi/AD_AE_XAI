from utils import *
from model_utils import *
from data import *
from c2d_models import *

from _lrp import *
from _lime import *
from _counterfactual import *
from _shap import *

'''
Example self.configuration

confg = {
    'lime': {
        'batch_size': 32,
        'num_features': 128*128,
        'mask_features': 5,
        'num_samples': 1000
    },
    'counterfactual': {
        'threshold_pct': 0.98,
        'block_size': 32
    },
    'shap': {
        'blend_alpha': 0.85,
        'background_samples': 100
    },
    'elrp': {
        'model_path': "",
        'max_loss_value': 1000
    },
    'output_path': "",
    'test_data_path': "",
    'train_data': None, # np.array of normal images
    'self.configuration': "",
    'max_loss_value': 1000,
    'batch_size': 32,
}
'''
class Pipeline:
    def __init__(self, model, loss_model, configuration, normal_tag = "normal"):
        self.model = model
        self.loss_model = loss_model
        self.configuration = configuration
        self.normal_tag = normal_tag

        self.test_dict = dict()
        for class_dir in read_directory_contents(self.configuration['test_data_path']):
            self.test_dict[class_dir.split("/")[-1]] = np.array([read_image(file, resize_to=(128,128))/255. for file in read_directory_contents(class_dir)])
        
        self.lime_explainer = LimeExplainer(
            model = loss_model,
            num_features = self.configuration['lime']['num_features'],
            num_samples = self.configuration['lime']['num_samples'],
            batch_size = self.configuration['lime']['batch_size']
        ) if 'lime' in self.configuration.keys() else None

        self.counterfactual_explainer = Counterfactual(
            model = model
        ) if 'counterfactual' in self.configuration.keys() else None
        
        self.lrp_explainer = E_LRP(
            model_path = self.configuration['lime']['model_path'],
            max_loss_value = self.configuration['lime']['max_loss_value']
        ) if 'lrp' in self.configuration.keys() else None
        
        self.shap_explainer = SHAP_Explainer(
            model = self.loss_model,
            X_train = self.configuration['train_data'],
            background_samples = self.configuration['shap']['background_samples'],
            blend_alpha = self.configuration['shap']['blend_alpha'],
        ) if 'shap' in self.configuration.keys() else None

    def run(self):
        for (method_name, method) in zip(["e-LRP", "LIME", "CounterFactual", "SHAP"], [self.run_elrp, self.run_lime, self.run_counterfactual, self.run_shap]):
            INFO("%s ready"%(method_name))
            try: method(method_name)
            except Exception as e: print("%s failed\nError: %s"%(method_name, e))
            
    def run_elrp(self, method_name):
        _output_path = join_paths([self.configuration['output_path'], method_name, "/"])
        create_directory(_output_path)
        for key in self.test_dict.keys():
            lrp_results = self.lrp_explainer.explain(self.test_dict[key], batch_size = self.configuration['batch_size'])
            for idx, lrp_result in enumerate(lrp_results):
                save_image(lrp_result, join_paths([_output_path, "%HBox(children=(FloatProgress(value=0.0, max=1000.0), HTML(value='')))s_%d.png"%(key, idx)]))

    def run_lime(self, method_name):
        _output_path = join_paths([self.configuration['output_path'], method_name, "/"])
        create_directory(_output_path)
        _mask_features = self.configuration['lime']['mask_features']
        for key in self.test_dict.keys():
            self.lime_explainer.explain(
                dataset = self.test_dict[key], mask_features = _mask_features, results_path = _output_path, anomaly_type = key, save_result = True
            )

    def run_counterfactual(self, method_name):
        _output_path = join_paths([self.configuration['output_path'], method_name, "/"])
        create_directory(_output_path)
        _threshold_pct = self.configuration['counterfactual']['threshold_pct']
        _block_size = self.configuration['counterfactual']['block_size']

        _norms = self.test_dict[self.normal_tag]
        for key in self.test_dict.keys():
            if key != self.normal_tag:
                self.counterfactual_explainer.explain(
                    normal_samples = _norms,
                    anomalous_samples = self.test_dict[key],
                    threshold_pct = _threshold_pct, 
                    block_size = _block_size,
                    anomaly_type = key,
                    save_path = _output_path,
                    save_results = True
                )
    
    def run_shap(self, method_name):
        _output_path = join_paths([self.configuration['output_path'], method_name, "/"])
        create_directory(_output_path)
        for key in self.test_dict.keys():
            shap_results = self.shap_explainer.explain(
                X_test = self.test_dict[key],
            )
            for idx, shap_result in enumerate(shap_results):
                save_image(shap_result, join_paths([_output_path, "%s_%d.png"%(key, idx)]))