from _lime import *
from _counterfactual import *
from utils import *
from model_utils import *
from data import *
from c2d_models import *

'''
Example configuration

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
    'output_path': "",
    'test_data_path': "",
    'training_data_path': ""
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
            num_features = configuration['lime']['num_features'],
            num_samples = configuration['lime']['num_samples'],
            batch_size = configuration['lime']['batch_size']
        )

        self.counterfactual_explainer = Counterfactual(
            model = model
        )

    def run(self):
        self.run_counterfactual()
        self.run_lime()

    def run_lime(self):
        _output_prefix = self.configuration['output_path']
        _output_path = '%slime/' % (_output_prefix)
        create_directory(_output_path)
        _mask_features = self.configuration['lime']['mask_features']
        for key in self.test_dict.keys():
            self.lime_explainer.explain(
                dataset = self.test_dict[key], mask_features = _mask_features, results_path = _output_path, anomaly_type = key, save_result = True
            )

    def run_counterfactual(self):
        _output_prefix = self.configuration['output_path']
        _output_path = '%scounterfactual/' % (_output_prefix)
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