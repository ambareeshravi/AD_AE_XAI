from model_utils import *
from utils import *
from data import *
from sklearn.metrics import roc_auc_score

from c2d_models import *

class Tester:
    def __init__(
        self,
        model,
        test_data
    ):
        if not isinstance(model, str):
            self.model = model
        else:
            self.model = load_model(model)
        self.test_data = test_data
        
    def test(self, batch_size = 64, return_results = True):
        results = dict()
        normal_images = self.test_data.normal_data
        for anomaly_type, anomaly_images in self.test_data.abnormal_data.items():
            labels = np.array([self.test_data.NORMAL_LABEL] * len(normal_images) + [self.test_data.ABNORMAL_LABEL] * len(anomaly_images))
            images = np.concatenate((normal_images, anomaly_images))
            reconstructions = self.model.predict(images)
            losses = np.sum((reconstructions - images)**2, axis = (1,2,3))
            normalized = (losses - losses.min()) / (losses.max() - losses.min())
            anomaly_roc_auc_score = roc_auc_score(labels, normalized)
            results[anomaly_type] = {
                "targets": labels,
                "losses": losses,
                "normalized": normalized,
                "roc_auc_score": anomaly_roc_auc_score,
            }
        results["mean_roc_auc_score"] = np.mean([results[at]["roc_auc_score"] for at in results.keys()])
        
        for key in results.keys():
            try: print(key, results[key]["roc_auc_score"])
            except: print(key, results[key])
                
        if return_results: return results
        else: return True