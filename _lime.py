from utils import *

from lime import lime_image
from skimage.segmentation import mark_boundaries

class LimeExplainer:
    def __init__(self, model, num_features, num_samples = 1000, batch_size = 64):
        self.model = model
        self.explainer = lime_image.LimeImageExplainer()
        self.num_features = num_features
        self.num_samples = num_samples
        self.batch_size = batch_size
        
    def explain(self, dataset, results_path = "", mask_features = 5, anomaly_type = None, save_result = True):
        results = list()
        for idx, data in enumerate(dataset):
            if not isinstance(data, np.ndarray): data = np.array(data)
            exp = self.explainer.explain_instance(
                data.astype('double'),
                self.model.predict,
                num_features=self.num_features,
                num_samples=self.num_samples,
                batch_size=self.batch_size
            )
            tmp, mask = exp.get_image_and_mask(
                exp.top_labels[0],
                positive_only=True, # high loss -> towards probab 1 -> anomaly -> so positive class is true
                num_features = mask_features,
                hide_rest=False # make sure hide_rest is False -> needs tmp normalization
            )
            tmp = normalize(tmp)
            result_mask = mark_boundaries(tmp / 2 + 0.5, mask)
            results.append(result_mask)
            if save_result: save_image(result_mask, join_paths([results_path, '%s_%02d.png' % (anomaly_type, idx)]))
        return np.array(results)