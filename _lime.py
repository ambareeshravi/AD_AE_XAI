import numpy as np

from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image

class LimeExplainer:
    def __init__(self, model, num_features, batch_size):
        self.model = model
        self.explainer = lime_image.LimeImageExplainer()
        self.num_features = num_features
        self.batch_size = batch_size
    def explain(self, dataset, results_path, mask_features, anomaly_type):
        idx = 0
        for data in dataset:
            exp = self.explainer.explain_instance(np.array(data).astype('double'), 
                    self.model.predict, num_features=self.num_features,
                    batch_size=self.batch_size)
            tmp, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=True,
                    num_features = mask_features, hide_rest=True)
            result_mask = mark_boundaries(tmp / 2 + 0.5, mask)
            result_mask = (result_mask * 255).astype(np.uint8)
            result_mask = Image.fromarray(result_mask)
            result_mask.save('%s%s-%s.png' % (results_path, anomaly_type, str(idx)))
            idx += 1