from utils import *

from skimage.metrics import mean_squared_error
from skimage.segmentation import mark_boundaries

def get_segments(img, height, width, imgheight, imgwidth):
    segments = []
    for i in range(0,imgheight, height):
        for j in range(0,imgwidth, width):
            box = (j, i, j+width, i+height)
            segments.append(box)
    return segments

def calculate_error(_img1, _img2):
    return mean_squared_error(_img1, _img2)

def get_reconstruction(_img, model):
    _reshaped = _img.reshape((1, 128, 128, 3))
    return model.predict(_reshaped)[0]

def minimum_edits(_img1, _img2, _threshold, _filter_size, model):
    img_height = np.array(_img1).shape[0]
    img_width = np.array(_img1).shape[1]
    segs1 = get_segments(_img1, _filter_size, _filter_size, img_height, img_width)
    segs2 = get_segments(_img2, _filter_size, _filter_size, img_height, img_width)
    edits = []

    current_img = np.array(_img1, copy=True)
    cur_iter = 0
    while True:
        if (cur_iter > len(segs1)):
            break
        max_error = 0
        seg_idx = 0
        for idx, seg in enumerate(segs2):
            edited_img = np.array(current_img, copy=True)
            edited_img[seg[0]: seg[2], seg[1]: seg[3], :] = _img2[seg[0]: seg[2], seg[1]: seg[3], :]
            edited_reconstruction = get_reconstruction(edited_img, model)
            edited_error = calculate_error(edited_img, edited_reconstruction)
            if edited_error > max_error:
                max_error = edited_error
                seg_idx = idx
        best_seg = segs2.pop(seg_idx)
        current_img[best_seg[0]: best_seg[2], best_seg[1]: best_seg[3], :] = _img2[best_seg[0]: best_seg[2], best_seg[1]: best_seg[3], :]
        edits.append(best_seg)
        if (max_error > _threshold):
            return edits
        cur_iter += 1
    return edits

def draw_boundaries(_edits, _img):
    masks = np.zeros((128, 128)).astype(int)
    for _edit in _edits:
        masks[_edit[0]: _edit[2], _edit[1]: _edit[3]] = 1
    _final_img = mark_boundaries(_img / 2 + 0.5, masks)
    return _final_img # returns value between 0 and 1
#     _final_img = (_final_img * 255).astype(np.uint8)
#     _im = Image.fromarray(_final_img)
#     _im.save(_path)


class Counterfactual:
    def __init__(self, model, debug = True):
        self.model = model
        self.debug = debug
    
    def get_mse(self, a, b):
        return ((a-b)**2).mean(axis = (1,2,3))
        
    def explain(
        self,
        normal_samples,
        anomalous_samples,
        threshold_pct = 0.98, 
        block_size = 32,
        anomaly_type = "",
        save_path = None,
        save_results = False
    ):
        results = dict()
        
        # Check for datatype
        if not isinstance(normal_samples, np.ndarray): normal_samples = np.array(normal_samples)
        if not isinstance(anomalous_samples, np.ndarray): anomalous_samples = np.array(anomalous_samples)
        
        # Reconstructions and error calculations can be reduced to being done once
        normal_reconstructions = self.model.predict(normal_samples)[0]
        normal_mse = self.get_mse(normal_samples, normal_reconstructions)
        
        anomalous_reconstructions = self.model.predict(anomalous_samples)[0]
        anomalous_mse = self.get_mse(anomalous_samples, anomalous_reconstructions)
        
        if self.debug: print("Models loaded. Starting analysis")
            
        # Run the counterfactual part
        for norm_idx, (norm_image, norm_reconstruction, norm_err) in tqdm(enumerate(zip(normal_samples, normal_reconstructions, normal_mse))):
            results[norm_idx] = dict()
            for anomaly_idx, (anomaly_image, anomaly_reconstruction, anomaly_err) in enumerate(zip(anomalous_samples, anomalous_reconstructions, anomalous_mse)):
                if norm_err < anomaly_err:
                    threshold = anomaly_err * threshold_pct
                    edits_made = minimum_edits(norm_image, anomaly_image, threshold, block_size, self.model)
                    masked_result = draw_boundaries(edits_made, anomaly_image)
                    results[norm_idx][anomaly_idx] = masked_result
                    if save_results: join_paths([save_path, "%s_N-%d_A-%d.png"%(anomaly_type, norm_idx, anomaly_idx)])
        return results