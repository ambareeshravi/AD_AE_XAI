import shap
import numpy as np
from utils import *

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Created a color map for visualization
colors = []
for l in np.linspace(1, 0, 100):
    colors.append((230./255, 236./255, 229./255,l))
for l in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,l))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)
cmap_rtb = plt.get_cmap(red_transparent_blue)

class SHAP_Explainer:
    '''
    Class container for SHAP based explanation on images
    
    https://github.com/slundberg/shap/blob/master/
    '''
    def __init__(
        self,
        blend_alpha = 0.85
    ):
        '''
        Different datasets need different alpha values for blending for better visualization
        '''
        self.blend_alpha = blend_alpha
    
    def explain(
        self,
        model,
        X_train,
        X_test,
        background_samples = 100
    ):
        '''
        Args:
            model - <keras.Model> that returns probabilities with weights loaded -> get_model(rec = False)
            X_train - <np.ndarray> - normal (train) images of shape N,W,H,C for setting background
            X_test - <np.ndarray> - test images of shape N,W,H,C for testing
            background_samples - <int> - number of images to take expectations over
        Returns:
            outputs for visualization as <np.ndarray> N,W,H,C
        '''
        # set of background examples to take an expectation over
        background = X_train[np.random.choice(X_train.shape[0], background_samples, replace=False)]
        # explain predictions of the model on four images
        explainter = shap.DeepExplainer(model, background)
        # get shap values
        shap_values = explainter.shap_values(X_test)
        # get visualization with overlayed annotations
        return self.get_shap_vis(X_test, shap_values)
    
    def get_shap_vis(
        self,
        images,
        shap_values,
    ):
        '''
        Args:
            images - test images as <np.ndarray>
            shap_values as <np.ndarray>
        Returns:
            annotated images as <np.ndarray> of shape N,W,H,C
        '''
        assert len(images) == len(shap_values[0]), "Number of images and shap values should be same"
        
        vis_results = list()
        
        for idx, (image, shap) in enumerate(zip(images, shap_values[0])):
            img = image.copy()

            # reduce dim for grayscale
            if (len(img.shape) == 3) and (img.shape[-1] == 1): img = img.reshape(img.shape[:2])
            # normalize image
            if (img.max() > 1): img /= 255.
            # convert to gray scale - some special formula
            if (len(img.shape) == 3) and (img.shape[-1] == 3):
                img_gray = (0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]) # rgb to gray
            else:
                img_gray = img

            if len(shap.shape) == 2:
                abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
            else:
                abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()

            max_val = np.nanpercentile(abs_vals, 99.9)
            
            # Reduce to 2D W,H
            sv = shap if len(shap.shape) == 2 else shap.sum(-1)
            # Clip for better visualization
            scaled_sv = np.clip(sv, -max_val, max_val)
            
            # Blend images
            pil_gray = Image.fromarray(im_to_255(img)).convert("RGBA")
            pil_sv = Image.fromarray(im_to_255(cmap_rtb(normalize(scaled_sv))))
            blended = np.array(Image.blend(pil_gray, pil_sv, self.blend_alpha).convert("RGB"))
            vis_results.append(blended)
        return np.array(vis_results)