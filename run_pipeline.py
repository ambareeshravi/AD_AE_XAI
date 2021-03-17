from utils import *
from model_utils import *
from data import *
from c2d_models import *
from test_pipeline import *

model_path = "trained_models/C2D_AE_128_3x3_IR_DISTRACTION/model.h5"
train_data = IR_DISTRACTION(isTrain = True)
output_path = join_paths(["results/", train_data.__name__, ""])
create_directory(output_path)

config = {
    'lime': {
        'num_features': 128*128,
        'mask_features': 5,
        'num_samples': 1000,
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
        'max_loss_value': 1000
    },
    'output_path': output_path,
    'test_data_path': join_paths(["TEST_SETS/", train_data.__name__, ""]),
    'train_data': train_data.data, # np.array of normal images
    'max_loss_value': 1000,
    'batch_size': 64,
    'data_name': train_data.__name__,
    'model_path': model_path
}

rec_model = get_model(model_path)
loss_model = get_model(model_path, rec=False)

p = Pipeline(rec_model, loss_model, config)
p.run()