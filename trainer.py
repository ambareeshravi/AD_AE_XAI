from model_utils import *
from utils import *
from data import *
from c2d_models import *

from tester import *

class Trainer:
    def __init__(
        self,
        model,
        model_path,
        dataset
    ):
        self.model = model
        self.model_path = model_path
        if not os.path.exists: os.mkdir(self.model_path)
        self.dataset = dataset
        self.model_name = join_paths([self.model_path, "model.h5"])
        
    def get_optmizer(self, optimizer_type, learning_rate, epochs = None):

        if "adam" in optimizer_type.lower():
            optimizer = Adam(lr = learning_rate)
        elif "sgd" in optimizer_type.lower():
            optimizer = SGD(lr = learning_rate)
        elif "grad" in optimizer_type.lower():
            optimizer = Adagrad(lr = learning_rate, decay = learning_rate/epochs)
        elif "rms" in optimizer_type.lower():
            optimizer = RMSprop(lr = learning_rate)
        else:
            raise NotImplementedError
        return optimizer
    
    def train(
        self, 
        learning_rate = 1e-3,
        epochs = 300,
        batch_size = 64,
        optimizer_type = "adam",
        loss = "mse"
    ):
        callbacks = [
            ModelCheckpoint(self.model_name, monitor = "val_loss", verbose = 1, save_best_only = True, mode='min')
        ]
              
        optimizer = self.get_optmizer(optimizer_type, learning_rate, epochs)
        self.model.compile(loss = "mse", optimizer = optimizer)
        self.model.fit_generator(
            self.dataset.train_batch_generator(),
            validation_data = self.dataset.train_batch_generator(),
            epochs = epochs,
            steps_per_epoch = len(self.dataset.train_data) // batch_size,
            shuffle = True,
            validation_steps = 1,
            callbacks = callbacks
        )
        return self.model
    
if __name__ == "__main__":
    '''
    basic structure for now without argparser
    will edit and make it a better implementation later
    '''
    IMAGE_SIZE = 128
    CHANNELS = 3
    VAL_SPLIT  = 0.05
    EPOCHS = 300
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    DATASET = "HAM10000"
    
    dataset = HAM10000(
        batch_size = BATCH_SIZE,
        val_split = VAL_SPLIT,
        image_size = IMAGE_SIZE,
        isTrain = True
    )
    INFO("Data loaded")
    c2d_model = C2D_AE_128_3x3(input_shape = IMAGE_SIZE, channels = CHANNELS, useACB = True)
    INFO("Model ready")
    MODEL_PATH = join_paths(["dummy", "%s_%s"%(c2d_model.__name__, DATASET)])
    create_directory(MODEL_PATH)
    
    ae_trainer = Trainer(
        c2d_model.model,
        model_path = MODEL_PATH,
        dataset = dataset
    )
    INFO("Started Training")
    trained_model = ae_trainer.train(
        learning_rate = LEARNING_RATE,
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        optimizer_type = "adam",
        loss = "mse"
    )
    
    INFO("Testing the Trained Model")
    test_data = HAM10000(isTrain=False, useAllTestData=True)
    ae_tester = Tester(
        trained_model,
        test_data
    )
    results = ae_tester.test(True)