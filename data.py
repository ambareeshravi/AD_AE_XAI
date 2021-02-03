import numpy as np
from sklearn.model_selection import train_test_split
from utils import *

class HAM10000:
    def __init__(
        self,
        data_path = "../datasets/VAD_Datasets/HAM10000_SPLIT/",
        batch_size = 64,
        val_split = 0.1,
        image_size = 128,
        isTrain = True,
        NORMAL_LABEL  = 0,
        useAllTestData = False,
        size = 100,
        random_state = 42,
        deriving = False
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.isTrain = isTrain
        self.NORMAL_LABEL = NORMAL_LABEL
        self.ABNORMAL_LABEL = np.abs(1 - self.NORMAL_LABEL)
        self.useAllTestData = useAllTestData
        self.size = size
        self.random_state = random_state
        
        if self.isTrain:
            self.train_dir = join_paths([self.data_path, "NORMAL", "nv"])
            if not deriving: self.create_train_data()
        else:
            self.normal_test_dir = join_paths([self.data_path, "NORMAL_TEST", "nv"])
            self.abnormal_test_dir = join_paths([self.data_path, "ABNORMAL_TEST"])
            if not deriving: self.create_test_data()
    
    def create_train_data(self,):
        self.data = self.read_images(read_directory_contents(self.train_dir))
        self.train_data, self.val_data = train_test_split(self.data, test_size = self.val_split, random_state = self.random_state)
        self.batch_read_count = 0
        self.firstVal = True
    
    def create_test_data(self,):
        self.normal_test_files = read_directory_contents(self.normal_test_dir)
        self.abnormal_test_files = dict()
        for folder in read_directory_contents(self.abnormal_test_dir):
            anomaly_type = folder.split("/")[-1].lower()
            self.abnormal_test_files[anomaly_type] = read_directory_contents(join_paths([folder, anomaly_type]))

        if not self.useAllTestData:
            self.normal_test_files = np.random.choice(self.normal_test_files, self.size)

            for anomaly_type, anomalous_files in self.abnormal_test_files.items():
                self.abnormal_test_files[anomaly_type] = np.random.choice(anomalous_files, self.size)

            # read data and labels [normal, abnormal] for testing
            self.normal_data = self.read_images(self.normal_test_files)
            self.abnormal_data = dict([(anomaly_type, sef.read_images(anomalous_files)) for (anomaly_type, anomalous_files) in self.abnormal_test_files.items()])
    
    def read_images(self, files):
        return np.array([read_image(file, self.image_size)/255. for file in files])
                           
    def train_batch_generator(self):
        while True:
            start = self.batch_read_count
            end = (self.batch_read_count + self.batch_size) if (self.batch_read_count + self.batch_size) < len(self.train_data) else len(self.train_data)
            X = self.train_data[start:end]
            self.batch_read_count += self.batch_size
            if self.batch_read_count >= len(self.train_data): self.batch_read_count = 0
            yield(X, X)
            
    def val_batch_generator(self):
        while True:
            if self.firstVal:
                np.random.shuffle(self.val_data)
                self.firstVal = False
            yield(self.val_data, self.val_data)
            
class IR_DISTRACTION(HAM10000):
    def __init__(
        self,
        data_path = "../datasets/VAD_Datasets/IR_DISTRACTION/",
        batch_size = 64,
        val_split = 0.1,
        image_size = 128,
        isTrain = True,
        NORMAL_LABEL  = 0,
        useAllTestData = False,
        size = 100,
        random_state = 42
    ):
        HAM10000.__init__(self, deriving = True)
        self.data_path = data_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.isTrain = isTrain
        self.NORMAL_LABEL = NORMAL_LABEL
        self.ABNORMAL_LABEL = np.abs(1 - self.NORMAL_LABEL)
        self.useAllTestData = useAllTestData
        self.size = size
        self.random_state = random_state
        
        if self.isTrain:
            self.train_dir = join_paths([self.data_path, "NORMAL_TRAIN", "normal_train"])
            self.create_train_data()
        else:
            self.normal_test_dir = join_paths([self.data_path, "NORMAL_TEST", "normal_test"])
            self.abnormal_test_dir = join_paths([self.data_path, "ABNORMAL_TEST"])
            self.create_test_data()            