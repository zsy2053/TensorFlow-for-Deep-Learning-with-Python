import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict

def one_hot_encode(vec, vals = 10):
    n=len(vec)
    out=np.zeros((n, vals))
    out[range(n), vec]=1
    return out

class CifarHelper():
    def __init__(self):
        self.i = 0
        self.all_train_batches = [data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5]
        self.test_batch = [test_batch]
        self.training_images = None
        self.training_labels = None
        self.test_images = None
        self.test_labels = None
    def set_up_images(self):
        print("Setting up training images and labels")
        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)

        self.training_images = self.training_images.reshape(train_len,3,32,32).transpose(0,2,3,1)/255
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)

        print("Setting Up Test Images and labels")

        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)
        self.test_images = self.test_images.reshape(test_len, 3, 32, 32).transpose(0,2,3,1)/255
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), 10)

    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i + batch_size].reshape(100,32,32,3)
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y

CIFAR_DIR = 'cifar-10-batches-py'
dirs = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
all_data = [0,1,2,3,4,5,6]
for i,direc in zip(all_data, dirs):
    all_data[i] = unpickle(CIFAR_DIR+"/"+direc)
batch_meta = all_data[0]
data_batch_1 = all_data[1]
data_batch_2 = all_data[2]
data_batch_3 = all_data[3]
data_batch_4 = all_data[4]
data_batch_5 = all_data[5]
test_batch = all_data[6]

# reshape the array
X = data_batch_1[b"data"]
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
plt.imshow(X[0])
X = data_batch_1[b'data']
all_images = X.reshape(10000,3,32,32)
sample = all_images[0]
plt.imshow(sample.transpose(1,2,0))

# use functions
ch = CifarHelper()
ch.set_up_images()
batch = ch.next_batch(100)
