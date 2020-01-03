import os
from pathlib import Path
from model import unet
from data import (trainGenerator, testGenerator, saveResult)
from keras.callbacks import ModelCheckpoint, Callback

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# ========== Saving GPU memory ========== #
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   
session = tf.Session(config=config)
KTF.set_session(session)
# ======================================= #

# ========== Get GPU numbers ========== #
from tensorflow.python.client import device_lib

gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
gpu_num= len(gpus)
# ===================================== #

train_path='data/membrane/train'
test_path='data/membrane/test'
predict_path='data/membrane/predict'
checkpoint_path='./unet_membrane.hdf5'

#aug_path='data/membrane/aug'
aug_path=None
batch_one_gpu=2

Path(predict_path).mkdir(parents=True, exist_ok=True)
if aug_path:
    Path(aug_path).mkdir(parents=True, exist_ok=True)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

model = unet(gpus=gpu_num)

class TrainLogger(Callback):
    def on_batch_end(self, batch, logs={}):
        print("Train step={} loss={} acc={}".format(batch, logs.get('loss'), logs.get('accuracy')))

myGene = trainGenerator(batch_one_gpu,train_path,'image','label',data_gen_args,save_to_dir = aug_path)
model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=400,epochs=1,callbacks=[model_checkpoint, TrainLogger()], verbose=0)

testGene = testGenerator(test_path)
results = model.predict_generator(testGene,30,verbose=1)
saveResult(predict_path,results)
