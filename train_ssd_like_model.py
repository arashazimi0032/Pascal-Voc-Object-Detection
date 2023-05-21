from Utils import *
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from Models import SSDLikeModel
from CustomLoss import ssd_loss

model_check_point = ModelCheckpoint(filepath=r'ssd_like_model.h5',
                                    save_best_only=False,
                                    verbose=1,
                                    monitor='val_loss')

xml_path = r'D:\E\freelancer projects\Advance-Computer-Vision\Data\VOC2012\Annotations'
train_folder_path = r'D:\E\freelancer projects\Advance-Computer-Vision\Data\VOC2012\small\train'
validation_folder_path = r'D:\E\freelancer projects\Advance-Computer-Vision\Data\VOC2012\small\test'
pre_parsed_csv_path = r'./Utils/data_details.csv'

batch_size = 32

ssd_like_model = SSDLikeModel(train_folder_path, validation_folder_path, batch_size,
                              pre_parsed_csv_path=pre_parsed_csv_path)

optimizer = SGD(learning_rate=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_like_model.compile(optimizer=optimizer, loss=ssd_loss)

ssd_like_model.summary()

ssd_like_model.fit(epochs=300, batch_size=batch_size,
                   callbacks=[LearningRateScheduler(scheduler_ssd), model_check_point]
                   )
