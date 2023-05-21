from Utils import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from Models import InnovativeModel


model_check_point = ModelCheckpoint(filepath=r'model.h5',
                                    save_best_only=True,
                                    verbose=1,
                                    monitor='val_loss')

xml_path = r'D:\E\freelancer projects\Advance-Computer-Vision\Data\VOC2012\Annotations'
train_folder_path = r'D:\E\freelancer projects\Advance-Computer-Vision\Data\VOC2012\small\train'
validation_folder_path = r'D:\E\freelancer projects\Advance-Computer-Vision\Data\VOC2012\small\test'
pre_parsed_csv_path = r'./Utils/data_details.csv'

batch_size = 32

innovative_model = InnovativeModel(train_folder_path, validation_folder_path, batch_size,
                                   pre_parsed_csv_path=pre_parsed_csv_path)

optimizer = Adam(learning_rate=0.001)

innovative_model.compile(optimizer=optimizer, loss={"offsets": "mae", "classes":  'categorical_crossentropy'})

innovative_model.summary()

innovative_model.fit(epochs=220, batch_size=batch_size,
                     callbacks=[LearningRateScheduler(innovative_model_scheduler), model_check_point]
                     )
