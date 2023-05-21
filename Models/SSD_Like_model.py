import numpy as np
from keras.models import Model, load_model
from keras import layers as lay
from keras.regularizers import L2
from keras.optimizers import SGD
from CustomLayers import L2Normalization
from CustomDataGenerators import create_ssd_data_generator
from CustomLoss import ssd_loss
from Utils import load_input_image, box_detection, stack_data, generate_anchor_boxes, annotation_parser
import os


class SSDLikeModel:
    def __init__(self, train_folder_path, validation_folder_path, batch_size, pre_parsed_csv_path='',
                 xml_path='', segmented_data_path='', save_parsed_xml_to_csv_path=''):
        vgg_model = load_model(os.path.join(os.path.split(__file__)[0], 'vgg_model_300.h5'))
        vgg_model.trainable = False
        vgg_model = Model(inputs=vgg_model.inputs, outputs=[vgg_model.get_layer('block5_conv3').output,
                                                            vgg_model.get_layer('block4_conv3').output])

        self.img_size = (300, 300)
        self.num_classes = 21
        self.num_anchors = 4
        self.history = None
        self.all_train = None
        self.all_test = None
        self.data_details = None
        self.pre_parsed_csv_path = pre_parsed_csv_path
        self.xml_path = xml_path
        self.segmented_data_path = segmented_data_path
        self.save_parsed_xml_to_csv_path = save_parsed_xml_to_csv_path
        self.train_folder_path = train_folder_path
        self.validation_folder_path = validation_folder_path
        self.batch_size = batch_size

        self.anchors = SSDLikeModel.get_ssd_anchors(self.img_size)

        inputs = lay.Input(shape=self.img_size + (3,))

        vgg_output, conv4_3 = vgg_model(inputs)

        pool5 = lay.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(vgg_output)

        fc6 = lay.Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same',
                         kernel_initializer='he_normal', kernel_regularizer=L2(0.0005), name='fc6')(pool5)

        fc7 = lay.Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=L2(0.0005), name='fc7')(fc6)

        conv6_1 = lay.Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                             kernel_regularizer=L2(0.0005), name='conv6_1')(fc7)

        conv6_1 = lay.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)

        conv6_2 = lay.Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                             kernel_initializer='he_normal', kernel_regularizer=L2(0.0005), name='conv6_2')(conv6_1)

        conv7_1 = lay.Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                             kernel_regularizer=L2(0.0005), name='conv7_1')(conv6_2)

        conv7_1 = lay.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)

        conv7_2 = lay.Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid',
                             kernel_initializer='he_normal', kernel_regularizer=L2(0.0005), name='conv7_2')(conv7_1)

        conv8_1 = lay.Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                             kernel_regularizer=L2(0.0005), name='conv8_1')(conv7_2)

        conv8_2 = lay.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid',
                             kernel_initializer='he_normal', kernel_regularizer=L2(0.0005), name='conv8_2')(conv8_1)

        conv9_1 = lay.Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                             kernel_regularizer=L2(0.0005), name='conv9_1')(conv8_2)

        conv9_2 = lay.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid',
                             kernel_initializer='he_normal', kernel_regularizer=L2(0.0005), name='conv9_2')(conv9_1)

        conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)

        conv4_3_norm_mbox_conf = lay.Conv2D(self.num_anchors * self.num_classes, (3, 3), padding='same',
                                            kernel_initializer='he_normal', kernel_regularizer=L2(0.0005),
                                            name='conv4_3_norm_mbox_conf')(conv4_3_norm)

        fc7_mbox_conf = lay.Conv2D(self.num_anchors * self.num_classes, (3, 3), padding='same',
                                   kernel_initializer='he_normal', kernel_regularizer=L2(0.0005),
                                   name='fc7_mbox_conf')(fc7)

        conv6_2_mbox_conf = lay.Conv2D(self.num_anchors * self.num_classes, (3, 3), padding='same',
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=L2(0.0005), name='conv6_2_mbox_conf')(conv6_2)

        conv7_2_mbox_conf = lay.Conv2D(self.num_anchors * self.num_classes, (3, 3), padding='same',
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=L2(0.0005), name='conv7_2_mbox_conf')(conv7_2)

        conv8_2_mbox_conf = lay.Conv2D(self.num_anchors * self.num_classes, (3, 3), padding='same',
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=L2(0.0005), name='conv8_2_mbox_conf')(conv8_2)

        conv9_2_mbox_conf = lay.Conv2D(self.num_anchors * self.num_classes, (3, 3), padding='same',
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=L2(0.0005), name='conv9_2_mbox_conf')(conv9_2)

        conv4_3_norm_mbox_loc = lay.Conv2D(self.num_anchors * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                           kernel_regularizer=L2(0.0005), name='conv4_3_norm_mbox_loc')(conv4_3_norm)

        fc7_mbox_loc = lay.Conv2D(self.num_anchors * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                  kernel_regularizer=L2(0.0005), name='fc7_mbox_loc')(fc7)

        conv6_2_mbox_loc = lay.Conv2D(self.num_anchors * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                      kernel_regularizer=L2(0.0005), name='conv6_2_mbox_loc')(conv6_2)

        conv7_2_mbox_loc = lay.Conv2D(self.num_anchors * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                      kernel_regularizer=L2(0.0005), name='conv7_2_mbox_loc')(conv7_2)

        conv8_2_mbox_loc = lay.Conv2D(self.num_anchors * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                      kernel_regularizer=L2(0.0005), name='conv8_2_mbox_loc')(conv8_2)

        conv9_2_mbox_loc = lay.Conv2D(self.num_anchors * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                      kernel_regularizer=L2(0.0005), name='conv9_2_mbox_loc')(conv9_2)

        conv4_3_norm_mbox_conf_reshape = lay.Reshape((-1, self.num_classes), name='conv4_3_norm_mbox_conf_reshape')(
            conv4_3_norm_mbox_conf)

        fc7_mbox_conf_reshape = lay.Reshape((-1, self.num_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)

        conv6_2_mbox_conf_reshape = lay.Reshape((-1, self.num_classes), name='conv6_2_mbox_conf_reshape')(
            conv6_2_mbox_conf)

        conv7_2_mbox_conf_reshape = lay.Reshape((-1, self.num_classes), name='conv7_2_mbox_conf_reshape')(
            conv7_2_mbox_conf)

        conv8_2_mbox_conf_reshape = lay.Reshape((-1, self.num_classes), name='conv8_2_mbox_conf_reshape')(
            conv8_2_mbox_conf)

        conv9_2_mbox_conf_reshape = lay.Reshape((-1, self.num_classes), name='conv9_2_mbox_conf_reshape')(
            conv9_2_mbox_conf)

        conv4_3_norm_mbox_loc_reshape = lay.Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(
            conv4_3_norm_mbox_loc)

        fc7_mbox_loc_reshape = lay.Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)

        conv6_2_mbox_loc_reshape = lay.Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)

        conv7_2_mbox_loc_reshape = lay.Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)

        conv8_2_mbox_loc_reshape = lay.Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)

        conv9_2_mbox_loc_reshape = lay.Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)

        mbox_conf = lay.Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                               fc7_mbox_conf_reshape,
                                                               conv6_2_mbox_conf_reshape,
                                                               conv7_2_mbox_conf_reshape,
                                                               conv8_2_mbox_conf_reshape,
                                                               conv9_2_mbox_conf_reshape])

        mbox_loc = lay.Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                             fc7_mbox_loc_reshape,
                                                             conv6_2_mbox_loc_reshape,
                                                             conv7_2_mbox_loc_reshape,
                                                             conv8_2_mbox_loc_reshape,
                                                             conv9_2_mbox_loc_reshape])

        mbox_conf_softmax = lay.Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

        predictions = lay.Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc])

        self.model = Model(inputs=inputs, outputs=predictions)

    def data_loader(self):
        self.data_details, _ = annotation_parser(self.img_size, pre_parsed_csv_path=self.pre_parsed_csv_path,
                                                 xml_path=self.xml_path, segmented_data_path=self.segmented_data_path,
                                                 save_parsed_xml_to_csv_path=self.save_parsed_xml_to_csv_path)

        self.all_train = create_ssd_data_generator(self.train_folder_path, self.data_details.copy(), self.anchors,
                                                   self.num_classes, self.img_size, self.batch_size)
        self.all_test = create_ssd_data_generator(self.validation_folder_path, self.data_details.copy(),
                                                  self.anchors, self.num_classes, self.img_size, self.batch_size)

    @staticmethod
    def load_model(path_to_model):
        return load_model(path_to_model, custom_objects={'L2Normalization': L2Normalization, 'ssd_loss': ssd_loss})

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def summary(self):
        self.model.summary()

    def fit(self, **kwargs):
        self.data_loader()
        self.model.fit(self.all_train, validation_data=self.all_test, **kwargs)
        self.history = self.model.history

    def predict(self, image, **kwargs):
        return SSDLikeModel.predict_from_load_model(image, self.model,
                                                    pre_parsed_csv_path=self.pre_parsed_csv_path,
                                                    xml_path=self.xml_path,
                                                    segmented_data_path=self.segmented_data_path,
                                                    save_parsed_xml_to_csv_path=self.save_parsed_xml_to_csv_path,
                                                    **kwargs)

    @staticmethod
    def predict_from_load_model(image, model, pre_parsed_csv_path='', xml_path='', segmented_data_path='',
                                save_parsed_xml_to_csv_path='', **kwargs):
        img_size = (300, 300)
        anchors = SSDLikeModel.get_ssd_anchors(img_size)

        data_details, _ = annotation_parser(img_size, pre_parsed_csv_path=pre_parsed_csv_path, xml_path=xml_path,
                                            segmented_data_path=segmented_data_path,
                                            save_parsed_xml_to_csv_path=save_parsed_xml_to_csv_path)

        if type(model) == str:
            model = SSDLikeModel.load_model(model)

        if type(image) == str:
            if '.jpg' not in image and '.png' not in image:
                image = stack_data(image, img_size=img_size)
            else:
                image = np.expand_dims(load_input_image(image, img_size), axis=0)

        elif type(image) == list:
            images = []
            for i in image:
                images.append(load_input_image(i, img_size))
            image = np.array(images)

        prediction_model = SSDLikeModel.create_prediction_model(model)

        offset, cls = prediction_model.predict(image, **kwargs)

        cls_prop = cls.transpose([0, 2, 1])
        output = box_detection(cls_prop, offset, anchors)
        o = []
        for j in range(len(output)):
            idx = [i for i, row in enumerate(output[j]) if row[0] != -1]
            o.append(output[j, idx])
        return o, data_details

    @staticmethod
    def create_prediction_model(model):
        return Model(inputs=model.inputs,
                     outputs=[model.get_layer('mbox_loc').output, model.get_layer('mbox_conf_softmax').output])

    @staticmethod
    def get_ssd_anchors(img_size):
        scales = [0.1, 0.2, 0.29, 0.37, 0.54, 0.71, 0.88, 1.05]
        aspect_ratios = [[1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                         [1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5]]
        steps = [8, 16, 32, 32, 64, 100, 128, 300]
        anchors = np.empty([0, 4])
        for i in range(len(scales)):
            anchor_sizes = [scales[i]]
            anchor_ratios = aspect_ratios[i]
            anchor_stride = steps[i]
            anchors = np.vstack(
                [anchors, generate_anchor_boxes(img_size, anchor_sizes, anchor_ratios, stride=anchor_stride)[0]])
        anchors = np.expand_dims(anchors, 0)
        return anchors[[0], :7236]
