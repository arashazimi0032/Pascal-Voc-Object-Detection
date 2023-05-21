import numpy as np
from keras.models import Model, load_model
from keras import layers as lay
from CustomLayers import BooleanMask
from CustomDataGenerators import create_data_generator
from Utils import load_input_image, box_detection, stack_data, generate_anchor_boxes, annotation_parser
import os


class InnovativeModel:
    def __init__(self, train_folder_path, validation_folder_path, batch_size, pre_parsed_csv_path='',
                 xml_path='', segmented_data_path='', save_parsed_xml_to_csv_path=''):

        vgg_model = load_model(os.path.join(os.path.split(__file__)[0], 'vgg_model_256.h5'))

        self.img_size = (256, 256)

        self.batch_size = batch_size

        self.pre_parsed_csv_path = pre_parsed_csv_path
        self.xml_path = xml_path
        self.segmented_data_path = segmented_data_path
        self.save_parsed_xml_to_csv_path = save_parsed_xml_to_csv_path

        self.train_folder_path = train_folder_path
        self.validation_folder_path = validation_folder_path
        self.data_details = None

        self.num_anchor_pre_pixel = 5

        self.num_classes = 21

        self.anchors = np.empty([0, 4])

        self.anchors = InnovativeModel.get_innovative_anchors(self.img_size)

        num_total_anchors = np.shape(self.anchors)[1]
        self.history = None
        self.all_train = None
        self.all_test = None

        if type(vgg_model) == str:
            vgg_model = load_model(vgg_model)

        vgg_model.trainable = False
        vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.get_layer('block4_conv3').output)

        mask_input = lay.Input(shape=(num_total_anchors * 4,))
        inputs = lay.Input(shape=self.img_size + (3,))
        vgg_output = vgg_model(inputs)

        offset_output = lay.Conv2D(64, 5, padding='same')(vgg_output)
        offset_output = lay.BatchNormalization()(offset_output)
        offset_output = lay.LeakyReLU(alpha=0.1)(offset_output)
        offset_output = lay.Conv2D(64, 3, padding='same')(offset_output)
        offset_output = lay.BatchNormalization()(offset_output)
        offset_output = lay.LeakyReLU(alpha=0.1)(offset_output)
        offset_output = lay.Conv2D(self.num_anchor_pre_pixel * 4, 3, padding='same')(offset_output)
        offset_output = lay.BatchNormalization()(offset_output)
        offset_output = lay.Flatten(name='flatten')(offset_output)
        offset_output = BooleanMask(name='offsets')([offset_output, mask_input])

        class_output = lay.Conv2D(128, 5, activation=lay.LeakyReLU(alpha=0.1), padding='same')(vgg_output)
        class_output = lay.Conv2D(64, 3, activation=lay.LeakyReLU(alpha=0.1), padding='same')(class_output)
        class_output = lay.Conv2D(self.num_anchor_pre_pixel * self.num_classes, 3, activation=lay.LeakyReLU(alpha=0.3),
                                  padding='same')(class_output)
        class_output = lay.Reshape(target_shape=(-1, self.num_classes))(class_output)
        class_output = lay.Softmax(axis=-1, name='classes')(class_output)

        self.model = Model(inputs=[inputs, mask_input], outputs=[offset_output, class_output])

    def data_loader(self):
        self.data_details, _ = annotation_parser(self.img_size, pre_parsed_csv_path=self.pre_parsed_csv_path,
                                                 xml_path=self.xml_path, segmented_data_path=self.segmented_data_path,
                                                 save_parsed_xml_to_csv_path=self.save_parsed_xml_to_csv_path)

        self.all_train = create_data_generator(self.train_folder_path, self.data_details.copy(),
                                               self.anchors, self.num_classes, self.img_size, self.batch_size)
        self.all_test = create_data_generator(self.validation_folder_path, self.data_details.copy(),
                                              self.anchors, self.num_classes, self.img_size, self.batch_size)

    @staticmethod
    def load_model(path_to_model):
        return load_model(path_to_model, custom_objects={'BooleanMask': BooleanMask})

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def summary(self):
        self.model.summary()

    def fit(self, **kwargs):
        self.data_loader()
        self.model.fit(self.all_train, validation_data=self.all_test, **kwargs)
        self.history = self.model.history

    def predict(self, image, **kwargs):
        return InnovativeModel.predict_from_load_model(image, self.model,
                                                       pre_parsed_csv_path=self.pre_parsed_csv_path,
                                                       xml_path=self.xml_path,
                                                       segmented_data_path=self.segmented_data_path,
                                                       save_parsed_xml_to_csv_path=self.save_parsed_xml_to_csv_path,
                                                       **kwargs)

    @staticmethod
    def predict_from_load_model(image, model, pre_parsed_csv_path='', xml_path='', segmented_data_path='',
                                save_parsed_xml_to_csv_path='', **kwargs):
        img_size = (256, 256)
        anchors = InnovativeModel.get_innovative_anchors(img_size)

        data_details, _ = annotation_parser(img_size, pre_parsed_csv_path=pre_parsed_csv_path, xml_path=xml_path,
                                            segmented_data_path=segmented_data_path,
                                            save_parsed_xml_to_csv_path=save_parsed_xml_to_csv_path)

        if type(model) == str:
            model = InnovativeModel.load_model(model)

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

        prediction_model = InnovativeModel.create_prediction_model(model)

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
        return Model(inputs=model.inputs[0],
                     outputs=[model.get_layer('flatten').output, model.outputs[1]])

    @staticmethod
    def get_innovative_anchors(img_size):
        scales = [0.1, 0.2, 0.37, 0.54, 0.63, 0.71, 0.88]
        aspect_ratios = [[1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0, 1.0 / 4.0],
                         [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0, 1.0 / 4.0],
                         [1.0, 2.0, 0.5, 1.0 / 3.0],
                         [1.0, 2.0, 0.5, 1.0 / 3.0],
                         [1.0, 2.0, 0.5],
                         [1.0, 2.0, 0.5]]
        steps = [8, 16, 32, 64, 96, 100, 300]
        anchors = np.empty([0, 4])
        for i in range(len(scales)):
            anchor_sizes = [scales[i]]
            anchor_ratios = aspect_ratios[i]
            anchor_stride = steps[i]
            anchors = np.vstack(
                [anchors, generate_anchor_boxes(img_size, anchor_sizes, anchor_ratios, stride=anchor_stride)[0]])
        anchors = np.expand_dims(anchors, 0)
        return anchors[[0], :5120]
