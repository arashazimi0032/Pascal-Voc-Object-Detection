from Utils import get_image_rows_from_data_details, compute_anchor_targets
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical
import numpy as np
import os


class BboxGenerator(Sequence):
    def __init__(self, data_details, anchors, num_classes, batch_size=32):
        super(BboxGenerator, self).__init__()
        self.data_details = data_details
        self.anchors = anchors
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.unique_file_names = list(self.data_details.groupby('file name').indices.keys())
        self.n = len(self.unique_file_names)

    def __getitem__(self, index):
        bbox_offsets = []
        bbox_labels = []
        bbox_masks = []
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        if end > self.n:
            end = self.n
        for i in range(start, end):
            data_slice = get_image_rows_from_data_details(self.data_details, self.unique_file_names[i])
            labels = data_slice[['label', 'scaled_xmin', 'scaled_ymin', 'scaled_xmax', 'scaled_ymax']].values
            bbox_offset, bbox_mask, bbox_label = compute_anchor_targets(self.anchors, np.expand_dims(labels, axis=0))
            bbox_offsets.extend(bbox_offset)
            bbox_masks.extend(bbox_mask)
            bbox_labels.extend(bbox_label)

        bbox_masks = np.array(bbox_masks)
        bbox_offsets = np.array(bbox_offsets)
        bbox_labels = to_categorical(np.array(bbox_labels), dtype=np.int32, num_classes=self.num_classes)
        return bbox_masks, bbox_offsets, bbox_labels

    def __len__(self):
        return int(np.round(self.n / self.batch_size))


class BboxGeneratorSSD(Sequence):
    def __init__(self, data_details, anchors, num_classes, batch_size=32):
        super(BboxGeneratorSSD, self).__init__()
        self.data_details = data_details
        self.anchors = anchors
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.unique_file_names = list(self.data_details.groupby('file name').indices.keys())
        self.n = len(self.unique_file_names)

    def __getitem__(self, index):
        bbox_offsets = []
        bbox_labels = []
        bbox_masks = []
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        if end > self.n:
            end = self.n
        for i in range(start, end):
            data_slice = get_image_rows_from_data_details(self.data_details, self.unique_file_names[i])
            labels = data_slice[['label', 'scaled_xmin', 'scaled_ymin', 'scaled_xmax', 'scaled_ymax']].values
            bbox_offset, bbox_mask, bbox_label = compute_anchor_targets(self.anchors, np.expand_dims(labels, axis=0))
            bbox_offsets.append(bbox_offset.reshape(-1, 4))
            bbox_masks.append(bbox_mask.reshape(-1, 4))
            bbox_labels.extend(bbox_label)

        bbox_masks = np.array(bbox_masks)
        bbox_offsets = np.array(bbox_offsets)
        bbox_labels = to_categorical(np.array(bbox_labels), dtype=np.int32, num_classes=self.num_classes)
        return bbox_masks, bbox_offsets, bbox_labels

    def __len__(self):
        return int(np.round(self.n / self.batch_size))


class DataGenerator(Sequence):
    def __init__(self, data_details, anchors, num_classes, image_generator, batch_size=32):
        super(DataGenerator, self).__init__()
        self.data_details = data_details
        self.anchors = anchors
        self.image_generator = image_generator
        self.batch_size = batch_size

        self.bbox_generator = BboxGenerator(data_details, anchors, num_classes, batch_size)

    def __len__(self):
        return self.image_generator.__len__()

    def __getitem__(self, index):
        bbox_mask_gen, bbox_offset_gen, bbox_label_gen = self.bbox_generator.__getitem__(index)

        return [self.image_generator.__getitem__(index), bbox_mask_gen], [bbox_offset_gen, bbox_label_gen]


class DataGeneratorSSD(Sequence):
    def __init__(self, data_details, anchors, num_classes, image_generator, batch_size=32):
        super(DataGeneratorSSD, self).__init__()
        self.data_details = data_details
        self.anchors = anchors
        self.image_generator = image_generator
        self.batch_size = batch_size

        self.bbox_generator = BboxGeneratorSSD(data_details, anchors, num_classes, batch_size)

    def __len__(self):
        return self.image_generator.__len__()

    def __getitem__(self, index):
        bbox_mask_gen, bbox_offset_gen, bbox_label_gen = self.bbox_generator.__getitem__(index)

        return self.image_generator.__getitem__(index), np.concatenate(
            [bbox_label_gen, bbox_offset_gen], axis=2)


def create_data_generator(data_folder_path, data_details, anchors, num_classes, img_size, batch_size):
    data_images_path = os.path.join(data_folder_path, 'data')
    train_generator = ImageDataGenerator(rescale=1. / 255)
    train_set = train_generator.flow_from_directory(data_folder_path,
                                                    class_mode=None,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    target_size=img_size)
    train_indices = [i.split('.')[0] for i in sorted(os.listdir(data_images_path))]
    slice_data_train = data_details.set_index('file name').loc[train_indices].reset_index()
    all_train = DataGenerator(slice_data_train, anchors, num_classes, train_set, batch_size=batch_size)
    return all_train


def create_ssd_data_generator(data_folder_path, data_details, anchors, num_classes, img_size, batch_size):
    data_images_path = os.path.join(data_folder_path, 'data')
    train_generator = ImageDataGenerator(rescale=1. / 255)
    train_set = train_generator.flow_from_directory(data_folder_path,
                                                    class_mode=None,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    target_size=img_size)
    train_indices = [i.split('.')[0] for i in sorted(os.listdir(data_images_path))]
    slice_data_train = data_details.set_index('file name').loc[train_indices].reset_index()
    all_train = DataGeneratorSSD(slice_data_train, anchors, num_classes, train_set, batch_size=batch_size)
    return all_train
