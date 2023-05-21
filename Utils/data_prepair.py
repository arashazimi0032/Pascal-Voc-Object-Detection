from keras.utils import load_img, img_to_array
import xml.etree.ElementTree as Et
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def read_xml_details(xml_path, segmented_data_path=''):
    xml_to_read = find_data_to_read(xml_path, segmented_data_path, data_format='xml')

    data_details = []
    for file_name in xml_to_read:

        tree = Et.parse(os.path.join(xml_path, file_name))
        root = tree.getroot()

        width = eval(root.findall('size')[0].findall('width')[0].text)
        height = eval(root.findall('size')[0].findall('height')[0].text)

        objects = root.findall('object')

        for o in objects:
            row_details = {}

            object_class = o.findall('name')[0].text

            bbox = o.findall('bndbox')[0]

            row_details['file name'] = file_name.split('.')[0]
            row_details['n objects'] = len(objects)
            row_details['object class'] = object_class

            row_details['width'] = width
            row_details['height'] = height

            row_details['xmin'] = eval(bbox.find('xmin').text)
            row_details['ymin'] = eval(bbox.find('ymin').text)
            row_details['xmax'] = eval(bbox.find('xmax').text)
            row_details['ymax'] = eval(bbox.find('ymax').text)

            data_details.append(row_details)

    return pd.DataFrame(data_details)


def find_data_to_read(data_path, segmented_data_path='', data_format=''):
    if data_format == '':
        data_format = find_data_format(os.listdir(data_path)[0])

    if segmented_data_path == '':
        file_to_read = sorted(os.listdir(data_path))
    else:
        file_to_read = [p for p in map(lambda x: x.split('.')[0] + '.' + data_format,
                                       sorted(os.listdir(segmented_data_path)))]
    return file_to_read


def find_data_format(data):
    return data.split('.')[-1]


def load_input_image(path, img_size=None, **kwargs):
    return img_to_array(load_img(path, target_size=img_size, **kwargs))/255.


def stack_data(image_path, segmented_data_path='', img_size=None, num_to_read=None, pass_through_model=None,
               batch_size=64, **kwargs):
    if type(segmented_data_path) != str:
        img_to_load = [n.split('.')[0] + '.jpg' for n in segmented_data_path]
    else:
        img_to_load = find_data_to_read(image_path, segmented_data_path, data_format='jpg')

    images = []
    model_features = []

    if img_size is not None:
        for it, img_name in enumerate(img_to_load):
            if num_to_read is not None:
                if it == num_to_read:
                    break

            images.append(load_input_image(os.path.join(image_path, img_name), img_size=img_size, **kwargs))

            if pass_through_model is not None:
                if len(images) == batch_size:
                    model_features.extend(pass_through_model.predict(np.array(images)))
                    images = []

            if it % 200 == 0:
                print(it)

        if pass_through_model is not None:
            if len(images) > 0:
                model_features.extend(pass_through_model.predict(np.array(images)))

            images = np.array(model_features)

        images = np.array(images)

    else:
        it = 0
        for img_name in img_to_load:
            if num_to_read is not None:
                if it == num_to_read:
                    break

            images.append(load_input_image(os.path.join(image_path, img_name), img_size=img_size, **kwargs))

            if it % 200 == 0:
                print(it)
            it += 1

    return images


def create_bbox(bbox):
    """
    :param bbox: [x_min, y_min, x_max, y_max]
    :return:
    """

    x_min, y_min, x_max, y_max = bbox
    corners_x, corners_y = [x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min]
    return corners_x, corners_y


def bbox_shape_modify(bbox):
    if len(np.shape(bbox)) == 1:
        bbox = np.reshape(bbox, (1, -1))
    else:
        return bbox
    return bbox.tolist()


def get_image_rows_from_data_details(data_details, image_name):
    return data_details[data_details['file name'] == image_name]


def get_bbox_from_data_details(data_details):
    return data_details[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()


def draw_bbox_on_image(image, bbox, object_class=(), colors=()):
    if len(colors) == 0:
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    bbox = bbox_shape_modify(bbox)
    plt.imshow(image.astype(np.uint8))
    for it, b in enumerate(bbox):
        corners_x, corners_y = create_bbox(b)
        plt.plot(corners_x, corners_y, linewidth=1, color=colors[it % len(colors)])
        if len(object_class) > 0:
            plt.text(b[0], b[1], object_class[it], fontsize=8, fontdict={'color': 'w'},
                     bbox=dict(facecolor=colors[it % len(colors)], lw=0), va='center', ha='center')


def draw_all_image_bboxes(image, image_name, data_details, object_class=()):
    image_data = get_image_rows_from_data_details(data_details, image_name)
    bbox = get_bbox_from_data_details(image_data)
    draw_bbox_on_image(image, bbox, object_class=object_class)


def draw_all_image_bboxes_from_path(image_path, data_details, img_size=None, object_class=()):
    image = load_input_image(image_path, img_size)

    if len(image_path.split('/')) == 1:
        image_name = image_path.split('\\')[-1].split('.')[0]
    else:
        image_name = image_path.split('/')[-1].split('.')[0]

    draw_all_image_bboxes(image, image_name, data_details, object_class=object_class)


def resize_bboxes_from_data_frame(data_details, img_size=None):
    if img_size is not None:
        bbox = data_details[['xmin', 'ymin', 'xmax', 'ymax']].values
        width = data_details['width'].values
        height = data_details['height'].values

        bbox = resize_bboxes(bbox, height, width, img_size)

        data_details['xmin'] = bbox[:, 0]
        data_details['ymin'] = bbox[:, 1]
        data_details['xmax'] = bbox[:, 2]
        data_details['ymax'] = bbox[:, 3]
    return data_details


def resize_bboxes(bbox, height, width, img_size=None):
    """
    :param bbox:   [[x_min_1, y_min_1, x_max_1, y_max_1],
                    [x_min_2, y_min_2, x_max_2, y_max_2],
                    ...
                    ...
                    ...
                    [x_min_n, y_min_n, x_max_n, y_max_n]]

    :param height: [h_1, h_2, ... h_n]
    :param width: [w_1, w_2, ... w_n]
    :param img_size: (height, width)
    :return:
    """

    if img_size is not None:
        bbox = np.array(bbox)
        width = np.array(width)
        height = np.array(height)

        new_x_min = bbox[:, None, 0] * img_size[1] / width[:, None]
        new_x_max = bbox[:, None, 2] * img_size[1] / width[:, None]
        new_y_min = bbox[:, None, 1] * img_size[0] / height[:, None]
        new_y_max = bbox[:, None, 3] * img_size[0] / height[:, None]
        bbox = np.concatenate([new_x_min, new_y_min, new_x_max, new_y_max], axis=1)

    return bbox


def rescale_bboxes_from_data_frame(data_details, img_size=None):
    bbox = data_details[['xmin', 'ymin', 'xmax', 'ymax']].values
    width = img_size[1]
    height = img_size[0]

    bbox = rescale_bboxes(bbox, height, width)

    data_details['scaled_xmin'] = bbox[:, 0]
    data_details['scaled_ymin'] = bbox[:, 1]
    data_details['scaled_xmax'] = bbox[:, 2]
    data_details['scaled_ymax'] = bbox[:, 3]
    return data_details


def rescale_bboxes(bbox, height, width):
    """
    :param bbox:   [[x_min_1, y_min_1, x_max_1, y_max_1],
                    [x_min_2, y_min_2, x_max_2, y_max_2],
                    ...
                    ...
                    ...
                    [x_min_n, y_min_n, x_max_n, y_max_n]]

    :param height: [h_1, h_2, ... h_n]
    :param width: [w_1, w_2, ... w_n]
    :return:
    """

    new_x_min = bbox[:, None, 0] / width
    new_x_max = bbox[:, None, 2] / width
    new_y_min = bbox[:, None, 1] / height
    new_y_max = bbox[:, None, 3] / height
    bbox = np.concatenate([new_x_min, new_y_min, new_x_max, new_y_max], axis=1)
    return bbox


def mat2gray(img):
    return (img - np.min(img))/(np.max(img) - np.min(img))


def compute_box_center(bbox):
    """
    :param bbox: [[x_min_1, y_min_1, x_max_1, y_max_1],
                  [x_min_2, y_min_2, x_max_2, y_max_2],
                    ...
                    ...
                    ...
                  [x_min_n, y_min_n, x_max_n, y_max_n]]
    :return:
    """

    x_c, y_c = np.mean([bbox[:, 0], bbox[:, 2]], axis=0), np.mean([bbox[:, 1], bbox[:, 3]], axis=0)
    return x_c, y_c


def compute_box_width_height(bbox):
    """
    :param bbox: [[x_min_1, y_min_1, x_max_1, y_max_1],
                  [x_min_2, y_min_2, x_max_2, y_max_2],
                    ...
                    ...
                    ...
                  [x_min_n, y_min_n, x_max_n, y_max_n]]
    :return:
    """

    width, height = np.abs(bbox[:, 2] - bbox[:, 0]), np.abs(bbox[:, 3] - bbox[:, 1])
    return width, height


def compute_bbox_area(bbox):
    """
    :param bbox: [[x_min_1, y_min_1, x_max_1, y_max_1],
                  [x_min_2, y_min_2, x_max_2, y_max_2],
                    ...
                    ...
                    ...
                  [x_min_n, y_min_n, x_max_n, y_max_n]]
    :return:
    """
    width, height = compute_box_width_height(bbox)
    return width * height


def annotation_parser(img_size, pre_parsed_csv_path='', xml_path='', segmented_data_path='',
                      save_parsed_xml_to_csv_path=''):
    if pre_parsed_csv_path != '':
        data_details = pd.read_csv(pre_parsed_csv_path)
    elif xml_path != '':
        data_details = read_xml_details(xml_path, segmented_data_path=segmented_data_path)
        if save_parsed_xml_to_csv_path != '':
            data_details.set_index('file name').to_csv(save_parsed_xml_to_csv_path)
    else:
        raise Exception('please enter either xml_path to parse it or pre_parsed_csv_path to load it!')

    unique_file_names = list(data_details.groupby('file name').indices.keys())

    data_details = resize_bboxes_from_data_frame(data_details, img_size=img_size)
    data_details = rescale_bboxes_from_data_frame(data_details, img_size=img_size)

    class_unique_labels = dict(
        zip(sorted(data_details['object class'].unique()), np.arange(data_details['object class'].nunique())))
    data_details['label'] = data_details['object class'].map(class_unique_labels)
    return data_details, unique_file_names


def draw_predictions(predictions, img, data_details, img_size, conf_threshold=None):
    if conf_threshold is None:
        conf_threshold = np.mean(predictions[predictions[:, 1] > 0.3, 1])

    if type(img) == str:
        img = load_input_image(img, img_size)

    img = (img * 255).astype(np.uint8)

    highest = predictions[predictions[:, 1] >= conf_threshold]
    if len(highest) == 0:
        highest = predictions[[0], :]

    box = highest[:, 2:] * np.repeat(np.repeat(np.array(np.expand_dims(np.flip(img_size), 0)), 2,
                                               0).flatten().reshape(1, -1), len(highest), axis=0)

    classes = []
    for j, b in enumerate(highest):
        classes.append(data_details['object class'][data_details['label'] == b[0]].iloc[0] + ' ' +
                       str(np.round((predictions[j, 1]) * 100, 3)))

    draw_bbox_on_image(img, box, classes)
