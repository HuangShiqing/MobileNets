import cv2
import numpy as np
from tqdm import tqdm
from varible import *
import os
import sys
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def read_xml(ANN, pick, exclusive=False):
    print('Parsing for {} {}'.format(
        pick, 'exclusively' * int(exclusive)))

    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(ANN)
    annotations = os.listdir('.')
    # annotations = glob.glob(str(annotations) + '*.xml')
    size = len(annotations)

    for i, file in enumerate(annotations):
        # progress bar
        sys.stdout.write('\r')
        percentage = 1. * (i + 1) / size
        progress = int(percentage * 20)
        bar_arg = [progress * '=', ' ' * (19 - progress), percentage * 100]
        bar_arg += [file]
        sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
        sys.stdout.flush()

        # actual parsing
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        jpg = str(root.find('filename').text) + '.jpg'
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)
        all = list()

        for obj in root.iter('object'):
            # current = list()
            current = dict()
            name = obj.find('name').text
            if name not in pick:
                continue

            xmlbox = obj.find('bndbox')
            xn = int(float(xmlbox.find('xmin').text))
            xx = int(float(xmlbox.find('xmax').text))
            yn = int(float(xmlbox.find('ymin').text))
            yx = int(float(xmlbox.find('ymax').text))
            # current = [name, xn, yn, xx, yx]
            current['name'] = name
            current['xmin'] = xn
            current['xmax'] = xx
            current['ymin'] = yn
            current['ymax'] = yx
            all += [current]

        add = [[jpg, [w, h, all]]]
        if len(all) is not 0:  # skip the image which not include any 'pick'
            dumps += add
        in_file.close()

    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current['name'] in pick:
                if current['name'] in stat:
                    stat[current['name']] += 1
                else:
                    stat[current['name']] = 1

    print('\nStatistics:')
    for i in stat: print('{}: {}'.format(i, stat[i]))
    print('Dataset size: {}'.format(len(dumps)))

    os.chdir(cur_dir)
    return dumps


def read_txt(txt_path, label_dir):
    """
        从trainval.txt中读取参与训练的图片路径，并根据路径读取包含box信息的文件，获得包含label信息的chunks

        Parameters
        ----------
        txt_path : str
        label_dir : str

        Returns
        -------
        chunks : list

        Examples
        --------
        txt_path = 'D:/DeepLearning/data2/DRBox/Ship-Opt/trainval.txt'
        label_dir = 'D:/DeepLearning/data2/DRBox/Ship-Opt/train_data/'
        chunks =[['ZHUJIANG2_Level_19.tif_res_0.71_207_flipud.tif', [['110.644815', '179.701814', '80.667319', '14.789432', '1', '82.000000']]]，
                 ['SHANGHAI1_Level_19_2.tif_res_1_605_rotate270.tif', [['141.800141', '275.244725', '41.258141', '8.500000', '1', '164.000000'], ['138.181806', '283.516839', '41.758240', '9.250000', '1', '164.000000']]]]

    """
    chunks = list()
    with open(txt_path, 'r') as fh:
        for line in tqdm(fh):
            img_name = line.split(' ')[0]
            label_name = line.split(' ')[1].strip()
            boxes = list()
            with open(label_dir + label_name, 'r') as fh2:
                for line2 in fh2:
                    boxes.append(line2.strip().split(' '))

            chunks.append([img_name, boxes])
    return chunks


def random_flip(image, flip):
    if flip == 1:
        return cv2.flip(image, 1)
    return image


def flip_boxes(boxes, flip):
    if flip == 1:
        for box in boxes:
            # box[0] = 416 - box[0]
            # box[5] = 180 - box[5]
            # box[1] = 416 - box[1]
            swap = box['xmin']
            box['xmin'] = 416 - box['xmax']
            box['xmax'] = 416 - swap
    return boxes


def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
    def _rand_scale(scale):
        scale = np.random.uniform(1, scale)
        return scale if (np.random.randint(2) == 0) else 1. / scale

    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation)
    dexp = _rand_scale(exposure)
    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')
    # change satuation and exposure
    image[:, :, 1] *= dsat
    image[:, :, 2] *= dexp
    # change hue
    image[:, :, 0] += dhue
    image[:, :, 0] -= (image[:, :, 0] > 180) * 180
    image[:, :, 0] += (image[:, :, 0] < 0) * 180
    # avoid overflow when astype('uint8')
    image[...] = np.clip(image[...], 0, 255)
    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)


def resize_img(img):
    """
        保持长宽比缩放图像到416*416大小，空余部分填128

        Parameters
        ----------
        img : np.array  [h,w,3]

        Returns
        -------
        im_sized : np.array  [416,416,3]

        Examples
        --------
    """
    img_w = img.shape[1]
    img_h = img.shape[0]

    ratio = img_w / img_h
    net_w, net_h = 416, 416
    if ratio < 1:
        new_h = int(net_h)
        new_w = int(net_h * ratio)
    else:
        new_w = int(net_w)
        new_h = int(net_w / ratio)
    im_sized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    dx = net_w - new_w
    dy = net_h - new_h

    if dx > 0:
        im_sized = np.pad(im_sized, ((0, 0), (int(dx / 2), 0), (0, 0)), mode='constant', constant_values=128)
        im_sized = np.pad(im_sized, ((0, 0), (0, dx - int(dx / 2)), (0, 0)), mode='constant', constant_values=128)
    else:
        im_sized = im_sized[:, -dx:, :]
    if dy > 0:
        im_sized = np.pad(im_sized, ((int(dy / 2), 0), (0, 0), (0, 0)), mode='constant', constant_values=128)
        im_sized = np.pad(im_sized, ((0, dy - int(dy / 2)), (0, 0), (0, 0)), mode='constant', constant_values=128)
    else:
        im_sized = im_sized[-dy:, :, :]
    return im_sized


def resize_boxes(labels, img_w, net_w):
    new_labels = list()
    for label in labels[2]:
        # new_x = float(label[0]) * net_w / img_w
        # new_y = float(label[1]) * net_w / img_w
        # new_w = float(label[2]) * net_w / img_w
        # new_h = float(label[3]) * net_w / img_w
        # new_labels.append([new_x, new_y, new_w, new_h, 1, float(label[5])])

        new_xmin = int(label['xmin'] * net_w / img_w)
        new_xmax = int(label['xmax'] * net_w / img_w)
        new_ymin = int(label['ymin'] * net_w / img_w)
        new_ymax = int(label['ymax'] * net_w / img_w)
        new_labels.append(
            {'xmin': new_xmin, 'xmax': new_xmax, 'ymin': new_ymin, 'ymax': new_ymax, 'name': label['name']})
    return new_labels


def get_data(chunk, img_dir):
    img = cv2.imread(img_dir + chunk[0])
    img = img[:, :, ::-1]  # RGB image
    img = resize_img(img)
    boxes = resize_boxes(chunk[1], 1598, 416)

    img = random_distort_image(img)

    flip = np.random.randint(2)
    img = random_flip(img, flip)
    boxes = flip_boxes(boxes, flip)

    # visualization2(img, boxes)
    return img, boxes


def bbox_iou(box1, box2):
    def _interval_overlap(interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b

        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    intersect_w = _interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
    intersect_h = _interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])
    intersect = intersect_w * intersect_h
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def get_y_true(boxes):
    batch_size = Gb_batch_size
    net_w = net_h = 416
    anchors = Gb_anchors
    anchors_BoundBox = [[0, 0, anchors[2 * i], anchors[2 * i + 1]] for i in range(len(anchors) // 2)]
    labels = Gb_label

    # initialize the inputs and the outputs
    y_true = np.zeros(
        (batch_size, Gb_cell, Gb_cell, 9, 4 + 1 + len(Gb_label)))  # desired network output 3

    for instance_index in range(batch_size):
        # allobj_sized = [{'xmin': 96, 'name': 'person', 'ymin': 96, 'xmax': 304, 'ymax': 304},
        #                 {'xmin': 329, 'name': 'person', 'ymin': 272, 'xmax': 337, 'ymax': 337}]
        allobj_sized = boxes[instance_index]
        for obj in allobj_sized:
            # find the best anchor box for this object
            max_anchor = None
            max_index = -1
            max_iou = -1

            shifted_box = [0, 0, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin']]
            for i in range(len(anchors_BoundBox)):
                anchor = anchors_BoundBox[i]
                iou = bbox_iou(shifted_box, anchor)
                if max_iou < iou:
                    max_anchor = anchor
                    max_index = i
                    max_iou = iou

                    # determine the yolo to be responsible for this bounding box
            grid_h, grid_w = y_true.shape[1:3]

            # determine the position of the bounding box on the grid
            center_x = .5 * (obj['xmin'] + obj['xmax'])
            center_x = center_x / float(net_w)  # * grid_w  # sigma(t_x) + c_x
            center_y = .5 * (obj['ymin'] + obj['ymax'])
            center_y = center_y / float(net_h)  # * grid_h  # sigma(t_y) + c_y

            # determine the sizes of the bounding box
            # w = np.log((obj['xmax'] - obj['xmin']) / float(max_anchor.xmax))  # t_w
            # h = np.log((obj['ymax'] - obj['ymin']) / float(max_anchor.ymax))  # t_h
            w = (obj['xmax'] - obj['xmin']) / float(net_w)  # t_w
            h = (obj['ymax'] - obj['ymin']) / float(net_h)  # t_h
            box = [center_x, center_y, w, h]

            # determine the index of the label
            obj_indx = labels.index(obj['name'])
            # determine the location of the cell responsible for this object
            grid_x = int(np.floor(center_x * grid_w))
            grid_y = int(np.floor(center_y * grid_h))
            # assign ground truth x, y, w, h, confidence and class probs to y_batch
            y_true[instance_index, grid_y, grid_x, max_index] = 0
            y_true[instance_index, grid_y, grid_x, max_index, 0:4] = box
            y_true[instance_index, grid_y, grid_x, max_index, 4] = 1.
            y_true[instance_index, grid_y, grid_x, max_index, 5 + obj_indx] = 1

    return y_true


def data_generator(chunks):
    img_dir = Gb_img_dir
    batch_size = Gb_batch_size

    n = len(chunks)
    i = 0
    count = 0
    while count < (n / batch_size):
        images_data = []
        boxes_data = []
        while len(boxes_data) < batch_size:
            # for t in range(batch_size):
            i %= n
            img_sized, box_sized = get_data(chunks[i], img_dir)
            i += 1
            # plt.cla()
            # plt.imshow(img_sized)
            # for obj in box_sized:
            #     x1 = obj['xmin']
            #     x2 = obj['xmax']
            #     y1 = obj['ymin']
            #     y2 = obj['ymax']
            #
            #     plt.hlines(y1, x1, x2, colors='red')
            #     plt.hlines(y2, x1, x2, colors='red')
            #     plt.vlines(x1, y1, y2, colors='red')
            #     plt.vlines(x2, y1, y2, colors='red')
            # plt.show()
            # if len(boxes_sized) is 0:  # in case all the box in a batch become empty becase of the augmentation
            #     continue
            images_data.append(img_sized)
            boxes_data.append(box_sized)

        y_true = get_y_true(boxes_data)

        images_data = np.array(images_data)
        images_data = images_data / 255.
        # boxes_labeled = np.array(boxes_labeled)
        yield images_data, y_true
        count += 1


if __name__ == '__main__':
    cur_dir = os.getcwd()
    os.chdir(Gb_img_dir)
    annotations = os.listdir('.')
    size = len(annotations)
    os.chdir(cur_dir)

    chunks = read_xml(Gb_label_dir, Gb_label)
    img_with_box = [chunk[0] for chunk in chunks]

    a = data_generator(chunks)
    for x in a:
        print('ok')
    exit()
