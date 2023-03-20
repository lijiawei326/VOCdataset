import torch.utils.data as data
from PIL import Image, ImageDraw
import os.path
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class TransformVOCDetectionAnnotation(object):
    """
        将读取的xml文件内容解析为：
            [[xmin, ymin, xmax, ymax, class_id],
             [xmin, ymin, xmax, ymax, class_id],
             ....., ...., ...., ...., ........]
    """
    def __init__(self, keep_difficult=False, class_to_index=None):
        self.keep_difficult = keep_difficult
        self.class_to_index = class_to_index or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))

    def __call__(self, target):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            # name = obj.find('name').text
            name = obj[0].text.lower().strip()
            # bb = obj.find('bndbox')
            bbox = obj.find('bndbox')
            # bndbox = [bb.find('xmin').text, bb.find('ymin').text,
            #    bb.find('xmax').text, bb.find('ymax').text]
            # supposes the order is xmin, ymin, xmax, ymax
            # attention with indices
            bndbox = [int(bb.text) - 1 for bb in bbox]

            res += [bndbox + [self.class_to_index[name]]]

        return res


class VOCDetection(data.Dataset):
    def __init__(self, root, image_sets, transform=None, target_transform=None):
        self.root = root
        self.image_sets = image_sets
        self.transform = transform
        self.target_transform = target_transform

        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()

        img = Image.open(self._imgpath % img_id).convert('RGB')
        img = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            target = np.array(target, dtype=np.float32)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def show(self, index):
        img, target = self.__getitem__(index)
        draw = ImageDraw.Draw(img)
        for obj in target:
            draw.rectangle(obj[0:4], outline=(255, 0, 0))
            draw.text(obj[0:2], obj[4], fill=(0, 255, 0))
        img.show()


if __name__ == '__main__':
    from data_aug import *
    from bbox_util import *
    import matplotlib.pyplot as plt
    ds = VOCDetection('./VOCdevkit/', [('2007-test', 'test')],
                      transform=BaseTransform(448),
                      target_transform=TransformVOCDetectionAnnotation(False))
    img, bboxes = ds[0]
    img = draw_rect(img, bboxes)
    plt.imshow(img)
    plt.show()






