import torch
import torch.utils.data
import torchvision
import numpy as np

import pickle
import json
import os, sys
from PIL import Image, ImageFile
from tqdm import tqdm
from src.utils.geometry_utils import angle2class


class IGibsonDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, cfg):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        print("loading data")

        self.data_path = os.path.join(data_path, "frame_datas")
        self.img_path = os.path.join(data_path, "images")
        self.cfg = cfg

        self.angle_included_categories = json.load(
            open(os.path.join(data_path, "categories100.json"), "rb")
            # open(os.path.join(data_path, "categories_minor.json"), "rb")
        )
        idx = 0
        self.categories = {}
        for category in self.angle_included_categories:
            self.categories[category] = idx
            idx += 1

        self.targets = []
        self.img_names = []

        # for file in tqdm(os.listdir(self.data_path)):
        for n, file in enumerate(os.listdir(self.data_path)):
            if file.split("_")[-3] == "1":
                # skip test data
                continue

            if n % 5 != 0:
                continue

            try:
                frame_datas = pickle.load(
                    open(os.path.join(self.data_path, file), "rb")
                )
            except pickle.UnpicklingError:
                print(file)
                continue

            for frame_data in frame_datas:
                self.targets.append(self.get_target(self.preprocess_data(frame_data)))
                self.img_names.append(frame_data["file_name"])

    def __getitem__(self, idx):
        target = self.targets[idx]
        img = Image.open(os.path.join(self.img_path, self.img_names[idx]))

        return torchvision.transforms.ToTensor()(img), target

    def preprocess_data(self, data):
        invalid_ids = []

        for obj_id in data["objects"]:
            if not data["objects"][obj_id]["category"] in self.categories:
                invalid_ids.append(obj_id)

        for obj_id in invalid_ids:
            del data["objects"][obj_id]
        return data

    def get_target(self, datum):
        length = len(datum["objects"])
        objects = np.empty((length, 12))

        for idx, obj_id in enumerate(datum["objects"]):
            annotation = datum["objects"][obj_id]

            objects[idx][0] = self.categories[annotation["category"]]
            objects[idx][1:5] = annotation["bbox_2d"]["xywh"]
            objects[idx][5:8] = annotation["bbox_3d"]["global"]["size"]
            objects[idx][8:10] = annotation["bbox_3d"]["relative"]["offset"]
            objects[idx][-2] = annotation["bbox_3d"]["relative"]["depth"]
            objects[idx][-1] = angle2class(
                torch.tensor(annotation["bbox_3d"]["relative"]["angle"]) / 180 * np.pi,
                60,
            )

        objects = torch.tensor(objects).float()
        target = {"objects": objects}
        return target

    def __len__(self):
        return len(self.targets)


# import torch
# import torch.utils.data
# import torchvision
# import numpy as np

# from glob import glob
# import pickle
# import json
# import os, sys
# from PIL import Image
# from tqdm import tqdm
# from src.utils.geometry_utils import angle2class
# import src.datasets.transforms as T
# #from igibson.utils.transform_utils import mat2euler
# from .bins import ORI_BINS, DEPTH_BINS, num2bins


# class IGibsonDataset(torch.utils.data.Dataset):
#     def __init__(self, data_path, cfg):
#         super(IGibsonDataset, self).__init__()
#         print("loading data")

#         self.data_path = os.path.join(data_path, "frame_datas")
#         self.img_path = os.path.join(data_path, "images")
#         self.cfg = cfg
#         self.mode = cfg['mode']

#         if self.mode == 'test':
#             self.transforms = T.Compose([
#                 T.RandomResize([800], max_size=1333),
#                 T.ToTensor(),
#                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ])
#         else:
#             normalize = T.Compose([
#                 T.ToTensor(),
#                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ])
#             scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
#             self.transforms = T.Compose([
#                 T.RandomResize(scales, max_size=1333),
#                 normalize])

#         self.w = cfg['dataset']['img_w']
#         self.h = cfg['dataset']['img_h']

#         self.categories = json.load(
#             open(os.path.join(data_path, "categories100.json"), "rb")
#         )
#         self.category_to_id = {k: v for v, k in enumerate(self.categories)}
#         self.id_to_category = {k: v for k, v in enumerate(self.categories)}

#         self.targets = []
#         self.img_names = []

#         self.ori_bins = ORI_BINS
#         self.depth_bins = DEPTH_BINS

#         files = sorted(glob(f'{self.data_path}/*'))
#         train1_list = []
#         train2_list = []
#         test_list = []

#         for n in range(100):
#             scenes = []
#             for file in files[5*n:5*n+5]:
#                 scenes.append(file.split('_0_')[1].split('_')[0])
#             for file in files[5*n:5*n+5]:
#                 if scenes.count(file.split('_0_')[1].split('_')[0]) == 1:
#                     test_list.append(file)
#                 elif "int_1" in file:
#                     train2_list.append(file)
#                 else:
#                     train1_list.append(file)

#         if cfg['dataset'][self.mode]['split'] == 'train1' : files = train1_list
#         elif cfg['dataset'][self.mode]['split'] == 'train2' : files = train2_list
#         elif cfg['dataset'][self.mode]['split'] == 'test' : files = test_list

#         if self.mode == 'train':
#             for n, file in enumerate(files):
#                 if n % cfg['dataset'][self.mode]['drop'] == 0 : files.remove(file)

#         for file in files:
#             with open(file, 'rb') as f:
#                 frame_datas = pickle.load(f)
#                 f.close()

#             idx = 0
#             for frame_data in frame_datas:
#                 idx += 1
#                 if idx % cfg['dataset'][self.mode]['interval'] == cfg['dataset'][self.mode]['interval_seed']:
#                     datum = self.preprocess_data(frame_data)
#                     if len(datum['objects']) == 0 : idx-=1
#                     else:
#                         self.targets.append(
#                             self.get_target(datum)
#                         )
#                         self.img_names.append(frame_data["file_name"])

#         print(f'LOADED {len(files)} episodes with {len(self.img_names)} frames for MODE : {self.mode}')
#         print("="*100)

#     def __getitem__(self, idx):
#         target = self.targets[idx]
#         with Image.open(os.path.join(self.img_path, self.img_names[idx])) as img_f:
#             img = img_f
#             if self.transforms is not None:
#                 img, target = self.transforms(img, target)
#             img_f.close()

#         return img, target

#     def preprocess_data(self, data):
#         invalid_ids = []

#         for obj_id, obj in data["objects"].items():
#             if not data["objects"][obj_id]["category"] in self.categories:
#                 invalid_ids.append(obj_id)
#                 continue
#             if True in obj['fixed'] : obj['fixed'] = True
#             else : obj['fixed'] = False
#             offset_0 = obj['bbox_3d']['relative']['offset']/obj['bbox_3d']['relative']['depth']
#             center_3d = np.array([(offset_0[0]+1)*self.w/2,(1-offset_0[1])*self.h/2])
#             xyxy = obj['bbox_2d']['xyxy']
#             center_2d = np.array([(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2])
#             obj['bbox_2d']['xywh'] = np.array([center_2d[0],center_2d[1],xyxy[2]-xyxy[0],xyxy[3]-xyxy[1]])
#             offset = center_3d - center_2d
#             obj['bbox_3d']['relative']['offset'] = offset
#             obj['bbox_3d']['relative']['angle'] = mat2euler(obj['bbox_3d']['relative']['rotation'])[2]

#         for obj_id in invalid_ids:
#             del data["objects"][obj_id]
#         return data

#     def get_target(self, datum):
#         length = len(datum["objects"])
#         objects = np.empty((length, 14))

#         idx = 0
#         for obj_id in datum["objects"]:
#             annotation = datum["objects"][obj_id]

#             objects[idx][0] = self.category_to_id[annotation["category"]]
#             objects[idx][1:5] = annotation["bbox_2d"]["xywh"]
#             objects[idx][5:8] = annotation["bbox_3d"]["global"]["size"]
#             objects[idx][8:10] = annotation["bbox_3d"]["relative"]["offset"]
#             objects[idx][10], objects[idx][11] = num2bins(
#                 self.depth_bins,
#                 annotation["bbox_3d"]["relative"]["depth"]
#                 )
#             objects[idx][12], objects[idx][13] = num2bins(
#                 self.ori_bins,
#                 fit_rad(annotation["bbox_3d"]["relative"]["angle"])
#             )
#             #objects[idx][12] = annotation['on_floor']
#             #objects[idx][13] = annotation['in_hand_of']
#             idx += 1

#         objects = torch.tensor(objects).float()
#         target = {"objects": objects}
#         return target

#     def __len__(self):
#         return len(self.targets)

# def fit_rad(rad):
#     if rad < - np.pi/2 : return rad + np.pi
#     elif rad > np.pi/2 : return rad - np.pi
#     else : return rad


# import math

# PI = np.pi
# EPS = np.finfo(float).eps * 4.0

# # axis sequences for Euler angles
# _NEXT_AXIS = [1, 2, 0, 1]

# # map axes strings to/from tuples of inner axis, parity, repetition, frame
# _AXES2TUPLE = {
#     "sxyz": (0, 0, 0, 0),
#     "sxyx": (0, 0, 1, 0),
#     "sxzy": (0, 1, 0, 0),
#     "sxzx": (0, 1, 1, 0),
#     "syzx": (1, 0, 0, 0),
#     "syzy": (1, 0, 1, 0),
#     "syxz": (1, 1, 0, 0),
#     "syxy": (1, 1, 1, 0),
#     "szxy": (2, 0, 0, 0),
#     "szxz": (2, 0, 1, 0),
#     "szyx": (2, 1, 0, 0),
#     "szyz": (2, 1, 1, 0),
#     "rzyx": (0, 0, 0, 1),
#     "rxyx": (0, 0, 1, 1),
#     "ryzx": (0, 1, 0, 1),
#     "rxzx": (0, 1, 1, 1),
#     "rxzy": (1, 0, 0, 1),
#     "ryzy": (1, 0, 1, 1),
#     "rzxy": (1, 1, 0, 1),
#     "ryxy": (1, 1, 1, 1),
#     "ryxz": (2, 0, 0, 1),
#     "rzxz": (2, 0, 1, 1),
#     "rxyz": (2, 1, 0, 1),
#     "rzyz": (2, 1, 1, 1),
# }

# _TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# def vec(values):
#     """
#     Converts value tuple into a numpy vector.

#     Args:
#         values (n-array): a tuple of numbers

#     Returns:
#         np.array: vector of given values
#     """
#     return np.array(values, dtype=np.float32)

# def mat2euler(rmat, axes="sxyz"):
#     """
#     Converts given rotation matrix to euler angles in radian.

#     Args:
#         rmat (np.array): 3x3 rotation matrix
#         axes (str): One of 24 axis sequences as string or encoded tuple (see top of this module)

#     Returns:
#         np.array: (r,p,y) converted euler angles in radian vec3 float
#     """
#     try:
#         firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
#     except (AttributeError, KeyError):
#         firstaxis, parity, repetition, frame = axes

#     i = firstaxis
#     j = _NEXT_AXIS[i + parity]
#     k = _NEXT_AXIS[i - parity + 1]

#     M = np.array(rmat, dtype=np.float32, copy=False)[:3, :3]
#     if repetition:
#         sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
#         if sy > EPS:
#             ax = math.atan2(M[i, j], M[i, k])
#             ay = math.atan2(sy, M[i, i])
#             az = math.atan2(M[j, i], -M[k, i])
#         else:
#             ax = math.atan2(-M[j, k], M[j, j])
#             ay = math.atan2(sy, M[i, i])
#             az = 0.0
#     else:
#         cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
#         if cy > EPS:
#             ax = math.atan2(M[k, j], M[k, k])
#             ay = math.atan2(-M[k, i], cy)
#             az = math.atan2(M[j, i], M[i, i])
#         else:
#             ax = math.atan2(-M[j, k], M[j, j])
#             ay = math.atan2(-M[k, i], cy)
#             az = 0.0

#     if parity:
#         ax, ay, az = -ax, -ay, -az
#     if frame:
#         ax, az = az, ax
#     return vec((ax, ay, az))
