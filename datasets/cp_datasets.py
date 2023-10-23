import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import json

import os
import os.path as osp
import numpy as np


class CPDataset(data.Dataset):
    """
        Dataset for training Warp-Net.
        Modified on: https://github.com/bcmi/DCI-VTON-Virtual-Try-On/blob/main/warp/train/data/cp_dataset.py
    """

    def __init__(self, dataroot, image_size=512, mode='train', semantic_nc=13,
                 is_debug: bool = False,
                 debug_folder: str = "tmp_snapshot",
                 ):
        super(CPDataset, self).__init__()
        # base setting
        self.root = dataroot
        self.datamode = mode  # train or test or self-defined
        self.data_list = mode + '_pairs.txt'
        self.fine_height = image_size
        self.fine_width = int(image_size / 256 * 192)
        self.semantic_nc = semantic_nc
        # self.data_path = osp.join(dataroot, mode)
        self.data_path = osp.join(dataroot, "")
        self.toTensor = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.key_person = "person"
        self.key_cloth = "cloth"
        self.key_cloth_mask = "cloth_mask"
        self.key_pose = "dwpose"
        self.key_densepose = "densepose"
        self.key_parsing = "m2fp"
        self.key_parsing_ag = "agnostic_gen"

        # load data list
        # im_names = []
        # c_names = []
        # with open(osp.join(dataroot, self.data_list), 'r') as f:
        #     for line in f.readlines():
        #         im_name, c_name = line.strip().split()
        #         im_names.append(im_name)
        #         c_names.append(c_name)
        im_names = self.load_data_list()

        self.im_names = im_names
        self.c_names = {
            "paired": im_names,
        }
        # self.c_names['unpaired'] = c_names  # not used

        self.is_debug = is_debug
        self.debug_folder = debug_folder
        if is_debug:
            os.system(f"rm -r {debug_folder}")
            os.makedirs(debug_folder)
            print(f"[CPDataset] Debug mode, images will be saved to: {debug_folder}")

    def load_data_list(self):
        person_abs = os.path.join(self.root, "person")
        fns = os.listdir(person_abs)
        fns.sort()
        return fns

    def name(self):
        return "CPDataset"

    def get_agnostic(self, im, im_parse, pose_data):
        if pose_data is None:
            return im
        parse_array = np.array(im_parse)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 16).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))

        agnostic = im.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        r = int(length_a / 16) + 1

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r * 12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx - r * 5, pointy - r * 9, pointx + r * 5, pointy), 'gray', 'gray')

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r * 12)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                    pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r * 10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'gray', 'gray')

        for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
            # mask_arm = Image.new('L', (self.fine_width, self.fine_height), 'white')
            mask_arm = Image.new('L', (768, 1024), 'white')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            pointx, pointy = pose_data[pose_ids[0]]
            mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), 'black', 'black')
            for i in pose_ids[1:]:
                if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                        pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r * 10)
                pointx, pointy = pose_data[i]
                if i != pose_ids[-1]:
                    mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'black',
                                          'black')
            mask_arm_draw.ellipse((pointx - r * 4, pointy - r * 4, pointx + r * 4, pointy + r * 4), 'black', 'black')

            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
        return agnostic

    def __getitem__(self, index):
        im_name = self.im_names[index]
        # im_name = 'image/' + im_name
        c_name = {}
        c = {}
        cm = {}
        for key in ['paired']:
            c_name[key] = self.c_names[key][index]
            c[key] = Image.open(osp.join(self.data_path, self.key_cloth, c_name[key])).convert('RGB')
            c[key] = transforms.Resize(self.fine_width, interpolation=2)(c[key])
            cm[key] = Image.open(osp.join(self.data_path, self.key_cloth_mask, c_name[key])).convert('L')
            cm[key] = transforms.Resize(self.fine_width, interpolation=0)(cm[key])

            c[key] = self.transform(c[key])  # [-1,1]
            cm_array = np.array(cm[key])
            cm_array = (cm_array >= 128).astype(np.float32)
            cm[key] = torch.from_numpy(cm_array)  # [0,1]
            cm[key].unsqueeze_(0)

        # person image
        im_pil_big = Image.open(osp.join(self.data_path, self.key_person, im_name))
        im_pil = transforms.Resize(self.fine_width, interpolation=2)(im_pil_big)
        im = self.transform(im_pil)

        # load parsing image
        # parse_name = im_name.replace('image', 'image-parse-v3').replace('.jpg', '.png')
        parse_name = osp.join(self.key_parsing, im_name)
        im_parse_pil_big = Image.open(osp.join(self.data_path, parse_name))
        im_parse_pil = transforms.Resize(self.fine_width, interpolation=0)(im_parse_pil_big)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()
        im_parse = self.transform(im_parse_pil.convert('RGB'))

        # parse map
        labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }

        parse_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()

        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_map[i] += parse_map[label]

        parse_onehot = torch.FloatTensor(1, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse_onehot[0] += parse_map[label] * i

        # load image-parse-agnostic
        # image_parse_agnostic = Image.open(
        #     osp.join(self.data_path, parse_name.replace('image-parse-v3', 'image-parse-agnostic-v3.2')))
        image_parse_agnostic = Image.open(
                osp.join(self.data_path, self.key_parsing_ag, im_name))
        image_parse_agnostic = transforms.Resize(self.fine_width, interpolation=0)(image_parse_agnostic)
        parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
        image_parse_agnostic = self.transform(image_parse_agnostic.convert('RGB'))

        parse_agnostic_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

        # parse cloth & parse cloth mask
        pcm = new_parse_map[3:4]
        im_c = im * pcm + (1 - pcm)

        # load pose points
        # pose_name = im_name.replace('image', 'openpose_img').replace('.jpg', '_rendered.png')
        # pose_map = Image.open(osp.join(self.data_path, pose_name))
        pose_map = Image.open(osp.join(self.data_path, self.key_pose, im_name))
        pose_map = transforms.Resize(self.fine_width, interpolation=2)(pose_map)
        pose_map = self.transform(pose_map)  # [-1,1]

        # pose name (json, not used)
        # pose_name = im_name.replace('image', 'openpose_json').replace('.jpg', '_keypoints.json')
        # with open(osp.join(self.data_path, pose_name), 'r') as f:
        #     pose_label = json.load(f)
        #     pose_data = pose_label['people'][0]['pose_keypoints_2d']
        #     pose_data = np.array(pose_data)
        #     pose_data = pose_data.reshape((-1, 3))[:, :2]

        # load densepose
        # densepose_name = im_name.replace('image', 'image-densepose')
        # densepose_map = Image.open(osp.join(self.data_path, densepose_name))
        densepose_map = Image.open(osp.join(self.data_path, self.key_densepose, im_name))
        # print("P mode:", np.array(densepose_map).max())
        densepose_map = densepose_map.convert("RGB")
        # print("RGB mode:", np.array(densepose_map).max())
        densepose_map = transforms.Resize(self.fine_width, interpolation=2)(densepose_map)
        densepose_map = self.transform(densepose_map)  # [-1,1]

        # agnostic
        # agnostic = self.get_agnostic(im_pil_big, im_parse_pil_big, pose_data)
        agnostic = self.get_agnostic(im_pil_big, im_parse_pil_big, pose_data=None)
        agnostic = transforms.Resize(self.fine_width, interpolation=2)(agnostic)
        agnostic = self.transform(agnostic)

        # warped_cloth_name = im_name.replace('image', 'cloth-warp')
        # warped_cloth = Image.open(osp.join(self.data_path, warped_cloth_name))
        # warped_cloth = transforms.Resize(self.fine_width, interpolation=2)(warped_cloth)
        # warped_cloth = self.transform(warped_cloth)
        #
        # warped_cloth_mask_name = im_name.replace('image', 'cloth-warp-mask')
        # warped_cloth_mask = Image.open(osp.join(self.data_path, warped_cloth_mask_name))
        # warped_cloth_mask = transforms.Resize(self.fine_width, interpolation=transforms.InterpolationMode.NEAREST) \
        #     (warped_cloth_mask)
        # warped_cloth_mask = self.toTensor(warped_cloth_mask)
        warped_cloth = ""
        warped_cloth_mask = ""

        if self.is_debug:
            to_img = transforms.ToPILImage()
            to_img((c['paired'] + 1) / 2.0).save(osp.join(self.debug_folder, 'cloth.jpg'))
            to_img((densepose_map + 1) / 2.0).save(osp.join(self.debug_folder, 'densepose.jpg'))
            to_img((pose_map + 1) / 2.0).save(osp.join(self.debug_folder, 'pose.jpg'))
            to_img((agnostic + 1) / 2.0).save(osp.join(self.debug_folder, 'agnostic.jpg'))
            to_img((im_c + 1) / 2.0).save(osp.join(self.debug_folder, 'warped_cloth_gt.jpg'))

        result = {
            'c_name': c_name,  # for visualization
            'im_name': im_name,  # for visualization or ground truth
            # intput 1 (clothfloww)
            'cloth': c,  # for input
            'cloth_mask': cm,  # for input
            # intput 2 (segnet)
            'parse_agnostic': new_parse_agnostic_map,
            'densepose': densepose_map,
            'pose': pose_map,  # for conditioning
            # generator input
            'agnostic': agnostic,
            # GT
            'parse_onehot': parse_onehot,  # Cross Entropy
            'parse': new_parse_map,  # GAN Loss real
            'pcm': pcm,  # L1 Loss & vis
            'parse_cloth': im_c,  # VGG Loss & vis
            # visualization & GT
            'image': im,  # for visualization
            'warped_cloth': warped_cloth,  # maybe ""
            'warped_cloth_mask': warped_cloth_mask  # maybe ""
        }

        return result

    def __len__(self):
        return len(self.im_names)
