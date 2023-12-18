import argparse
import os

import numpy as np
import tqdm
from PIL import Image
import cv2

import torch


from tools import add_palette, tensor_to_rgb


def check_m2fp():
    from third_party import M2FPBatchInfer, DWPoseBatchInfer
    img = np.array(Image.open("./samples/hoodie.jpg")).astype(np.uint8)
    infer = M2FPBatchInfer()
    seg_pil = infer.forward_rgb_as_pil(img)
    seg_pil.save("./tmp_m2fp_seg.png")


def check_dwpose():
    from third_party import M2FPBatchInfer, DWPoseBatchInfer
    img = np.array(Image.open("./samples/hoodie.jpg")).astype(np.uint8)
    infer = DWPoseBatchInfer()
    pose_arr = infer.forward_rgb_as_rgb(img)
    Image.fromarray(pose_arr.astype(np.uint8)).save("./tmp_dwpose.png")


def check_cp_dataset():
    dataset = CPDataset(
        "/cfs/yuange/datasets/xss/standard/hoodie/720_20231017_reordered_subpart/",
        mode="train",
        is_debug=True
    )
    sample: dict = dataset[0]
    for key in sample.keys():
        print(f"({key}):type={type(sample[key])}")


def check_distribute():
    import torch
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--local-rank', type=int)
    opt = args.parse_args()

    local_rank = opt.local_rank
    torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    device = torch.device(f'cuda:{local_rank}')

    model = torch.nn.Linear(512, 512).cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank)


def check_palette():
    # add_palette("/cfs/yuange/datasets/GPVTON/VITONHD-1024/train/cloth_parse-bytedance")
    # add_palette("/cfs/yuange/datasets/GPVTON/VITONHD-1024/test/cloth_parse-bytedance")
    # add_palette("/cfs/yuange/datasets/GPVTON/VITONHD-1024/train/parse-bytedance")
    # add_palette("/cfs/yuange/datasets/GPVTON/VITONHD-1024/test/parse-bytedance")
    from tools.cvt_data import get_coco_palette
    img_grid = np.zeros((1000, 1000))
    label = 0
    for i in range(10):
        for j in range(10):
            img_grid[i * 100: (i + 1) * 100, j * 100: (j + 1) * 100] = label
            label += 1
    img_grid = Image.fromarray(img_grid.astype(np.uint8)).convert("P")
    palette = get_coco_palette()
    viton_img = Image.open("/cfs/yuange/datasets/xss/processed/VITON-HD/train/parse/14684_00.png")
    viton_img = Image.fromarray(np.array(viton_img).astype(np.uint8))
    viton_img.save("tmp_viton_gray.png")
    img_grid.putpalette(palette)
    img_grid.save("tmp_palette_viton.png")


def check_gpvton_dataset():
    from tqdm import tqdm
    from tools.cvt_data import get_coco_palette
    from tools.cvt_data import seg_to_labels_and_one_hots, label_and_one_hot_to_seg
    from datasets import CPDataset, GPVTONSegDataset, GPMergedSegDataset
    # dataset = GPVTONSegDataset(
    #     "/cfs/yuange/datasets/VTON-HD/",
    #     mode="train",
    #     process_scale_ratio=0.5,
    # )
    dataset = GPMergedSegDataset(
        "/cfs/yuange/datasets/VTON-HD/",
        "/cfs/yuange/datasets/DressCode/",
        mode="train",
        process_scale_ratio=0.5,
    )
    snapshot_folder = "tmp_gpvton_snapshot"
    os.makedirs(snapshot_folder, exist_ok=True)
    n = len(dataset)
    n1 = dataset.len1
    n2 = dataset.len2
    assert n1 + n2 == n
    test_list = list(range(0, 10)) + list(range(n1, n1 + 10)) + list(range(n - 10, n))

    def save_image_and_seg(img: torch.Tensor, seg: torch.LongTensor, suffix: str = ""):
        pil = tensor_to_rgb(img, out_as_pil=True)
        seg_pil = tensor_to_rgb(seg, out_as_pil=True, is_segmentation=True).convert("L").convert("P")
        pil.save(os.path.join(snapshot_folder, f"{idx:05d}_{suffix}.jpg"))
        seg_pil.putpalette(get_coco_palette())
        seg_pil.save(os.path.join(snapshot_folder, f"{idx:05d}_{suffix}_seg.png"))

        mask_labels, class_labels = seg_to_labels_and_one_hots(seg)
        seg_gt = label_and_one_hot_to_seg(mask_labels[0], class_labels[0])
        seg_gt = Image.fromarray(seg_gt).convert("P")
        seg_gt.putpalette(get_coco_palette())
        seg_gt.save(os.path.join(snapshot_folder, f"{idx:05d}_{suffix}_seg_gt.png"))

    for idx in tqdm(test_list):
        batch = dataset[idx]
        cloth = batch["cloth"].unsqueeze(0)  # add batch dim
        cloth_seg = batch["cloth_seg"].unsqueeze(0)  # add batch dim
        person = batch["person"].unsqueeze(0)  # add batch dim
        person_seg = batch["person_seg"].unsqueeze(0)  # add batch dim

        save_image_and_seg(cloth, cloth_seg, "cloth")
        save_image_and_seg(person, person_seg, "person")


def check_mask2former(is_train : bool = False):
    import datetime
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from models import Mask2FormerPL
    pl.seed_everything(42)

    cloth_or_person = "person"

    log_root = "lightning_logs/"
    log_project = f"m2f_{cloth_or_person}"
    log_version = "version_12/"

    m2f = Mask2FormerPL(cloth_or_person=cloth_or_person)
    m2f.train_set.is_debug = True
    m2f.test_set.is_debug = True
    if is_train:
        log_version = now = datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
        weight = torch.load("./pretrained/m2f/pytorch_model.pt", map_location="cpu")
        m2f.load_state_dict(weight)
        tensorboard_logger = TensorBoardLogger(
            save_dir=log_root,
            name=log_project,
            version=log_version,
        )
    else:
        # weight = torch.load("./pretrained/m2f/pytorch_model.pt", map_location="cpu")
        # m2f.load_state_dict(weight)
        m2f = Mask2FormerPL.load_from_checkpoint(os.path.join(log_root, log_version, "checkpoints/last.ckpt"))
        tensorboard_logger = TensorBoardLogger(
            save_dir=log_root,
            name="",
            version=log_version,
            sub_dir="test",
        )

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=5,
        save_top_k=-1,
        save_on_train_epoch_end=True,
        save_last=True,
        verbose=True
    )
    trainer = pl.Trainer(
        # strategy="ddp",
        devices="2,3",
        fast_dev_run=False,
        max_epochs=100,
        limit_val_batches=2,
        val_check_interval=0.4,
        limit_test_batches=2,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        logger=tensorboard_logger,
    )

    if is_train:
        trainer.fit(m2f)
    else:
        trainer.test(m2f)


def check_ckpt():
    from tools.cvt_data import save_ckpt_as_pt
    save_ckpt_as_pt(
        "/cfs/yuange/code/XTryOn/lightning_logs/m2f_person/2023_11_03T16_09_30/checkpoints/epoch=99-step=84396.ckpt",
        # "/cfs/yuange/code/XTryOn/pretrained/m2f/pytorch_model.bin",
        "/cfs/yuange/code/XTryOn/pretrained/m2f/person_model.pt",
        remove_prefix=True,
    )


def check_crop_upper_and_shift():
    from tools.crop_image import calc_crop_upper_and_shift
    from tools.cvt_data import get_coco_palette
    from torch.utils.data import Dataset, DataLoader

    test_person = "./tools/dress_code_person.png"
    test_seg = "./tools/dress_code_person_parse.png"
    test_out_person = "tmp_person_crop.png"
    test_out_seg = "tmp_person_seg_crop.png"

    def process_crop_and_shift(in_image_path: str, in_seg_path: str, out_image_path: str, out_seg_path: str):
        person_pil = Image.open(in_image_path)
        seg_pil = Image.open(in_seg_path)
        person = np.array(person_pil).astype(np.uint8)
        seg = np.array(seg_pil).astype(np.uint8)
        bbox_xywh = calc_crop_upper_and_shift(person, seg, label_candidates=(5, 6, 11, 14))
        fx, fy, fw, fh = bbox_xywh
        final_person_pil = person_pil.crop((fx, fy, fx + fw, fy + fh))
        final_person_pil = final_person_pil.resize(person_pil.size, resample=Image.BILINEAR)
        final_seg_pil = seg_pil.crop((fx, fy, fx + fw, fy + fh))
        final_seg_pil = final_seg_pil.resize(seg_pil.size, resample=Image.NEAREST)
        final_seg_pil.putpalette(get_coco_palette())
        final_person_pil.save(out_image_path)
        final_seg_pil.save(out_seg_path)

    process_crop_and_shift(test_person, test_seg, test_out_person, test_out_seg)

    class DressCodeDataset(Dataset):
        def __init__(self):
            root = "/cfs/yuange/datasets/DressCode/upper"
            person_key = "image"
            person_seg_key = "parse-bytedance"
            person_upper_key = "person_upper"
            person_seg_upper_key = "person_upper_parse"
            os.makedirs(os.path.join(root, person_upper_key), exist_ok=True)
            os.makedirs(os.path.join(root, person_seg_upper_key), exist_ok=True)
            fns = os.listdir(os.path.join(root, person_key))
            fns.sort()
            self.root = root
            self.person_key = person_key
            self.person_seg_key = person_seg_key
            self.person_upper_key = person_upper_key
            self.person_seg_upper_key = person_seg_upper_key
            self.fns = fns

        def __len__(self):
            return len(self.fns)

        def __getitem__(self, index):  # process here
            fn = self.fns[index]
            in_image = os.path.join(self.root, self.person_key, fn)
            in_seg = os.path.join(self.root, self.person_seg_key, fn.replace(".jpg", ".png"))
            out_image = os.path.join(self.root, self.person_upper_key, fn)
            out_seg = os.path.join(self.root, self.person_seg_upper_key, fn.replace(".jpg", ".png"))
            process_crop_and_shift(in_image, in_seg, out_image, out_seg)
            return fn

    dress_dataset = DressCodeDataset()
    dataloader = DataLoader(dress_dataset, batch_size=1, shuffle=False, num_workers=12)
    for idx, batch in enumerate(tqdm.tqdm(dataloader)):
        pass


def check_mgd():
    # from third_party.pidinet.image_infer import PiDiNetBatchInfer
    # test_img = Image.open("samples/dresscode_warped.png").convert("RGB")
    # infer = PiDiNetBatchInfer()
    # pil = infer.forward_rgb_as_pil(np.array(test_img))
    # pil.save("tmp_pidinet.png")

    test_weight_path = "/cfs/yuange/code/XTryOn/lightning_logs/mgd/2023_11_17T15_44_47/checkpoints/epoch=99-step=112000.ckpt"

    # from models.generate.image_infer import mgd
    # model = mgd(
    #     in_channels=28 + 4,
    #     out_channels=4 + 1,
    #     pretrained=True,
    #     weight_path=None,
    # )

    from models.generate.image_infer import MGDBatchInfer
    infer_width, infer_height = 768, 1024
    model_img = Image.open("samples/ufo_ori.png").resize((infer_width, infer_height))
    model_rgb = np.array(model_img)
    warped_rgb = np.array(Image.open("samples/ufo_warped.png"))
    test_prompt = "a t-shirt with red logo, happy weekend texts"
    mgd_infer = MGDBatchInfer(
        infer_height=infer_height,
        infer_width=infer_width,
        unet_in_channels=28 + 4,
        unet_weight_path=test_weight_path
    )
    mgd_infer.forward_rgb_as_pil(model_rgb, test_prompt, warped_rgb)


def check_blip2():
    from third_party import BLIP2BatchInfer, BLIPBatchInfer
    infer1 = BLIPBatchInfer()
    infer2 = BLIP2BatchInfer()
    test_rgb = np.array(Image.open("samples/shirt_long_cloth.png"))

    out_text = ""
    for _ in tqdm.tqdm(range(100)):
        out_text = infer1.forward_rgb_as_str(test_rgb)
    print(out_text)

    for _ in tqdm.tqdm(range(100)):
        out_text = infer2.forward_rgb_as_str(test_rgb)
    print(out_text)


def check_json():
    from tools.task_tools import merge_json
    merge_json(
        "/cfs/yuange/datasets/m2f/DressCode/upper/processed/blip2_cloth",
        "/cfs/yuange/datasets/m2f/DressCode/upper/processed/blip2_cloth/blip2_cloth_all.json"
    )


def check_processed_dataset():
    from datasets.xss_datasets import ProcessedDataset, MergedProcessedDataset
    from tools.cvt_data import tensor_to_rgb
    # dataset = ProcessedDataset(
    #     "/cfs/yuange/datasets/xss/processed/",
    #     "DressCode/upper",
    #     debug_len=10,
    #     output_keys=(
    #         "person", "densepose", "inpaint_mask", "pose_map", "warped_person",
    #         "pidinet", "blip2_cloth", "person_fn"
    #     )
    # )

    dataset = MergedProcessedDataset(
        "/cfs/yuange/datasets/xss/processed/",
        ["DressCode/upper", "VITON-HD/train"],
        debug_len=10,
        output_keys=(
            "person", "cloth", "dwpose", "warped_person",
            "person_fn",
        ),
        scale_height=1024,
        scale_width=768,
        downsample_warped=True,
        mode="val",
        aug_flip=1.,
        aug_shift_prob=1.,
        aug_scale_prob=1.,
        aug_hsv_prob=1.,
        aug_contrast_prob=1.,
    )

    def print_item(key, val):
        print(f"({key}):")
        if isinstance(val, torch.Tensor):
            print(val.shape, val.min(), val.max())
            save_fn = f"./tmp_data_{key}.jpg"
            pil = tensor_to_rgb(val.unsqueeze(0), out_as_pil=True)
            pil.save(save_fn)
        else:
            print(val)

    test_item = dataset[0]
    for k, v in test_item.items():
        print_item(k, v)

    test_item = dataset[len(dataset) - 1]
    for k, v in test_item.items():
        print_item(k, v)


def check_gen_file_list():
    zhihui_folder = "/cfs/zhlin/projects/DCI-VTON-Virtual-Try-On/repositories/FlowStyleVTON/test/results/PBAFN_short_e2e_fs_fine_512_test/"
    level1_dir = "VITON-HD/train/"  # VITON-HD/train/ or DressCode/upper/
    out_fn = os.path.join("/cfs/yuange/datasets/xss/processed/", level1_dir, "train_list.txt")
    person_dir = "warped_person"
    cloth_dir = "warped_cloth"
    dir_abs = os.path.join(zhihui_folder, level1_dir, person_dir)
    fns = os.listdir(dir_abs)
    fns.sort()
    with open(out_fn, "w") as tmp_f:
        tmp_f.writelines([f"{fn}\n" for fn in fns])
    with open(out_fn, "r") as tmp_f:
        fns = [line.strip() for line in tmp_f.readlines()]
        print(fns)


def check_vis_point():
    def _vis_pose_map_as_pils(pose_map: torch.Tensor):
        b, c, h, w = pose_map.shape
        pose_map_arr = pose_map.cpu().numpy()
        ret_pils = []
        for b_idx in range(b):
            points = pose_map_arr[b_idx]  # (C,H,W)
            black_board = np.zeros((h, w), dtype=np.uint8)
            for c_idx in range(c):
                point = points[c_idx]
                black_board[point > 0.9] = (point[point > 0.9] * 255.).astype(np.uint8)
            ret_pils.append(Image.fromarray(black_board))
        return ret_pils

    pose_map = torch.randn((4, 18, 512, 384))
    pils = _vis_pose_map_as_pils(pose_map)
    print(len(pils))


def check_ddim_inversion():
    from diffusers.schedulers import DDIMInverseScheduler, DDIMScheduler
    from diffusers.models import UNet2DModel
    sc = DDIMInverseScheduler(
        beta_end=0.012,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        clip_sample=False,
        num_train_timesteps=1000,
        set_alpha_to_one=False,
        steps_offset=1 + 20,
        trained_betas=None,
    )
    sc.set_timesteps(50)
    print(sc.timesteps)

    s0 = DDIMScheduler.from_pretrained("pretrained/stable-diffusion-inpainting/scheduler")
    s0.set_timesteps(50)
    print(s0.timesteps)


def check_divide():
    from models.generate.image_infer import MGDBatchInfer
    from torchvision.transforms import transforms
    infer_width, infer_height = 768, 1024
    model_img = Image.open("samples/ufo_ori.png").resize((infer_width, infer_height))
    model_rgb = np.array(model_img).astype(np.float32)
    warped_rgb = np.array(Image.open("samples/ufo_warped.png").resize((infer_width, infer_height))).astype(np.float32)
    res = (model_rgb + 1) / (warped_rgb + 1)
    print(res.min(), res.max(), res.mean())
    scale = 255. / res.max()
    res = res * scale
    Image.fromarray(res.astype(np.uint8)).save("tmp_divide.png")

    # disturb
    h, w, c = res.shape
    # res += np.random.randn(h, w, c)
    trans = transforms.RandomResizedCrop((h, w))
    res_disturb = trans(Image.fromarray(res.astype(np.uint8)))
    res_disturb.save("tmp_divide_disturb.png")
    res = res_disturb
    res = ((res / scale) * (warped_rgb + 1) - 1).clip(0, 255).astype(np.uint8)
    Image.fromarray(res.astype(np.uint8)).save("tmp_divide_mul_back.png")


def check_aniany():
    # from models.generate.aniany import ConditionFCN
    # net = ConditionFCN()
    # pose_cond = torch.randn(4, 3, 512, 384)
    # out = net(pose_cond)
    # print(out.shape)

    # from models.generate.aniany import FrozenCLIPTextImageEmbedder
    # net = FrozenCLIPTextImageEmbedder()
    # source_img = torch.randn(4, 3, 224, 224)
    # out = net.encode({"image": source_img, "text": [""]})
    # print(out.shape)

    def cat_width_in_transformer(x1: torch.Tensor, x2: torch.Tensor, height: int, width: int):
        assert x1.shape == x2.shape
        batch_size, hw, channels = x1.shape
        assert hw == height * width
        x1 = x1.permute(0, 2, 1).reshape(batch_size, channels, height, width)
        x2 = x2.permute(0, 2, 1).reshape(batch_size, channels, height, width)
        x_cat = torch.cat([x1, x2], dim=-1)
        x_cat = x_cat.permute(0, 2, 3, 1).reshape(batch_size, height * width * 2, channels)
        return x_cat

    def chunk_width_in_transformer(x_cat: torch.Tensor, height: int, width: int, chunk_num: int = 2, keep_num: int = 0):
        batch_size, hw, channels = x_cat.shape
        assert hw == height * width * chunk_num
        x_cat = x_cat.permute(0, 2, 1).reshape(batch_size, channels, height, width * 2)
        x_0 = x_cat.chunk(chunk_num, dim=-1)[keep_num]
        x_0 = x_0.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        return x_0

    # h, w = 4, 3
    # x_ref = torch.ones((2, h * w, 4)) * 2
    # x_main = torch.ones((2, h * w, 4)) * 3
    # x_cat = cat_width_in_transformer(x_main, x_ref, height=h, width=w)
    # print(x_cat.shape)
    # x_main = chunk_width_in_transformer(x_cat, h, w)
    # print(x_main.shape)
    # print(x_main)

    weight_path = "/cfs/yuange/code/XTryOn/lightning_logs/aniany/2023_12_11T12_00_12/checkpoints/epoch=49-step=72700.ckpt"

    # from models.generate.aniany import aniany_unet
    # unet_ref = aniany_unet(in_channels=8, out_channels=4, weight_path=weight_path, weight_key="unet_ref.")
    # unet_main = aniany_unet(in_channels=4, out_channels=4, weight_path=weight_path, weight_key="unet_main.")

    from models.generate.aniany import AnimateAnyonePL
    model_pl = AnimateAnyonePL(
        train_set=None,
        val_set=None,
        noise_offset=0.1,
        input_perturbation=0.1,
        snr_gamma=5.0,
        resume_ckpt=weight_path,
    )


if __name__ == "__main__":
    # check_m2fp()
    # check_dwpose()
    # check_cp_dataset()
    # check_distribute()
    # check_palette()
    # check_gpvton_dataset()
    # check_mask2former(is_train=True)
    # check_ckpt()
    # check_crop_upper_and_shift()
    # check_mgd()
    # check_blip2()
    # check_json()
    # check_processed_dataset()
    # check_gen_file_list()
    # check_vis_point()
    # check_ddim_inversion()
    # check_divide()
    check_aniany()
