# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List, Tuple

from mmengine.dataset import BaseDataset
from mmrotate.registry import DATASETS


@DATASETS.register_module()
class DVDataset(BaseDataset):
    METAINFO = {
        "classes": ("car", "truck", "bus", "van", "freight-car"),
        "palette": [(0, 255, 0), (255, 0, 0), (255, 255, 0), (128, 0, 128), (255, 165, 0)],
    }

    def __init__(
        self,
        data_root: str = "",
        img_shape: Tuple[int, int] = (1024, 1024),
        diff_thr: int = 100,
        img_suffix: str = "png",
        aux_ann_file: str = "",
        **kwargs,
    ) -> None:
        self.img_shape = img_shape
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        self.data_root = data_root
        self.aux_ann_file = osp.join(self.data_root, aux_ann_file)
        super().__init__(data_root=data_root, **kwargs)

    def load_data_list(self) -> List[dict]:
        cls_map = {c: i for i, c in enumerate(self.metainfo["classes"])}

        data_list = []

        txt_path = self.ann_file
        aux_txt_path = self.aux_ann_file

        txt_files = glob.glob(osp.join(txt_path, "*.txt"))

        if len(txt_files) == 0:
            raise ValueError("There is no txt file in " f"{txt_path}")

        for txt_file in txt_files:
            data_info = {}
            img_id = osp.split(txt_file)[1][:-4]
            data_info["img_id"] = img_id
            img_name = img_id + f".{self.img_suffix}"
            data_info["file_name"] = img_name
            data_info["img_path"] = osp.join(self.data_prefix["img_path"], img_name)
            data_info["img_path2"] = osp.join(self.data_prefix["img_path2"], img_name)

            instances = []
            with open(txt_file, "r", encoding="utf-8") as f:
                s = f.readlines()
                for si in s:
                    instance = {}
                    bbox_info = si.split()
                    instance["bbox"] = [float(i) for i in bbox_info[:8]]
                    cls_name = bbox_info[8]
                    instance["bbox_label"] = cls_map[cls_name]
                    difficulty = int(bbox_info[9])
                    if difficulty > self.diff_thr:
                        instance["ignore_flag"] = 1
                    else:
                        instance["ignore_flag"] = 0
                    instances.append(instance)
            data_info["instances"] = instances  # use ir labels

            instances2 = []
            with open(osp.join(aux_txt_path, data_info["img_id"] + ".txt")) as f:
                s = f.readlines()
                for si in s:
                    instance = {}
                    bbox_info = si.split()
                    instance["bbox"] = [float(i) for i in bbox_info[:8]]
                    cls_name = bbox_info[8]
                    instance["bbox_label"] = cls_map[cls_name]
                    difficulty = int(bbox_info[9])
                    if difficulty > self.diff_thr:
                        instance["ignore_flag"] = 1
                    else:
                        instance["ignore_flag"] = 0
                    instances2.append(instance)
            data_info["instances2"] = instances2

            data_list.append(data_info)

        return data_list
