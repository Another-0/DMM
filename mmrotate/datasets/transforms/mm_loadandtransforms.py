import mmcv
import numpy as np

import torch
from mmcv.transforms import LoadImageFromFile
from mmcv.transforms import to_tensor
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
import mmengine.fileio as fileio
from mmengine.structures import InstanceData

from mmdet.structures.bbox import get_box_type, BaseBoxes
from mmdet.structures.bbox.box_type import autocast_box_type
from mmdet.structures import DetDataSample
from mmdet.datasets.transforms import RandomFlip, Resize
from mmdet.datasets.transforms.formatting import PackDetInputs

from mmrotate.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadRGBTImageFromFile(LoadImageFromFile):
    def transform(self, results: dict) -> dict:
        filename = results["img_path"]
        filename2 = results["img_path2"]
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(self.file_client_args, filename)
                file_client2 = fileio.FileClient.infer_client(self.file_client_args, filename2)
                img_bytes = file_client.get(filename)
                img_bytes2 = file_client2.get(filename2)
            else:
                img_bytes = fileio.get(filename, backend_args=self.backend_args)
                img_bytes2 = fileio.get(filename2, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            img2 = mmcv.imfrombytes(img_bytes2, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        if self.to_float32:
            img = img.astype(np.float32)
            img2 = img2.astype(np.float32)

        results["img"] = img  # vi
        results["img2"] = img2  # ir
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]
        results["scale_factor"] = (1, 1)
        return results


@TRANSFORMS.register_module()
class LoadRGBTAnnotations(MMCV_LoadAnnotations):
    def __init__(
        self,
        with_mask: bool = False,
        poly2mask: bool = True,
        box_type: str = "hbox",
        # use for semseg
        reduce_zero_label: bool = False,
        ignore_index: int = 255,
        **kwargs,
    ) -> None:
        super(LoadRGBTAnnotations, self).__init__(**kwargs)
        self.with_mask = with_mask
        self.poly2mask = poly2mask
        self.box_type = box_type
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index = ignore_index

    def _load_bboxes(self, results: dict) -> None:
        gt_bboxes = []
        gt_bboxes2 = []
        gt_ignore_flags = []
        for instance in results.get("instances", []):
            gt_bboxes.append(instance["bbox"])
            gt_ignore_flags.append(instance["ignore_flag"])

        for instance2 in results.get("instances2", []):
            gt_bboxes2.append(instance2["bbox"])

        if self.box_type is None:
            results["gt_bboxes"] = np.array(gt_bboxes, dtype=np.float32).reshape((-1, 4))
            results["gt_bboxes2"] = np.array(gt_bboxes2, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results["gt_bboxes"] = box_type_cls(gt_bboxes, dtype=torch.float32)
            results["gt_bboxes2"] = box_type_cls(gt_bboxes2, dtype=torch.float32)
        results["gt_ignore_flags"] = np.array(gt_ignore_flags, dtype=bool)

    def _load_labels(self, results: dict) -> None:
        gt_bboxes_labels = []
        gt_bboxes_labels2 = []
        for instance in results.get("instances", []):
            gt_bboxes_labels.append(instance["bbox_label"])
        for instance2 in results.get("instances2", []):
            gt_bboxes_labels2.append(instance2["bbox_label"])
        results["gt_bboxes_labels"] = np.array(gt_bboxes_labels, dtype=np.int64)
        results["gt_bboxes_labels2"] = np.array(gt_bboxes_labels2, dtype=np.int64)


@TRANSFORMS.register_module()
class ResizeRGBT(Resize):
    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""
        if results.get("img", None) is not None:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results["img"],
                    results["scale"],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend,
                )
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results["img"].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results["img"],
                    results["scale"],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend,
                )

            results["img"] = img
            results["img_shape"] = img.shape[:2]
            results["scale_factor"] = (w_scale, h_scale)
            results["keep_ratio"] = self.keep_ratio

        if results.get("img2", None) is not None:
            if self.keep_ratio:
                img2, _ = mmcv.imrescale(
                    results["img2"],
                    results["scale"],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend,
                )
            else:
                img2, _, _ = mmcv.imresize(
                    results["img2"],
                    results["scale"],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend,
                )
            results["img2"] = img2

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get("gt_bboxes", None) is not None:
            results["gt_bboxes"].rescale_(results["scale_factor"])
            if self.clip_object_border:
                results["gt_bboxes"].clip_(results["img_shape"])

        if results.get("gt_bboxes2", None) is not None:
            results["gt_bboxes2"].rescale_(results["scale_factor"])
            if self.clip_object_border:
                results["gt_bboxes2"].clip_(results["img_shape"])


@TRANSFORMS.register_module()
class RandomFlipRGBT(RandomFlip):
    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results["img"] = mmcv.imflip(results["img"], direction=results["flip_direction"])
        results["img2"] = mmcv.imflip(results["img2"], direction=results["flip_direction"])

        img_shape = results["img"].shape[:2]

        # flip bboxes
        if results.get("gt_bboxes", None) is not None:
            results["gt_bboxes"].flip_(img_shape, results["flip_direction"])
            # results["gt_bboxes"].regularize_boxes(self.angle_version)

        # flip bboxes
        if results.get("gt_bboxes2", None) is not None:
            results["gt_bboxes2"].flip_(img_shape, results["flip_direction"])

        # flip masks
        if results.get("gt_masks", None) is not None:
            results["gt_masks"] = results["gt_masks"].flip(results["flip_direction"])

        # flip segs
        if results.get("gt_seg_map", None) is not None:
            results["gt_seg_map"] = mmcv.imflip(results["gt_seg_map"], direction=results["flip_direction"])

        self._record_homography_matrix(results)


@TRANSFORMS.register_module()
class PackRGBTDetInputs(PackDetInputs):
    mapping_table = {
        "gt_bboxes": "bboxes",
        "gt_bboxes_labels": "labels",
        "gt_bboxes2": "bboxes",
        "gt_bboxes_labels2": "labels",
        "gt_masks": "masks",
    }

    def __init__(
        self,
        meta_keys=(
            "img_id",
            "img_path",
            "img_path2",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "flip_direction",
        ),
    ):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        if "img" in results:
            img = results["img"]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results["inputs"] = img

        if "img2" in results:
            img2 = results["img2"]
            if len(img2.shape) < 3:
                img2 = np.expand_dims(img2, -1)
            if not img2.flags.c_contiguous:
                img2 = np.ascontiguousarray(img2.transpose(2, 0, 1))
                img2 = to_tensor(img2)
            else:
                img2 = to_tensor(img2).permute(2, 0, 1).contiguous()

            packed_results["inputs2"] = img2

        if "gt_ignore_flags" in results:
            valid_idx = np.where(results["gt_ignore_flags"] == 0)[0]
            ignore_idx = np.where(results["gt_ignore_flags"] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        instance_data2 = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == "gt_masks" or isinstance(results[key], BaseBoxes):
                if "gt_ignore_flags" in results:
                    if key.endswith("2"):
                        instance_data2[self.mapping_table[key]] = results[key]
                    else:
                        instance_data[self.mapping_table[key]] = results[key][valid_idx]
                        ignore_instance_data[self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if "gt_ignore_flags" in results:
                    if key.endswith("2"):
                        instance_data2[self.mapping_table[key]] = to_tensor(results[key])
                    else:
                        instance_data[self.mapping_table[key]] = to_tensor(results[key][valid_idx])
                        ignore_instance_data[self.mapping_table[key]] = to_tensor(results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(results[key])

        data_sample.gt_instances = instance_data
        data_sample.gt_instances2 = instance_data2
        data_sample.ignored_instances = ignore_instance_data

        if "proposals" in results:
            proposals = InstanceData(
                bboxes=to_tensor(results["proposals"]),
                scores=to_tensor(results["proposals_scores"]),
            )
            data_sample.proposals = proposals

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                assert key in results, f"`{key}` is not found in `results`, " f"the valid keys are {list(results)}."
                img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results["data_samples"] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(meta_keys={self.meta_keys})"
        return repr_str
