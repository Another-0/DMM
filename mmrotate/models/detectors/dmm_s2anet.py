import copy
from typing import Dict, List, Tuple, Union
import torch
from torch import Tensor
from collections.abc import Sequence

from mmengine.model import ModuleList
from mmengine.utils import is_list_of
from mmengine.config import ConfigDict

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.utils import unpack_gt_instances

from mmrotate.registry import MODELS

ForwardResults = Union[Dict[str, Tensor], List[DetDataSample], Tuple[Tensor], Tensor]


# final
@MODELS.register_module()
class virRefineSingleStageDetector(BaseDetector):
    def __init__(
        self,
        backbone_vi: ConfigType,
        backbone_ir: ConfigType,
        fusblock: ConfigType = None,
        neck: OptConfigType = None,
        mtablock: OptConfigType = None,
        aux_neck: OptConfigType = None,
        aux_rpn_head: OptConfigType = None,
        bbox_head_init: OptConfigType = None,
        bbox_head_refine: List[OptConfigType] = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone_vi = MODELS.build(backbone_vi)
        self.backbone_ir = MODELS.build(backbone_ir)

        if fusblock is not None:
            self.fusblock = MODELS.build(fusblock)

        if aux_neck is not None:
            self.aux_rpn_task = True
            self.aux_neck = MODELS.build(aux_neck)
            self.aux_neck.eval()
            for param in self.aux_neck.parameters():
                param.requires_grad = False
        else:
            self.aux_rpn_task = False

        if mtablock is not None:
            self.mtablock = MODELS.build(mtablock)

        if self.aux_rpn_task:
            if aux_rpn_head is not None and aux_neck is not None:
                self.aux_rpn_head = MODELS.build(aux_rpn_head)
                self.aux_rpn_head.eval()
                for param in self.aux_rpn_head.parameters():
                    param.requires_grad = False

        if neck is not None:
            self.neck = MODELS.build(neck)

        if train_cfg is not None:
            bbox_head_init.update(train_cfg=train_cfg["init"])
        bbox_head_init.update(test_cfg=test_cfg)
        self.bbox_head_init = MODELS.build(bbox_head_init)
        self.num_refine_stages = len(bbox_head_refine)
        self.bbox_head_refine = ModuleList()
        for i, refine_head in enumerate(bbox_head_refine):
            if train_cfg is not None:
                refine_head.update(train_cfg=train_cfg["refine"][i])
            refine_head.update(test_cfg=test_cfg)
            self.bbox_head_refine.append(MODELS.build(refine_head))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward(
        self,
        inputs: Tensor,
        inputs2: Tensor,
        data_samples: OptSampleList = None,
        mode: str = "tensor",
    ) -> ForwardResults:
        if mode == "loss":
            return self.loss(inputs, inputs2, data_samples)
        elif mode == "predict":
            return self.predict(inputs, inputs2, data_samples)
        elif mode == "tensor":
            return self._forward(inputs, inputs2, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". ' "Only supports loss, predict and tensor mode")

    def loss(self, batch_inputs: Tensor, batch_inputs2: Tensor, batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        losses = dict()
        x, x_vi = self.extract_feat2(batch_inputs, batch_inputs2)

        outs = self.bbox_head_init(x)
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas) = outputs
        loss_inputs = outs + (batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        init_losses = self.bbox_head_init.loss_by_feat(*loss_inputs)
        keys = init_losses.keys()
        for key in list(keys):
            if "loss" in key and "init" not in key:
                init_losses[f"{key}_init"] = init_losses.pop(key)
        losses.update(init_losses)

        rois = self.bbox_head_init.filter_bboxes(*outs)
        for i in range(self.num_refine_stages):
            weight = self.train_cfg.stage_loss_weights[i]
            x_refine = self.bbox_head_refine[i].feature_refine(x, rois)
            outs = self.bbox_head_refine[i](x_refine)
            loss_inputs = outs + (batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
            refine_losses = self.bbox_head_refine[i].loss_by_feat(*loss_inputs, rois=rois)
            keys = refine_losses.keys()
            for key in list(keys):
                if "loss" in key and "refine" not in key:
                    loss = refine_losses.pop(key)
                    if isinstance(loss, Sequence):
                        loss = [item * weight for item in loss]
                    else:
                        loss = loss * weight
                    refine_losses[f"{key}_refine_{i}"] = loss
            losses.update(refine_losses)

            if i + 1 in range(self.num_refine_stages):
                rois = self.bbox_head_refine[i].refine_bboxes(*outs, rois=rois)

        # aux task
        if self.aux_rpn_task:
            proposal_cfg = ConfigDict(
                nms_pre=2000,
                max_per_img=2000,
                nms=ConfigDict(type="nms", iou_threshold=0.8),
                min_bbox_size=0,
            )
            rpn_data_samples2 = copy.deepcopy(batch_data_samples)
            for data in rpn_data_samples2:
                data.gt_instances = data.gt_instances2
                data.gt_instances.labels = torch.zeros_like(data.gt_instances.labels)

            aux_losses, _ = self.aux_rpn_head.loss_and_predict(x_vi, rpn_data_samples2, proposal_cfg=proposal_cfg)

            aux_losses = self.parse_aux_losses(aux_losses)
            losses.update(aux_losses)

        return losses

    def predict(self, batch_inputs: Tensor, batch_inputs2: Tensor, batch_data_samples: SampleList, rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 5),
              the last dimension 5 arrange as (x, y, w, h, t).
        """
        x = self.extract_feat(batch_inputs, batch_inputs2)
        outs = self.bbox_head_init(x)
        rois = self.bbox_head_init.filter_bboxes(*outs)
        for i in range(self.num_refine_stages):
            x_refine = self.bbox_head_refine[i].feature_refine(x, rois)
            outs = self.bbox_head_refine[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.bbox_head_refine[i].refine_bboxes(*outs, rois)

        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        predictions = self.bbox_head_refine[-1].predict_by_feat(*outs, rois=rois, batch_img_metas=batch_img_metas, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, predictions)
        return batch_data_samples

    def _forward(self, batch_inputs: Tensor, batch_inputs2: Tensor, batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs, batch_inputs2)
        outs = self.bbox_head_init(x)
        rois = self.bbox_head_init.filter_bboxes(*outs)
        for i in range(self.num_refine_stages):
            x_refine = self.bbox_head_refine[i].feature_refine(x, rois)
            outs = self.bbox_head_refine[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.bbox_head_refine[i].refine_bboxes(*outs, rois)

        return outs

    def extract_feat(self, imgs_vi: Tensor, imgs_ir: Tensor) -> Tuple[Tensor]:
        x_vi = self.backbone_vi(imgs_vi)
        x_ir = self.backbone_ir(imgs_ir)

        if hasattr(self, "mtablock"):
            x_vi = self.mtablock(x_vi)

        if hasattr(self, "fusblock"):
            x = self.fusblock(x_vi, x_ir)
        else:
            x = tuple([x_vi[i] + x_ir[i] for i in range(len(x_vi))])

        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feat2(self, imgs_vi: Tensor, imgs_ir: Tensor) -> Tuple[Tensor]:
        x_vi = self.backbone_vi(imgs_vi)
        x_ir = self.backbone_ir(imgs_ir)

        if hasattr(self, "mtablock"):
            x_vi = self.mtablock(x_vi)

        if hasattr(self, "fusblock"):
            x = self.fusblock(x_vi, x_ir)
        else:
            x = tuple([x_vi[i] + x_ir[i] for i in range(len(x_vi))])

        if self.with_neck:
            x = self.neck(x)

        if self.aux_rpn_task:
            x_vi = self.aux_neck(x_vi)
        return x, x_vi

    def parse_aux_losses(self, losses: Dict[str, torch.Tensor]):
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(value for key, value in log_vars if "loss" in key)
        return {"loss_aux": loss}
