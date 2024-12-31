from numbers import Number
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseDataPreprocessor
from mmengine.utils import is_seq_of
from mmengine.model.utils import stack_batch

from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.registry import MODELS


@MODELS.register_module()
class RGBTDataPreprocessor(BaseDataPreprocessor):

    def __init__(
        self,
        mean: Optional[Sequence[Union[float, int]]] = None,
        std: Optional[Sequence[Union[float, int]]] = None,
        mean2: Optional[Sequence[Union[float, int]]] = None,
        std2: Optional[Sequence[Union[float, int]]] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        non_blocking: Optional[bool] = False,
    ):
        super().__init__(non_blocking)
        assert not (bgr_to_rgb and rgb_to_bgr), "`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time"
        assert (mean is None) == (std is None), "mean and std should be both None or tuple"

        if mean is not None:
            assert len(mean) == 3 or len(mean) == 1, "`mean` should have 1 or 3 values, to be compatible with " f"RGB or gray image, but got {len(mean)} values"
            assert len(std) == 3 or len(std) == 1, (  # type: ignore
                "`std` should have 1 or 3 values, to be compatible with RGB "  # type: ignore # noqa: E501
                f"or gray image, but got {len(std)} values"
            )  # type: ignore
            self._enable_normalize = True
            self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer("std", torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        if mean2 is not None:
            assert len(mean2) == 3 or len(mean2) == 1, "`mean2` should have 1 or 3 values, to be compatible with " f"RGB or gray image, but got {len(mean2)} values"
            assert len(std2) == 3 or len(std2) == 1, (  # type: ignore
                "`std2` should have 1 or 3 values, to be compatible with RGB "  # type: ignore # noqa: E501
                f"or gray image, but got {len(std2)} values"
            )  # type: ignore
            self._enable_normalize2 = True
            self.register_buffer("mean2", torch.tensor(mean2).view(-1, 1, 1), False)
            self.register_buffer("std2", torch.tensor(std2).view(-1, 1, 1), False)
        else:
            self._enable_normalize2 = False

        self._channel_conversion = rgb_to_bgr or bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        data = self.cast_data(data)  # type: ignore
        _batch_inputs = data["inputs"]
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                # channel transform
                if self._channel_conversion:
                    _batch_input = _batch_input[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_input = _batch_input.float()
                # Normalization.
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert _batch_input.dim() == 3 and _batch_input.shape[0] == 3, (
                            "If the mean has 3 values, the input tensor " "should in shape of (3, H, W), but got the tensor " f"with shape {_batch_input.shape}"
                        )
                    _batch_input = (_batch_input - self.mean) / self.std
                else:
                    _batch_input = _batch_input / 255.0
                batch_inputs.append(_batch_input)
            # Pad and stack Tensor.
            batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor, self.pad_value)
        data["inputs"] = batch_inputs

        if data.get("inputs2", None) is not None:
            _batch_inputs = data["inputs2"]
            if is_seq_of(_batch_inputs, torch.Tensor):
                batch_inputs = []
                for _batch_input in _batch_inputs:
                    # Convert to float after channel conversion to ensure
                    # efficiency
                    _batch_input = _batch_input.float()
                    # Normalization.
                    if self._enable_normalize2:
                        if self.mean2.shape[0] == 3:
                            assert _batch_input.dim() == 3 and _batch_input.shape[0] == 3, (
                                "If the mean has 3 values, the input tensor " "should in shape of (3, H, W), but got the tensor " f"with shape {_batch_input.shape}"
                            )
                        _batch_input = (_batch_input - self.mean2) / self.std2
                    else:
                        _batch_input = _batch_input / 255.0
                    batch_inputs.append(_batch_input)
                # Pad and stack Tensor.
                batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor, self.pad_value)
            data["inputs2"] = batch_inputs
        data.setdefault("data_samples", None)
        return data


@MODELS.register_module()
class DetRGBTDataPreprocessor(RGBTDataPreprocessor):

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        mean2: Sequence[Number] = None,
        std2: Sequence[Number] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        boxtype2tensor: bool = True,
        non_blocking: Optional[bool] = False,
        batch_augments: Optional[List[dict]] = None,
    ):
        super().__init__(
            mean=mean,
            std=std,
            mean2=mean2,
            std2=std2,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking,
        )
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList([MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None

        self.boxtype2tensor = boxtype2tensor

    def forward(self, data: dict, training: bool = False) -> dict:
        batch_pad_shape = self._get_pad_shape(data)
        data = super().forward(data=data, training=training)
        inputs, inputs2, data_samples = (
            data["inputs"],
            data["inputs2"],
            data["data_samples"],
        )

        if data_samples is not None:
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({"batch_input_shape": batch_input_shape, "pad_shape": pad_shape})

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {"inputs": inputs, "inputs2": inputs2, "data_samples": data_samples}

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs = data["inputs"]
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                pad_h = int(np.ceil(ori_input.shape[1] / self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(np.ceil(ori_input.shape[2] / self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                "The input of `ImgDataPreprocessor` should be a NCHW tensor " "or a list of tensor, but got a tensor with shape: " f"{_batch_inputs.shape}"
            )
            pad_h = int(np.ceil(_batch_inputs.shape[2] / self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(np.ceil(_batch_inputs.shape[3] / self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError("Output of `cast_data` should be a dict " "or a tuple with inputs and data_samples, but got" f"{type(data)}: {data}")
        return batch_pad_shape
