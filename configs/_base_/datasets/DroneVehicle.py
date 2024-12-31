# dataset settings
dataset_type = "mmrotate.DVDataset"
data_root = "data/DV_cropped"  #
backend_args = None

train_pipeline = [
    dict(type="mmrotate.LoadRGBTImageFromFile", backend_args=backend_args),
    dict(type="mmrotate.LoadRGBTAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox", gt_bboxes2="rbox")),
    dict(type="mmrotate.ResizeRGBT", scale=(640, 512), keep_ratio=True),
    dict(
        type="mmrotate.RandomFlipRGBT",
        prob=0.5,  #
        direction=["horizontal", "vertical", "diagonal"],
    ),
    dict(type="mmdet.FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type="mmrotate.PackRGBTDetInputs",
        meta_keys=("img_id", "img_path", "img_path2", "ori_shape", "img_shape", "scale_factor"),
    ),
]
val_pipeline = [
    dict(type="mmrotate.LoadRGBTImageFromFile", backend_args=backend_args),
    dict(type="mmrotate.LoadRGBTAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox", gt_bboxes2="rbox")),
    dict(type="mmrotate.ResizeRGBT", scale=(640, 512), keep_ratio=True),
    # dict(type="mmdet.FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type="mmrotate.PackRGBTDetInputs",
        meta_keys=("img_id", "img_path", "img_path2", "ori_shape", "img_shape", "scale_factor"),
    ),
]
test_pipeline = [
    dict(type="mmrotate.LoadRGBTImageFromFile", backend_args=backend_args),
    dict(type="mmrotate.ResizeRGBT", scale=(640, 512), keep_ratio=True),
    dict(
        type="mmrotate.PackRGBTDetInputs",
        meta_keys=("img_id", "img_path", "img_path2", "ori_shape", "img_shape", "scale_factor"),
    ),
]


train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="train/annofiles_ir",  #
        aux_ann_file="train/annofiles_vi",  #
        data_prefix=dict(
            img_path="train/image_vi",
            img_path2="train/image_ir",
        ),  #
        img_suffix="png",
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="val/annofiles_ir",  #
        aux_ann_file="val/annofiles_vi",  #
        data_prefix=dict(
            img_path="val/image_vi",
            img_path2="val/image_ir",
        ),  #
        img_suffix="png",
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # indices=100,
        ann_file="test/annofiles_ir",
        aux_ann_file="test/annofiles_vi",  #
        data_prefix=dict(
            img_path="test/image_vi",
            img_path2="test/image_ir",
        ),  #
        img_suffix="png",
        test_mode=True,
        pipeline=val_pipeline,
    ),
)

val_evaluator = dict(type="DOTAMetric", metric="mAP")
test_evaluator = val_evaluator

# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(img_path='other4/images'),
#         test_mode=True,
#         pipeline=test_pipeline))
#
# test_evaluator = dict(
#     type='DOTAMetric',
#     format_only=True,
#     merge_patches=False,
#     outfile_prefix='./outputs/dotav2/'
# )
