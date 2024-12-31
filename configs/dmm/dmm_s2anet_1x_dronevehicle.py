_base_ = [
    "../_base_/datasets/DroneVehicle.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

vmamba_pretained = "weights/vssm/vssm_tiny_0230_ckpt_epoch_262.pth"


angle_version = "le135"
model = dict(
    type="virRefineSingleStageDetector",
    data_preprocessor=dict(
        type="mmdet.DetRGBTDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        mean2=[123.675, 116.28, 103.53],
        std2=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False,
    ),
    backbone_vi=dict(
        type="mmdet.MM_VSSM",
        out_indices=(0, 1, 2, 3),
        pretrained=vmamba_pretained,
        dims=96,
        depths=(2, 2, 5, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz",
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.2,
    ),
    backbone_ir=dict(
        type="mmdet.MM_VSSM",
        out_indices=(0, 1, 2, 3),
        pretrained=vmamba_pretained,
        dims=96,
        depths=(2, 2, 5, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz",
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.2,
    ),
    fusblock=dict(
        type="DCFModule",
        in_channels=[96, 192, 384, 768],
        drop_path=0.2,
        ssm_ratio=2.0,
        d_state=16,
        dt_rank="auto",
        mlp_ratio=0.0,
        attn_drop_rate=0.0,
        mlp_drop=0.0,
        disable_z=False,
    ),
    neck=dict(
        type="mmdet.FPN",
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_input",
        num_outs=5,
    ),
    mtablock=dict(type="MTAttentionBlock", in_channels=[96, 192, 384, 768]),
    # aux_neck=dict(
    #     type="mmdet.FPN",
    #     in_channels=[96, 192, 384, 768],
    #     out_channels=256,
    #     num_outs=5,
    #     init_cfg=dict(type="Pretrained", checkpoint=None, prefix="neck."),
    # ),
    # aux_rpn_head=dict(
    #     type="mmdet.RPNHead",
    #     num_classes=1,
    #     in_channels=256,
    #     feat_channels=256,
    #     anchor_generator=dict(
    #         type="mmdet.AnchorGenerator",
    #         scales=[8],
    #         ratios=[0.5, 1.0, 2.0],
    #         strides=[4, 8, 16, 32, 64],
    #         use_box_type=True,
    #     ),
    #     bbox_coder=dict(
    #         type="DeltaXYWHHBBoxCoder",
    #         target_means=[0.0, 0.0, 0.0, 0.0],
    #         target_stds=[1.0, 1.0, 1.0, 1.0],
    #         use_box_type=True,
    #     ),
    #     loss_cls=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
    #     loss_bbox=dict(type="mmdet.SmoothL1Loss", beta=0.1111111111111111, loss_weight=1.0),
    #     train_cfg=dict(
    #         assigner=dict(
    #             type="mmdet.MaxIoUAssigner",
    #             pos_iou_thr=0.7,
    #             neg_iou_thr=0.3,
    #             min_pos_iou=0.3,
    #             match_low_quality=True,
    #             ignore_iof_thr=-1,
    #             iou_calculator=dict(type="RBbox2HBboxOverlaps2D"),
    #         ),
    #         sampler=dict(type="mmdet.RandomSampler", num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
    #         allowed_border=0,
    #         pos_weight=-1,
    #         debug=False,
    #     ),
    #     test_cfg=dict(nms_pre=2000, max_per_img=2000, nms=dict(type="nms", iou_threshold=0.8), min_bbox_size=0),
    #     init_cfg=dict(type="Pretrained", checkpoint=None, prefix="rpn_head."),
    # ),
    bbox_head_init=dict(
        type="S2AHead",
        num_classes=5,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        anchor_generator=dict(
            type="FakeRotatedAnchorGenerator",
            angle_version=angle_version,
            scales=[4],
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type="DeltaXYWHTRBBoxCoder",
            angle_version=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
            use_box_type=False,
        ),
        loss_cls=dict(type="mmdet.FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="mmdet.SmoothL1Loss", beta=0.11, loss_weight=1.0),
    ),
    bbox_head_refine=[
        dict(
            type="S2ARefineHead",
            num_classes=5,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            frm_cfg=dict(type="AlignConv", feat_channels=256, kernel_size=3, strides=[8, 16, 32, 64, 128]),
            anchor_generator=dict(type="PseudoRotatedAnchorGenerator", strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type="DeltaXYWHTRBBoxCoder",
                angle_version=angle_version,
                norm_factor=1,
                edge_swap=False,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
            ),
            loss_cls=dict(type="mmdet.FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type="mmdet.SmoothL1Loss", beta=0.11, loss_weight=1.0),
        )
    ],
    train_cfg=dict(
        init=dict(
            assigner=dict(
                type="mmdet.MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type="RBboxOverlaps2D"),
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        refine=[
            dict(
                assigner=dict(
                    type="mmdet.MaxIoUAssigner",
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type="RBboxOverlaps2D"),
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            )
        ],
        stage_loss_weights=[1.0],
    ),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms_rotated", iou_threshold=0.1),
        max_per_img=2000,
    ),
)


default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=2, save_best="auto", type="CheckpointHook"))

max_epochs = 12
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)

# learning rate
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type="MultiStepLR", begin=0, end=max_epochs, by_epoch=True, milestones=[8, 11], gamma=0.1),
]
# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(_delete_=True, type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
)
