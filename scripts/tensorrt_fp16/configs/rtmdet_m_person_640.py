"""Exact RTMDet-M person detector used by MMPoseInferencer ``auto``.

This local copy deliberately disables the ImageNet backbone initializer.  The
full detector checkpoint is loaded by the exporter, so fetching a second
pretrained checkpoint would be redundant and would break offline export.
"""

_base_ = "mmdet::rtmdet/rtmdet_m_8xb32-300e_coco.py"

model = dict(
    backbone=dict(init_cfg=None),
    bbox_head=dict(num_classes=1),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=100,
    ),
)

train_dataloader = dict(
    dataset=dict(metainfo=dict(classes=("person",)))
)
val_dataloader = dict(
    dataset=dict(metainfo=dict(classes=("person",)))
)
test_dataloader = val_dataloader
