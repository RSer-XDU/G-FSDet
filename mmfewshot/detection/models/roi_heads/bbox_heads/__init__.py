# Copyright (c) OpenMMLab. All rights reserved.
from .contrastive_bbox_head import ContrastiveBBoxHead
from .cosine_sim_bbox_head import CosineSimBBoxHead, DisCosSimBBoxHead
from .meta_bbox_head import MetaBBoxHead
from .multi_relation_bbox_head import MultiRelationBBoxHead
from .two_branch_bbox_head import TwoBranchBBoxHead

from .kd_bbox_head import DisKDBBoxHead
__all__ = [
    'CosineSimBBoxHead', 'ContrastiveBBoxHead', 'MultiRelationBBoxHead',
    'MetaBBoxHead', 'TwoBranchBBoxHead',

     'DisKDBBoxHead', 'DisCosSimBBoxHead'
]
