"""Procedural avatar construction from metric pose joints."""

from .procedural import (
    CapsulePrimitive,
    EllipsoidPrimitive,
    ProceduralAvatarConfig,
    ProceduralAvatarFrame,
    build_procedural_avatar,
)
from .smpl_sequence import (
    DISPLAY_FROM_SMPL,
    HALPE_TO_SMPL,
    SMPLFitConfig,
    SMPLFrameFit,
    SMPLHandObservation,
    SMPLSequenceCache,
    SMPLSequenceFitter,
    SMPLTarget,
    build_smpl_target,
    display_to_smpl,
    estimate_smpl_scale,
    sha256_file,
    smpl_to_display,
)
from .mixamo_asset import MixamoAsset, load_mixamo_fbx
from .mixamo_gltf import (
    MixamoGlbSummary,
    build_mixamo_glb,
    export_mixamo_fbx_glb,
)
from .mixamo_sequence import MixamoSequenceCache, skin_mixamo_vertices
from .stick_figure import StickFigureConfig, build_stick_figure_avatar
from .shape_preset import (
    SMPLShapePreset,
    load_shape_preset,
    save_shape_preset,
)

__all__ = [
    "MixamoAsset",
    "MixamoGlbSummary",
    "MixamoSequenceCache",
    "CapsulePrimitive",
    "EllipsoidPrimitive",
    "ProceduralAvatarConfig",
    "ProceduralAvatarFrame",
    "build_procedural_avatar",
    "build_mixamo_glb",
    "StickFigureConfig",
    "build_stick_figure_avatar",
    "DISPLAY_FROM_SMPL",
    "HALPE_TO_SMPL",
    "SMPLFitConfig",
    "SMPLFrameFit",
    "SMPLHandObservation",
    "SMPLSequenceCache",
    "SMPLSequenceFitter",
    "SMPLTarget",
    "build_smpl_target",
    "display_to_smpl",
    "estimate_smpl_scale",
    "export_mixamo_fbx_glb",
    "sha256_file",
    "smpl_to_display",
    "SMPLShapePreset",
    "load_shape_preset",
    "load_mixamo_fbx",
    "save_shape_preset",
    "skin_mixamo_vertices",
]
