#!/usr/bin/env python3
"""Interactively edit and save a zero-pose SMPL Neutral body shape."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from rgbd_avatar.avatar.shape_preset import (
    SMPLShapePreset,
    load_shape_preset,
    save_shape_preset,
)
from rgbd_avatar.avatar.smpl_sequence import sha256_file, smpl_to_display


LOGGER = logging.getLogger("edit_smpl_shape")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = PROJECT_ROOT / "assets/models/smpl/SMPL_NEUTRAL_CLEAN.pkl"
DEFAULT_PRESET = PROJECT_ROOT / "assets/models/smpl/presets/custom_shape.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--load-preset",
        type=Path,
        default=None,
        help="Resume from a previously saved SMPL shape JSON.",
    )
    parser.add_argument("--output-preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument(
        "--output-mesh",
        type=Path,
        default=None,
        help="Static zero-pose preview mesh; default: preset path with .ply.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--beta-limit", type=float, default=3.0)
    parser.add_argument("--scale-min", type=float, default=0.75)
    parser.add_argument("--scale-max", type=float, default=1.30)
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=900)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load the model and generate one shape without opening a window.",
    )
    parser.add_argument(
        "--save-and-exit",
        action="store_true",
        help="Save the initial or loaded preset without opening a window.",
    )
    return parser.parse_args()


def _resolve_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
    return requested


def generate_shape(
    model: Any,
    torch: Any,
    *,
    betas: np.ndarray,
    scale: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate native vertices/joints and a centered, grounded display mesh."""

    beta_values = np.asarray(betas, dtype=np.float32)
    if beta_values.shape != (int(model.num_betas),):
        raise ValueError(
            f"Expected {model.num_betas} betas, got {beta_values.shape}."
        )
    if not np.isfinite(beta_values).all():
        raise ValueError("SMPL betas must be finite.")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("SMPL scale must be finite and positive.")
    with torch.no_grad():
        output = model(
            betas=torch.as_tensor(
                beta_values, dtype=torch.float32, device=device
            ).reshape(1, -1),
            body_pose=torch.zeros((1, 69), dtype=torch.float32, device=device),
            global_orient=torch.zeros((1, 3), dtype=torch.float32, device=device),
            return_verts=True,
        )
    vertices_native = (
        float(scale) * output.vertices[0].detach().cpu().numpy()
    ).astype(np.float32)
    joints_native = (
        float(scale) * output.joints[0, :24].detach().cpu().numpy()
    ).astype(np.float32)
    vertices_display = smpl_to_display(vertices_native).astype(np.float64)
    vertices_display[:, 0] -= float(np.mean(vertices_display[:, 0]))
    vertices_display[:, 1] -= float(np.mean(vertices_display[:, 1]))
    vertices_display[:, 2] -= float(np.min(vertices_display[:, 2]))
    return vertices_native, joints_native, vertices_display.astype(np.float32)


def _open3d_mesh(
    o3d: Any,
    vertices_display: np.ndarray,
    faces: np.ndarray,
) -> Any:
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices_display.astype(np.float64)),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((0.68, 0.72, 0.76))
    return mesh


def _atomic_write_mesh(o3d: Any, path: Path, mesh: Any) -> None:
    output = path.expanduser().resolve()
    if output.suffix.lower() not in {".ply", ".obj"}:
        raise ValueError("--output-mesh must end in .ply or .obj.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.stem}.",
            suffix=output.suffix,
            dir=output.parent,
            delete=False,
        ) as file:
            temporary = Path(file.name)
        written = o3d.io.write_triangle_mesh(
            str(temporary),
            mesh,
            write_ascii=False,
            compressed=False,
            write_vertex_normals=True,
            write_vertex_colors=True,
            write_triangle_uvs=False,
            print_progress=False,
        )
        if not written:
            raise RuntimeError(f"Open3D failed to write mesh: {output}")
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class SMPLShapeEditor:
    """Open3D GUI with live beta and uniform-scale controls."""

    GEOMETRY_NAME = "smpl_neutral_shape"

    def __init__(
        self,
        *,
        o3d: Any,
        gui: Any,
        rendering: Any,
        model: Any,
        torch: Any,
        device: str,
        model_path: Path,
        model_digest: str,
        faces: np.ndarray,
        initial_betas: np.ndarray,
        initial_scale: float,
        output_preset: Path,
        output_mesh: Path,
        beta_limit: float,
        scale_limits: tuple[float, float],
        window_size: tuple[int, int],
    ) -> None:
        self.o3d = o3d
        self.gui = gui
        self.rendering = rendering
        self.model = model
        self.torch = torch
        self.device = device
        self.model_path = model_path
        self.model_digest = model_digest
        self.faces = np.asarray(faces, dtype=np.int32)
        self.betas = np.asarray(initial_betas, dtype=np.float32).copy()
        self.scale = float(initial_scale)
        self.output_preset = output_preset
        self.output_mesh = output_mesh
        self.beta_limit = float(beta_limit)
        self.scale_limits = scale_limits
        self._updating_controls = False
        self._current_mesh: Any | None = None

        app = gui.Application.instance
        self.window = app.create_window(
            "SMPL Neutral 体型编辑器",
            int(window_size[0]),
            int(window_size[1]),
        )
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background(np.array([0.075, 0.08, 0.09, 1.0]))
        self.scene.scene.show_axes(True)

        em = self.window.theme.font_size
        self.panel = gui.Vert(0.35 * em, gui.Margins(em, em, em, em))
        title = gui.Label("SMPL Neutral · Zero Pose")
        self.panel.add_child(title)
        self.panel.add_child(
            gui.Label("拖动滑条改变体型；鼠标旋转、缩放和平移视角。")
        )
        self.dimensions_label = gui.Label("")
        self.panel.add_child(self.dimensions_label)

        self.beta_sliders: list[Any] = []
        self.beta_value_labels: list[Any] = []
        for index, initial in enumerate(self.betas):
            row = gui.Horiz(0.25 * em)
            name = gui.Label(f"β{index}")
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(-self.beta_limit, self.beta_limit)
            slider.double_value = float(initial)
            value_label = gui.Label(f"{float(initial):+.2f}")
            slider.set_on_value_changed(
                lambda value, beta_index=index: self._on_beta_changed(
                    beta_index, value
                )
            )
            row.add_child(name)
            row.add_child(slider)
            row.add_child(value_label)
            self.panel.add_child(row)
            self.beta_sliders.append(slider)
            self.beta_value_labels.append(value_label)

        scale_row = gui.Horiz(0.25 * em)
        scale_row.add_child(gui.Label("整体尺度"))
        self.scale_slider = gui.Slider(gui.Slider.DOUBLE)
        self.scale_slider.set_limits(*self.scale_limits)
        self.scale_slider.double_value = self.scale
        self.scale_value_label = gui.Label(f"{self.scale:.3f}")
        self.scale_slider.set_on_value_changed(self._on_scale_changed)
        scale_row.add_child(self.scale_slider)
        scale_row.add_child(self.scale_value_label)
        self.panel.add_child(scale_row)

        button_row = gui.Horiz(0.5 * em)
        reset_button = gui.Button("恢复 Neutral")
        reset_button.set_on_clicked(self._on_reset)
        save_button = gui.Button("保存参数与网格")
        save_button.set_on_clicked(self._on_save)
        button_row.add_child(reset_button)
        button_row.add_child(save_button)
        self.panel.add_child(button_row)
        self.status_label = gui.Label(f"保存位置：{self.output_preset}")
        self.panel.add_child(self.status_label)

        self.window.add_child(self.scene)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)
        self._refresh_geometry(reset_camera=True)

    def _on_layout(self, layout_context: Any) -> None:
        rect = self.window.content_rect
        panel_width = min(430, max(330, int(rect.width * 0.34)))
        self.scene.frame = self.gui.Rect(
            rect.x, rect.y, rect.width - panel_width, rect.height
        )
        self.panel.frame = self.gui.Rect(
            rect.x + rect.width - panel_width,
            rect.y,
            panel_width,
            rect.height,
        )

    def _on_beta_changed(self, index: int, value: float) -> None:
        if self._updating_controls:
            return
        self.betas[index] = float(value)
        self.beta_value_labels[index].text = f"{float(value):+.2f}"
        self._refresh_geometry()

    def _on_scale_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        self.scale = float(value)
        self.scale_value_label.text = f"{self.scale:.3f}"
        self._refresh_geometry()

    def _on_reset(self) -> None:
        self._updating_controls = True
        try:
            self.betas.fill(0.0)
            self.scale = 1.0
            for slider, label in zip(
                self.beta_sliders, self.beta_value_labels
            ):
                slider.double_value = 0.0
                label.text = "+0.00"
            self.scale_slider.double_value = self.scale
            self.scale_value_label.text = f"{self.scale:.3f}"
        finally:
            self._updating_controls = False
        self._refresh_geometry(reset_camera=True)
        self.status_label.text = "已恢复 Neutral 平均体型，尚未保存。"

    def _refresh_geometry(self, *, reset_camera: bool = False) -> None:
        _, _, display_vertices = generate_shape(
            self.model,
            self.torch,
            betas=self.betas,
            scale=self.scale,
            device=self.device,
        )
        mesh = _open3d_mesh(self.o3d, display_vertices, self.faces)
        if self.scene.scene.has_geometry(self.GEOMETRY_NAME):
            self.scene.scene.remove_geometry(self.GEOMETRY_NAME)
        material = self.rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.base_color = (0.68, 0.72, 0.76, 1.0)
        # Open3D 0.18 exposes the scalar PBR value as base_roughness.
        material.base_roughness = 0.78
        self.scene.scene.add_geometry(self.GEOMETRY_NAME, mesh, material)
        self._current_mesh = mesh
        extents = np.ptp(display_vertices, axis=0)
        self.dimensions_label.text = (
            f"高 {extents[2]:.3f} m  宽 {extents[0]:.3f} m  "
            f"厚 {extents[1]:.3f} m"
        )
        if reset_camera:
            bounds = mesh.get_axis_aligned_bounding_box()
            self.scene.setup_camera(60.0, bounds, bounds.get_center())

    def _on_save(self) -> None:
        try:
            if self._current_mesh is None:
                raise RuntimeError("No generated SMPL mesh is available.")
            _atomic_write_mesh(self.o3d, self.output_mesh, self._current_mesh)
            preset = SMPLShapePreset(
                model_path=str(self.model_path),
                model_sha256=self.model_digest,
                betas=self.betas,
                scale=self.scale,
            )
            save_shape_preset(
                self.output_preset,
                preset,
                mesh_path=str(self.output_mesh),
            )
            self.status_label.text = (
                f"保存成功：{self.output_preset.name} / "
                f"{self.output_mesh.name}"
            )
            LOGGER.info("Saved SMPL shape preset: %s", self.output_preset)
            LOGGER.info("Saved grounded preview mesh: %s", self.output_mesh)
        except Exception as error:
            LOGGER.exception("Failed to save SMPL shape")
            self.status_label.text = f"保存失败：{error}"


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        if args.beta_limit <= 0:
            raise ValueError("--beta-limit must be positive.")
        if not 0 < args.scale_min < args.scale_max:
            raise ValueError("Scale limits must satisfy 0 < min < max.")
        if args.window_width < 640 or args.window_height < 600:
            raise ValueError("Editor window is too small.")
        model_path = args.model.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"SMPL model not found: {model_path}")
        output_preset = args.output_preset.expanduser().resolve()
        output_mesh = (
            args.output_mesh.expanduser().resolve()
            if args.output_mesh is not None
            else output_preset.with_suffix(".ply")
        )

        import smplx
        import torch

        device = _resolve_device(args.device, torch)
        model = smplx.SMPL(
            str(model_path),
            batch_size=1,
            create_betas=False,
            create_body_pose=False,
            create_global_orient=False,
            create_transl=False,
        ).to(device).eval()
        model_digest = sha256_file(model_path)
        betas = np.zeros(int(model.num_betas), dtype=np.float32)
        scale = 1.0
        if args.load_preset is not None:
            preset = load_shape_preset(args.load_preset)
            if preset.model_sha256 != model_digest:
                raise ValueError(
                    "Shape preset was created for a different SMPL model."
                )
            if preset.betas.shape != betas.shape:
                raise ValueError(
                    f"Preset has {len(preset.betas)} betas, model expects "
                    f"{len(betas)}."
                )
            betas = preset.betas.copy()
            scale = preset.scale
        if not args.scale_min <= scale <= args.scale_max:
            raise ValueError("Initial preset scale is outside slider limits.")

        vertices, joints, display_vertices = generate_shape(
            model,
            torch,
            betas=betas,
            scale=scale,
            device=device,
        )
        LOGGER.info(
            "Loaded %s on %s: vertices=%d joints=%d betas=%d height=%.3f m",
            model_path,
            device,
            len(vertices),
            len(joints),
            len(betas),
            float(np.ptp(display_vertices[:, 2])),
        )
        if args.validate_only:
            LOGGER.info("SMPL shape editor validation complete.")
            return 0

        import open3d as o3d
        if args.save_and_exit:
            mesh = _open3d_mesh(
                o3d,
                display_vertices,
                np.asarray(model.faces, dtype=np.int32),
            )
            _atomic_write_mesh(o3d, output_mesh, mesh)
            save_shape_preset(
                output_preset,
                SMPLShapePreset(
                    model_path=str(model_path),
                    model_sha256=model_digest,
                    betas=betas,
                    scale=scale,
                ),
                mesh_path=str(output_mesh),
            )
            LOGGER.info("Saved SMPL shape preset: %s", output_preset)
            LOGGER.info("Saved grounded preview mesh: %s", output_mesh)
            return 0

        from open3d.visualization import gui, rendering

        gui.Application.instance.initialize()
        try:
            SMPLShapeEditor(
                o3d=o3d,
                gui=gui,
                rendering=rendering,
                model=model,
                torch=torch,
                device=device,
                model_path=model_path,
                model_digest=model_digest,
                faces=np.asarray(model.faces, dtype=np.int32),
                initial_betas=betas,
                initial_scale=scale,
                output_preset=output_preset,
                output_mesh=output_mesh,
                beta_limit=args.beta_limit,
                scale_limits=(args.scale_min, args.scale_max),
                window_size=(args.window_width, args.window_height),
            )
        except Exception:
            gui.Application.instance.quit()
            raise
        gui.Application.instance.run()
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
