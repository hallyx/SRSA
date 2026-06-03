# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for keeping the play viewport camera relative to the SRSA socket."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import torch


DEFAULT_SOCKET_CAMERA_PROFILE = os.path.join("camera_profiles", "socket_camera_offset.json")


def _unwrap_isaac_env(env):
    current = env
    seen = set()
    for _ in range(16):
        if hasattr(current, "sim") and hasattr(current, "scene"):
            return current
        ident = id(current)
        if ident in seen:
            break
        seen.add(ident)
        next_env = getattr(current, "unwrapped", None)
        if next_env is not None and next_env is not current:
            current = next_env
            continue
        next_env = getattr(current, "env", None)
        if next_env is not None and next_env is not current:
            current = next_env
            continue
        next_env = getattr(current, "_env", None)
        if next_env is not None and next_env is not current:
            current = next_env
            continue
        break
    return current


def _tensor_to_list(value) -> list[float]:
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    return [float(item) for item in value]


def _vector_add(lhs: list[float], rhs: list[float]) -> list[float]:
    return [float(a) + float(b) for a, b in zip(lhs, rhs, strict=True)]


def _vector_sub(lhs: list[float], rhs: list[float]) -> list[float]:
    return [float(a) - float(b) for a, b in zip(lhs, rhs, strict=True)]


class SocketRelativeCameraController:
    """Save and apply a viewer camera offset relative to the current socket position."""

    def __init__(
        self,
        env,
        *,
        profile_path: str | None = None,
        calibrate: bool = False,
        follow: bool = False,
        capture_key: str = "C",
        env_index: int | None = None,
        camera_prim_path: str = "/OmniverseKit_Persp",
    ):
        self._env = _unwrap_isaac_env(env)
        self.profile_path = os.path.abspath(profile_path or DEFAULT_SOCKET_CAMERA_PROFILE)
        self.calibrate = bool(calibrate)
        self.follow = bool(follow)
        self.capture_key = str(capture_key or "C").upper()
        self.camera_prim_path = str(camera_prim_path or "/OmniverseKit_Persp")
        self.env_index = self._resolve_env_index(env_index)
        self._profile: dict[str, Any] | None = None
        self._missing_profile_warned = False
        self._keyboard_input = None
        self._keyboard = None
        self._keyboard_sub = None

        if self.follow:
            self._profile = self._load_profile(warn_if_missing=True)
        if self.calibrate:
            self._subscribe_keyboard()

    def close(self) -> None:
        if self._keyboard_sub is None or self._keyboard_input is None or self._keyboard is None:
            return
        try:
            self._keyboard_input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        except Exception as err:
            print(f"[WARN] Failed to unsubscribe socket camera keyboard handler: {err}")
        self._keyboard_sub = None

    def update(self, *, force: bool = False) -> None:
        if not self.follow:
            return
        if self._profile is None:
            self._profile = self._load_profile(warn_if_missing=force)
        if self._profile is None:
            return
        self.apply_profile(self._profile)

    def capture_and_save(self) -> None:
        socket_pos = self._get_socket_world_position()
        camera_eye = self._get_camera_eye_world_position()
        eye_offset = _vector_sub(camera_eye, socket_pos)
        profile = {
            "version": 1,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "assembly_id": os.environ.get("SRSA_ASSEMBLY_ID", ""),
            "env_index": int(self.env_index),
            "camera_prim_path": self.camera_prim_path,
            "socket_position_world": socket_pos,
            "camera_eye_world": camera_eye,
            "eye_offset_from_socket": eye_offset,
            "target_offset_from_socket": [0.0, 0.0, 0.0],
        }
        profile_dir = os.path.dirname(self.profile_path)
        if profile_dir:
            os.makedirs(profile_dir, exist_ok=True)
        with open(self.profile_path, "w", encoding="utf-8") as file:
            json.dump(profile, file, indent=2)
            file.write("\n")
        self._profile = profile
        self._missing_profile_warned = False
        print("[INFO] Saved socket-relative camera profile:")
        print(f"       profile: {self.profile_path}")
        print(f"       socket_world: {socket_pos}")
        print(f"       camera_world: {camera_eye}")
        print(f"       eye_offset_from_socket: {eye_offset}")

    def apply_profile(self, profile: dict[str, Any]) -> None:
        socket_pos = self._get_socket_world_position()
        eye_offset = [float(value) for value in profile.get("eye_offset_from_socket", [0.0, 0.0, 0.5])]
        target_offset = [float(value) for value in profile.get("target_offset_from_socket", [0.0, 0.0, 0.0])]
        camera_prim_path = str(profile.get("camera_prim_path") or self.camera_prim_path)
        eye = _vector_add(socket_pos, eye_offset)
        target = _vector_add(socket_pos, target_offset)
        self._env.sim.set_camera_view(eye=eye, target=target, camera_prim_path=camera_prim_path)

    def _resolve_env_index(self, env_index: int | None) -> int:
        if env_index is None:
            cfg = getattr(self._env, "cfg", None)
            viewer = getattr(cfg, "viewer", None)
            env_index = int(getattr(viewer, "env_index", 0))
        env_index = int(env_index)
        num_envs = int(getattr(self._env, "num_envs", 1))
        if env_index < 0 or env_index >= num_envs:
            raise ValueError(f"Socket camera env index {env_index} is out of range for {num_envs} envs.")
        return env_index

    def _load_profile(self, *, warn_if_missing: bool) -> dict[str, Any] | None:
        if not os.path.isfile(self.profile_path):
            if warn_if_missing and not self._missing_profile_warned:
                print(f"[WARN] Socket camera profile does not exist yet: {self.profile_path}")
                self._missing_profile_warned = True
            return None
        with open(self.profile_path, encoding="utf-8") as file:
            profile = json.load(file)
        if "eye_offset_from_socket" not in profile:
            raise ValueError(f"Socket camera profile missing 'eye_offset_from_socket': {self.profile_path}")
        print(f"[INFO] Loaded socket-relative camera profile: {self.profile_path}")
        return profile

    def _get_socket_world_position(self) -> list[float]:
        env_id = self.env_index
        if hasattr(self._env, "fixed_pos") and hasattr(self._env, "scene"):
            fixed_pos = self._env.fixed_pos[env_id]
            env_origin = self._env.scene.env_origins[env_id]
            return _tensor_to_list(fixed_pos + env_origin)
        if hasattr(self._env, "scene") and "fixed_asset" in self._env.scene.articulations:
            return _tensor_to_list(self._env.scene.articulations["fixed_asset"].data.root_pos_w[env_id])
        if hasattr(self._env, "_fixed_asset"):
            return _tensor_to_list(self._env._fixed_asset.data.root_pos_w[env_id])
        raise RuntimeError("Unable to resolve SRSA socket position from the environment.")

    def _get_camera_eye_world_position(self) -> list[float]:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.camera_prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Camera prim is not valid: {self.camera_prim_path}")
        xform = UsdGeom.Xformable(prim)
        transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = transform.ExtractTranslation()
        return [float(translation[0]), float(translation[1]), float(translation[2])]

    def _subscribe_keyboard(self) -> None:
        try:
            import carb.input
            import omni.appwindow

            app_window = omni.appwindow.get_default_app_window()
            if app_window is None:
                print("[WARN] No Omniverse app window; socket camera keyboard calibration is disabled.")
                return
            self._keyboard_input = carb.input.acquire_input_interface()
            self._keyboard = app_window.get_keyboard()
            self._keyboard_sub = self._keyboard_input.subscribe_to_keyboard_events(
                self._keyboard,
                self._on_keyboard_event,
            )
            print(
                "[INFO] Socket camera calibration enabled. "
                f"Move the viewport, then press '{self.capture_key}' to save {self.profile_path}."
            )
        except Exception as err:
            print(f"[WARN] Failed to enable socket camera keyboard calibration: {err}")

    def _on_keyboard_event(self, event, *args, **kwargs):
        import carb.input

        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name.upper() == self.capture_key:
            self.capture_and_save()
        return True
