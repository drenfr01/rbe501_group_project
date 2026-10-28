"""Tests for the pick-and-lift MuJoCo model: finger collision hulls and the resized cube."""

import mujoco
import numpy as np
import pytest

from rbe_501_so101.config import GripperConfig, ObjectConfig, SimConfig
from rbe_501_so101.envs.grasp_model import FIXED_PAD_GEOM, MOVING_PAD_GEOM, build_grasp_model


@pytest.fixture(scope="module")
def grasp_model() -> mujoco.MjModel:
    """Pick-and-lift model built once for this module."""
    return build_grasp_model(SimConfig(), GripperConfig(), ObjectConfig())


def _mesh_geom(model: mujoco.MjModel, mesh_name: str) -> int:
    """ID of the first geom that renders `mesh_name`."""
    mesh_id = model.mesh(mesh_name).id
    return next(g for g in range(model.ngeom) if model.geom_dataid[g] == mesh_id)


def _collides(model: mujoco.MjModel, geom: int) -> bool:
    """True if the geom takes part in contacts."""
    return bool(model.geom_contype[geom] or model.geom_conaffinity[geom])


def test_finger_pads_collide_and_sit_on_the_jaw_bodies(grasp_model):
    fixed, moving = grasp_model.geom(FIXED_PAD_GEOM), grasp_model.geom(MOVING_PAD_GEOM)
    assert _collides(grasp_model, fixed.id) and _collides(grasp_model, moving.id)
    assert grasp_model.body(int(fixed.bodyid[0])).name == "gripper_link"
    assert grasp_model.body(int(moving.bodyid[0])).name == "moving_jaw_so101_v1_link"


def test_whole_jaw_meshes_no_longer_collide(grasp_model):
    cfg = GripperConfig()
    for mesh in (cfg.fixed_jaw_mesh, cfg.moving_jaw_mesh):
        assert not _collides(grasp_model, _mesh_geom(grasp_model, mesh))


def test_cube_is_resized_to_configured_side(grasp_model):
    side = ObjectConfig().side_m
    assert side == 0.025
    np.testing.assert_allclose(grasp_model.geom(ObjectConfig().geom).size, side / 2)


def test_finger_pads_do_not_change_link_masses(grasp_model, raw_model):
    for body in ("gripper_link", "moving_jaw_so101_v1_link"):
        assert grasp_model.body(body).mass == pytest.approx(raw_model.body(body).mass)


def test_contacts_use_elliptic_cone_and_raised_impratio(grasp_model, raw_model):
    assert grasp_model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert grasp_model.opt.impratio > raw_model.opt.impratio


def test_reach_target_marker_is_hidden(grasp_model):
    site = grasp_model.site(SimConfig().target_site)
    assert site.rgba[3] == 0.0


def test_mjcf_on_disk_is_unchanged(raw_model):
    cfg = GripperConfig()
    assert _collides(raw_model, _mesh_geom(raw_model, cfg.fixed_jaw_mesh))
    with pytest.raises(KeyError):
        raw_model.geom(FIXED_PAD_GEOM)
