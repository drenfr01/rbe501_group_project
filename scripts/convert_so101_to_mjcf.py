from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]

URDF_PATH = (
    PROJECT_ROOT
    / "models"
    / "so101"
    / "so101_new_calib.urdf"
)

MJCF_PATH = (
    PROJECT_ROOT
    / "models"
    / "so101"
    / "so101.xml"
)


print("Loading:")
print(URDF_PATH)

model = mujoco.MjModel.from_xml_path(str(URDF_PATH))

print()
print("Saving MuJoCo model to:")
print(MJCF_PATH)

mujoco.mj_saveLastXML(
    str(MJCF_PATH),
    model
)

print()
print("Conversion complete.")