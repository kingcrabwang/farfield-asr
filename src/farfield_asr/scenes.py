"""可序列化场景与种子化抽样；几何与声学参数在仿真前检查。"""

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np

from farfield_asr.arrays import circular_array, linear_array


@dataclass(frozen=True)
class Scene:
    scene_id: str
    seed: int
    room_dimensions_m: tuple[float, float, float]
    array_geometry: str
    n_mics: int
    spacing_or_radius_m: float
    array_center_m: tuple[float, float, float]
    target_position_m: tuple[float, float, float]
    interference_position_m: tuple[float, float, float]
    rt60_s: float
    snr_db: float
    sir_db: float
    sample_rate: int = 16000
    sound_speed_m_s: float = 343.0
    reference_mic: int = 0

    def __post_init__(self):
        if self.array_geometry not in ("linear", "circular"):
            raise ValueError("array_geometry must be linear or circular")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not self.scene_id or "/" in self.scene_id or "\\" in self.scene_id:
            raise ValueError("invalid scene_id")
        if type(self.sample_rate) is not int or self.sample_rate < 8000:
            raise ValueError("sample_rate must be an integer >= 8000")
        parameters = [self.rt60_s, self.snr_db, self.sir_db, self.sound_speed_m_s]
        if not np.all(np.isfinite(parameters)) or self.rt60_s <= 0 or self.sound_speed_m_s <= 0:
            raise ValueError("invalid acoustic parameters")
        room = np.asarray(self.room_dimensions_m, dtype=float)
        if room.shape != (3,) or not np.all(np.isfinite(room)) or np.any(room <= 0):
            raise ValueError("room must have three positive finite dimensions")
        # 包括所有阵元，不能只检查阵列中心是否在房间内。
        points = [self.array_center_m, self.target_position_m, self.interference_position_m]
        if any(np.asarray(point).shape != (3,) for point in points):
            raise ValueError("all positions must have three coordinates")
        points = np.vstack([points, self.microphones_m])
        if not np.all(np.isfinite(points)) or np.any(points <= 0) or np.any(points >= room):
            raise ValueError("source/microphone positions must be strictly inside the room")
        if type(self.reference_mic) is not int or not 0 <= self.reference_mic < self.n_mics:
            raise ValueError("reference_mic out of range")
        for source in (self.target_position_m, self.interference_position_m):
            if np.min(np.linalg.norm(self.microphones_m - source, axis=1)) < 0.1:
                raise ValueError("source must be >= 0.1 m from microphones")
        if np.linalg.norm(np.subtract(self.target_position_m, self.interference_position_m)) < 0.2:
            raise ValueError("target and interference must be separated by >= 0.2 m")

    @property
    def microphones_m(self):
        if self.array_geometry == "linear":
            relative = linear_array(self.n_mics, self.spacing_or_radius_m)
        else:
            relative = circular_array(self.n_mics, self.spacing_or_radius_m)
        return relative + np.asarray(self.array_center_m)

    @property
    def target_azimuth_deg(self):
        delta = np.subtract(self.target_position_m, self.array_center_m)
        return float(np.rad2deg(np.arctan2(delta[1], delta[0])) % 360)

    def to_dict(self):
        return asdict(self)


def sample_scenes(config: dict, count: int = 100) -> list[Scene]:
    """从完整参数网格无放回抽样；每条场景都有不依赖执行顺序的随机种子。"""
    settings = config["scenes"]
    combinations = list(product(
        settings["room_dimensions_m"], settings["rt60_s"], config["arrays"].items(),
        settings["target_distance_m"], settings["target_azimuth_deg"], settings["snr_db"],
        settings["sir_db"],
    ))
    if type(count) is not int or not 1 <= count <= len(combinations):
        raise ValueError("count must be within the scene grid size")
    rng = np.random.default_rng(config["seed"])
    selected = rng.permutation(len(combinations))[:count]
    scenes = []
    for index, selected_index in enumerate(selected):
        room, rt60, (_, array), distance, azimuth, snr, sir = combinations[selected_index]
        center = np.array([room[0] / 2, room[1] / 2, 1.2])
        angle = np.deg2rad(azimuth)
        target = center + distance * np.array([np.cos(angle), np.sin(angle), 0])
        interference_angle = rng.uniform(np.pi, 11 * np.pi / 6)
        interference = center + 1.3 * np.array([
            np.cos(interference_angle), np.sin(interference_angle), 0
        ])
        scenes.append(Scene(
            scene_id=f"scene_{index:04d}", seed=int(config["seed"] + index),
            room_dimensions_m=tuple(room), array_geometry=array["geometry"],
            n_mics=array["n_mics"],
            spacing_or_radius_m=array.get("spacing_m", array.get("radius_m")),
            array_center_m=tuple(center), target_position_m=tuple(target),
            interference_position_m=tuple(interference), rt60_s=float(rt60),
            snr_db=float(snr), sir_db=float(sir), sample_rate=config["sample_rate"],
            sound_speed_m_s=config["sound_speed_m_s"], reference_mic=config["reference_mic"],
        ))
    return scenes

