import math
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from case1.geo.georef import pixel_to_latlon

# Реальные кадры лежат вне git-репозитория (см. docs/RUN.md); путь
# абсолютный и не зависит от того, в каком git worktree находится код.
REAL_FIELD_DIR = Path(
    "/Users/kr220/Documents/Projects/Qostanai_AgroTech_Hackathon_2026/Dataset 1 кейс/ФотоПолей"
)

def test_georef_center():
    # Center of frame should be drone coordinates
    exif = {
        'RelativeAltitude': 10.0,
        'FocalLength': 50.0,
        'SensorWidth': 36.0,
        'GimbalYaw': 0.0,
        'lat': 50.0,
        'lon': 60.0
    }
    lat, lon, quality, error = pixel_to_latlon(
        px_x=1920/2, px_y=1080/2,
        frame_width=1920, frame_height=1080,
        exif=exif
    )
    assert quality == "estimated"
    assert abs(lat - 50.0) < 1e-7
    assert abs(lon - 60.0) < 1e-7

def test_georef_offset():
    # GSD = 10.0 * 36.0 / (50.0 * 1920) = 360 / 96000 = 0.00375 m/px
    # Offset by 100 pixels in X -> 0.375 m East
    exif = {
        'RelativeAltitude': 10.0,
        'FocalLength': 50.0,
        'SensorWidth': 36.0,
        'GimbalYaw': 0.0,
        'lat': 50.0,
        'lon': 60.0
    }
    lat, lon, quality, error = pixel_to_latlon(
        px_x=1920/2 + 100, px_y=1080/2,
        frame_width=1920, frame_height=1080,
        exif=exif
    )
    assert quality == "estimated"
    assert lon > 60.0  # moved East
    assert abs(lat - 50.0) < 1e-7  # no North-South movement

def test_georef_rotation():
    # 90 degrees Yaw (heading East)
    # Moving +X in image (right) now moves South (or North depending on camera orientation)
    # In standard drone convention: 
    # Image +Y is down (backward), +X is right.
    # If heading East (90 deg), +Y (backward) is West, +X (right) is South.
    # Let's just check it rotates.
    exif = {
        'RelativeAltitude': 10.0,
        'FocalLength': 50.0,
        'SensorWidth': 36.0,
        'GimbalYaw': 90.0,
        'lat': 50.0,
        'lon': 60.0
    }
    lat, lon, quality, error = pixel_to_latlon(
        px_x=1920/2 + 100, px_y=1080/2,
        frame_width=1920, frame_height=1080,
        exif=exif
    )
    assert quality == "estimated"
    # It should move South (lat < 50.0)
    assert lat < 50.0

def test_georef_incomplete_exif():
    exif = {
        'lat': 50.0,
        'lon': 60.0
    }
    lat, lon, quality, error = pixel_to_latlon(
        px_x=1920/2 + 100, px_y=1080/2,
        frame_width=1920, frame_height=1080,
        exif=exif
    )
    assert quality == "frame_only"
    assert lat == 50.0
    assert lon == 60.0

def test_georef_negative_altitude():
    exif = {
        'RelativeAltitude': -5.0,
        'FocalLength': 50.0,
        'SensorWidth': 36.0,
        'GimbalYaw': 0.0,
        'lat': 50.0,
        'lon': 60.0
    }
    lat, lon, quality, error = pixel_to_latlon(
        px_x=1920/2 + 100, px_y=1080/2,
        frame_width=1920, frame_height=1080,
        exif=exif
    )
    assert quality == "frame_only"
    assert lat == 50.0
    assert lon == 60.0


def test_georef_no_sensor_width_uses_35mm_equivalent():
    """DJI EXIF never publishes SensorWidth, only FocalLength (real) and
    FocalLengthIn35mmFormat (35mm-equivalent). Mixing the 36mm fallback
    sensor width with the REAL focal length inflates GSD by the camera's
    crop factor. This models a DJI FC9313 (Mavic 3-class): real focal
    8.67mm, 35mm-equivalent 24mm (crop factor ~2.77)."""
    exif_no_sensor_width = {
        'RelativeAltitude': 2.2,
        'FocalLength': 8.67,
        'FocalLengthIn35mmFormat': 24.0,
        'GimbalYaw': 0.0,
        'lat': 50.0,
        'lon': 60.0,
    }
    exif_explicit_35mm_full_frame = {
        # Equivalent camera expressed directly as a 36mm full-frame sensor
        # with the 35mm-equivalent focal length -> must give the identical
        # GSD/offset as the DJI-style exif above.
        'RelativeAltitude': 2.2,
        'FocalLength': 24.0,
        'SensorWidth': 36.0,
        'GimbalYaw': 0.0,
        'lat': 50.0,
        'lon': 60.0,
    }
    frame_width, frame_height = 4096, 3072
    px_x, px_y = frame_width / 2 + 300, frame_height / 2

    lat1, lon1, q1, err1 = pixel_to_latlon(px_x, px_y, frame_width, frame_height, exif_no_sensor_width)
    lat2, lon2, q2, err2 = pixel_to_latlon(px_x, px_y, frame_width, frame_height, exif_explicit_35mm_full_frame)

    assert q1 == q2 == "estimated"
    assert lat1 == pytest.approx(lat2, abs=1e-9)
    assert lon1 == pytest.approx(lon2, abs=1e-9)

    # Regression guard for the fixed bug: mixing 36mm with the REAL 8.67mm
    # focal length (instead of the 35mm-equivalent 24mm) would move the
    # point ~2.77x further from the drone than the correct answer.
    from pyproj import Geod
    g = Geod(ellps='WGS84')
    _, _, dist_correct_m = g.inv(60.0, 50.0, lon1, lat1)
    assert 0.15 < dist_correct_m < 0.30  # GSD ~0.806mm/px * 300px ~= 0.24m


@pytest.mark.skipif(not REAL_FIELD_DIR.exists(), reason="real DJI dataset not available in this environment")
def test_georef_real_dji_frames_produce_submillimeter_gsd():
    """Sanity check on the real 5-frame dataset (RelativeAltitude in the
    0.2-2.3m range): GSD must land in the sub-millimeter/pixel range, not
    centimeters or meters, and geo_quality must be 'estimated' (both
    FocalLengthIn35mmFormat and GimbalYawDegree/FlightYawDegree must be
    extracted correctly from the real DJI XMP)."""
    from case1_main import extract_dji_metadata

    frames = sorted(REAL_FIELD_DIR.glob("*.JPG"))
    assert frames, "expected DJI frames in the real dataset directory"

    for frame_path in frames:
        meta = extract_dji_metadata(frame_path)
        assert meta["rel_alt"] is not None and 0.0 < meta["rel_alt"] <= 5.0
        assert meta["focal_length_35mm"] is not None, f"{frame_path.name}: FocalLengthIn35mmFormat not extracted"
        assert meta["gimbal_yaw"] is not None, f"{frame_path.name}: GimbalYawDegree/FlightYawDegree not extracted"

        lat, lon, quality, error_m = pixel_to_latlon(
            px_x=4096 / 2 + 200, px_y=3072 / 2 + 150,
            frame_width=4096, frame_height=3072,
            exif=meta,
        )
        assert quality == "estimated", f"{frame_path.name}: expected estimated geo_quality, got {quality}"
        assert error_m is not None and 0.5 < error_m < 5.0

        gsd_mm_per_px = meta["rel_alt"] * 36.0 / (meta["focal_length_35mm"] * 4096) * 1000.0
        assert 0.01 < gsd_mm_per_px < 5.0, f"{frame_path.name}: implausible GSD {gsd_mm_per_px} mm/px"
