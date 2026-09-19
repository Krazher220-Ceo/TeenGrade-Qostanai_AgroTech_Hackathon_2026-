import pytest
from case1.geo.georef import pixel_to_latlon

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
