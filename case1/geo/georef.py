import math
from pyproj import Geod

def pixel_to_latlon(px_x, px_y, frame_width, frame_height, exif):
    drone_lat = exif.get('lat')
    drone_lon = exif.get('lon')
    
    if drone_lat is None or drone_lon is None:
        return None, None, "error", None
        
    alt = exif.get('rel_alt') or exif.get('RelativeAltitude')
    focal = exif.get('focal_length') or exif.get('FocalLength')
    sensor_width = exif.get('SensorWidth')
    if sensor_width is None:
        # fallback for 35mm equivalent
        sensor_width = 36.0
        
    yaw = exif.get('gimbal_yaw') or exif.get('GimbalYaw')
    if yaw is None:
        yaw = exif.get('FlightYaw')
        
    if not all(v is not None for v in [alt, focal, sensor_width, yaw]) or alt <= 0:
        return drone_lat, drone_lon, "frame_only", None
        
    # GSD in meters per pixel
    gsd = alt * sensor_width / (focal * frame_width)
    
    # Image coordinates with origin at center
    dx_px = px_x - frame_width / 2
    dy_px = frame_height / 2 - px_y # Standard image Y is down, so positive dy_px is "up" (forward/North)
    
    # Coordinates in meters relative to drone (unrotated)
    dx_m = dx_px * gsd
    dy_m = dy_px * gsd
    
    # Rotate by yaw (clockwise from North)
    yaw_rad = math.radians(yaw)
    
    east_m = dx_m * math.cos(yaw_rad) + dy_m * math.sin(yaw_rad)
    north_m = -dx_m * math.sin(yaw_rad) + dy_m * math.cos(yaw_rad)
    
    # Use pyproj Geod to move in meters
    g = Geod(ellps='WGS84')
    azimuth = math.degrees(math.atan2(east_m, north_m))
    distance = math.hypot(east_m, north_m)
    
    lon2, lat2, _ = g.fwd(drone_lon, drone_lat, azimuth, distance)
    
    # Error estimation
    error_m = 1.0 + 0.05 * alt
    
    return lat2, lon2, "estimated", error_m
