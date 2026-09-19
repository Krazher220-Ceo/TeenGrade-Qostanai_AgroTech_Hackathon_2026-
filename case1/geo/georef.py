import math
from pyproj import Geod

# По определению "35-мм эквивалента" фокусного расстояния он выражен против
# полнокадрового сенсора шириной 36 мм. Большинство дронов (включая все DJI
# в этом датасете) не публикуют настоящую SensorWidth через EXIF, но всегда
# публикуют FocalLengthIn35mmFormat — поэтому это единственная величина,
# которую можно сочетать с 36.0 мм.
FULL_FRAME_SENSOR_WIDTH_MM = 36.0


def pixel_to_latlon(px_x, px_y, frame_width, frame_height, exif):
    """Перевод пиксельной координаты в надирном кадре дрона в WGS84 lat/lon.

    Модель надирной камеры:
        GSD [м/px] = высота_над_землёй[м] * ширина_сенсора[мм]
                     / (фокусное_расстояние[мм] * ширина_кадра[px])
    Смещение центра bbox от центра кадра переводится в метры через GSD,
    поворачивается на курс (yaw, по часовой стрелке от севера — картинка:
    "вверх" кадра = направление курса) и прикладывается к GPS-точке дрона
    через pyproj.Geod.fwd (WGS84 геодезия, а не плоская аппроксимация).

    exif может содержать (в порядке приоритета для каждой величины):
        lat, lon                                    — GPS дрона (обязательно)
        rel_alt / RelativeAltitude                   — высота над землёй, м
        SensorWidth                                   — реальная ширина
            сенсора, мм (публикуется редко)
        focal_length / FocalLength                    — реальное фокусное
            расстояние объектива, мм
        focal_length_35mm / FocalLengthIn35mmFormat    — 35-мм эквивалент
            фокусного расстояния, мм (у DJI есть всегда)
        gimbal_yaw / GimbalYaw, FlightYaw              — курс/yaw, градусы
            по часовой от севера

    ВАЖНО (исправленный баг): если SensorWidth неизвестна, то условные
    36 мм можно сочетать ТОЛЬКО с 35-мм эквивалентным фокусным расстоянием
    (focal_length_35mm), а не с реальным FocalLength — иначе GSD завышается
    в разы (крат-фактор камеры). Пример на DJI FC9313 (Mavic 3-класс):
    реальное FocalLength=8.7 мм, 35-мм эквивалент=24 мм (crop factor ~2.76).
    Смешивание 36 мм с реальными 8.7 мм давало GSD в ~2.76 раза больше
    правильного значения.
    """
    drone_lat = exif.get('lat')
    drone_lon = exif.get('lon')

    if drone_lat is None or drone_lon is None:
        return None, None, "error", None

    alt = exif.get('rel_alt')
    if alt is None:
        alt = exif.get('RelativeAltitude')

    sensor_width = exif.get('SensorWidth')
    real_focal = exif.get('focal_length')
    if real_focal is None:
        real_focal = exif.get('FocalLength')
    focal_35mm = exif.get('focal_length_35mm')
    if focal_35mm is None:
        focal_35mm = exif.get('FocalLengthIn35mmFormat')

    if sensor_width is not None and real_focal is not None:
        # Настоящая геометрия сенсора известна полностью — используем её
        # напрямую, без обращения к 36-мм эквиваленту.
        focal = real_focal
    elif focal_35mm is not None:
        # SensorWidth не опубликована (типичный случай для DJI EXIF) —
        # используем метод 35-мм эквивалента: он определён относительно
        # условного полнокадрового сенсора 36 мм, поэтому подставлять сюда
        # реальное FocalLength нельзя (см. docstring).
        sensor_width = FULL_FRAME_SENSOR_WIDTH_MM
        focal = focal_35mm
    else:
        sensor_width = None
        focal = None

    yaw = exif.get('gimbal_yaw')
    if yaw is None:
        yaw = exif.get('GimbalYaw')
    if yaw is None:
        yaw = exif.get('FlightYaw')

    if alt is None or focal is None or sensor_width is None or yaw is None or alt <= 0:
        return drone_lat, drone_lon, "frame_only", None

    # GSD в метрах на пиксель (мм сокращаются в числителе и знаменателе).
    gsd = alt * sensor_width / (focal * frame_width)

    # Координаты изображения с началом в центре кадра.
    dx_px = px_x - frame_width / 2
    dy_px = frame_height / 2 - px_y  # Y изображения растёт вниз, поэтому
    # положительный dy_px значит "вверх кадра" = "вперёд/на север" при yaw=0.

    # Смещение в метрах относительно дрона (ещё без поворота на курс).
    dx_m = dx_px * gsd
    dy_m = dy_px * gsd

    # Поворот на курс (yaw, по часовой стрелке от севера).
    yaw_rad = math.radians(yaw)

    east_m = dx_m * math.cos(yaw_rad) + dy_m * math.sin(yaw_rad)
    north_m = -dx_m * math.sin(yaw_rad) + dy_m * math.cos(yaw_rad)

    # Смещение в метрах по эллипсоиду WGS84 через pyproj.
    g = Geod(ellps='WGS84')
    azimuth = math.degrees(math.atan2(east_m, north_m))
    distance = math.hypot(east_m, north_m)

    lon2, lat2, _ = g.fwd(drone_lon, drone_lat, azimuth, distance)

    # Грубая оценка погрешности геопривязки (растёт с высотой — больше GSD
    # и больше неопределённость по yaw/gimbal на дальних от надира пикселях).
    error_m = 1.0 + 0.05 * alt

    return lat2, lon2, "estimated", error_m
