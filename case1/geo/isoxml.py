"""ISO 11783-10 TASKDATA export.

This module exports treatment zones to ISOXML (TaskData VersionMajor="4")
with a GRD Grid Type 1 raster pointing at the TZN (treatment zone) elements.
Since the official XSD from AEF requires registration and isn't trivially
downloadable, validation relies on structural checks in our unit tests and
in scripts/verify_prescription_and_taskdata.py, not on XSD conformance.

GRD attribute mapping used here (A-J), cross-checked against public
documentation (isoxml.tools GRD reference, dev4Agriculture ISOXML tooling
docs) since the paid/registered AEF PDF was not accessible from this
environment:
    A = GridMinimumNorthPosition (deg, south-west corner latitude)
    B = GridMinimumEastPosition  (deg, south-west corner longitude)
    C = GridCellNorthSize        (deg, cell height)
    D = GridCellEastSize         (deg, cell width)
    E = GridMaximumColumn        (grid width, columns)
    F = GridMaximumRow           (grid height, rows)
    G = GridFilename (pattern "GRD00001", no extension)
    H = GridFilename file length in bytes (optional, informational)
    I = GridType: 1 or 2
    J = TreatmentZoneCode (optional, 0-254)

GridType 1 vs 2 (this is the part the original implementation had
backwards): GridType 1 stores ONE byte per cell holding a
TreatmentZoneCode that indexes into this Task's <TZN A="..."> elements
(0 = no treatment / outside any zone); GridType 2 stores the raw
process-data value directly per cell (commonly 4 bytes), with no
TZN indirection. We use GridType 1 here because:
  - our zones are already discrete, non-overlapping polygons with a
    single rate each (natural fit for a zone-code lookup grid);
  - it is 4x more compact than encoding the rate as a 32-bit value per
    cell (the previous implementation used uint32 directly, which is why
    a 21 m^2 patch produced a ~3 MB GRD00001.bin);
  - it lets a terminal (or our own tests) cross-check "for a given
    lat/lon, which TZN/rate applies" by looking up the byte then reading
    that TZN's <PDV>, instead of trusting a duplicated raw value.
Because each cell already carries its own varying code, the optional
GRD.J (a single TreatmentZoneCode for the whole grid) does not apply to
a GridType 1 raster and is intentionally omitted.

Row order in GRD00001.bin: south to north (row 0 = GridMinimumNorthPosition,
i.e. the southernmost row), and west to east within each row (column 0 =
GridMinimumEastPosition). One byte per cell, C-order (row-major) flatten.
"""
import xml.etree.ElementTree as ET
import struct
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
from pyproj import Transformer

# GridType 1: 1 byte/cell TreatmentZoneCode (0 = no treatment). See module
# docstring for why GridType 1 (zone-coded) was chosen over GridType 2
# (raw value grid).
GRID_TYPE = 1
GRID_DTYPE = np.uint8
MAX_TREATMENT_ZONE_CODE = 254  # 255 cells' worth of zones ought to be enough
# 1 byte/cell -> 4M cells is a 4MB GRD00001.bin at worst, a sane cap for a
# file meant to fit on a terminal's USB stick. See the coarsening comment
# below for why this can be hit even with a small total treated area.
MAX_GRID_CELLS = 4_000_000


def export_isoxml(zones_geojson, output_dir, cell_size_m=1.0):
    output_dir = Path(output_dir)
    taskdata_dir = output_dir / "TASKDATA"
    taskdata_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read zones. gpd.GeoDataFrame.from_features([], ...) raises (it
    # can't infer a geometry column from zero features), so detect the
    # empty case before touching geopandas at all.
    if isinstance(zones_geojson, (str, Path)):
        zones = gpd.read_file(zones_geojson)
    elif not zones_geojson.get("features"):
        zones = gpd.GeoDataFrame()
    else:
        zones = gpd.GeoDataFrame.from_features(zones_geojson["features"], crs="EPSG:4326")

    # XML root (built regardless of whether there are any zones, so an
    # empty prescription still produces a structurally valid TASKDATA).
    root = ET.Element("ISO11783_TaskData", VersionMajor="4", VersionMinor="3", DataTransferOrigin="1")

    ctr = ET.SubElement(root, "CTR", A="CTR0001", B="AgroVision_Customer")
    frm = ET.SubElement(root, "FRM", A="FRM0001", B="AgroVision_Farm", I="CTR0001")
    pfd = ET.SubElement(root, "PFD", A="PFD0001", C="AgroVision_Field", D="0", E="FRM0001")
    pdt = ET.SubElement(root, "PDT", A="PDT0001", B="Herbicide")
    tsk = ET.SubElement(root, "TSK", A="TSK0001", B="Spot Spraying", C="PFD0001", G="1")  # G=1 Status=Planned

    if zones.empty:
        # No spray-worthy detections: emit a structurally valid, empty
        # TASKDATA (no boundary/TZN/GRD can be derived without at least
        # one zone geometry). Documented explicitly rather than silently
        # producing a corrupt file.
        tree = ET.ElementTree(root)
        xml_path = taskdata_dir / "TASKDATA.XML"
        ET.indent(tree, space="  ", level=0)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        import shutil
        shutil.make_archive(str(output_dir / "TASKDATA"), 'zip', taskdata_dir)
        return xml_path

    if "rate_l_ha" not in zones.columns:
        raise ValueError("zones_geojson features must carry a 'rate_l_ha' property")

    # Project zones to metric to size the grid meaningfully; the grid
    # itself is still stored/addressed in WGS84 degrees as required by the
    # ISO-XML GRD attributes.
    center_lon, center_lat = zones.iloc[0].geometry.centroid.x, zones.iloc[0].geometry.centroid.y

    min_lon, min_lat, max_lon, max_lat = zones.total_bounds

    # cell_size_m converted to degrees at the field's latitude. This is the
    # parameter the previous implementation accepted but never used —
    # lat/lon cell size was hardcoded to ~1m regardless of the argument.
    lat_size = cell_size_m / 111111.0
    lon_size = cell_size_m / (111111.0 * np.cos(np.radians(center_lat)))

    cols = max(1, int(np.ceil((max_lon - min_lon) / lon_size)) + 1)
    rows = max(1, int(np.ceil((max_lat - min_lat) / lat_size)) + 1)

    # A single GRD covers the whole bounding box of every zone, which is
    # correct for one contiguous field flight but pathological if the
    # input zones come from geographically scattered detections (e.g. a
    # demo dataset made of a few separate stills kilometers apart rather
    # than one continuous survey flight): the bbox — and therefore the
    # cell count at a fixed cell_size_m — balloons even though the actual
    # treated area stays tiny (this is exactly why a ~21 m^2 prescription
    # previously produced a multi-MB GRD00001.bin). Rather than silently
    # writing an unbounded file, coarsen the cell size just enough to keep
    # the grid under MAX_GRID_CELLS, and say so.
    if cols * rows > MAX_GRID_CELLS:
        scale = float(np.sqrt((cols * rows) / MAX_GRID_CELLS))
        cell_size_m = cell_size_m * scale
        lat_size = cell_size_m / 111111.0
        lon_size = cell_size_m / (111111.0 * np.cos(np.radians(center_lat)))
        cols = max(1, int(np.ceil((max_lon - min_lon) / lon_size)) + 1)
        rows = max(1, int(np.ceil((max_lat - min_lat) / lat_size)) + 1)
        print(
            f"[case1.geo.isoxml] zones span a bounding box too large for a "
            f"1m grid ({cols}x{rows} cells would exceed MAX_GRID_CELLS="
            f"{MAX_GRID_CELLS}); this usually means the input zones are not "
            f"one contiguous field flight. Coarsened the grid to "
            f"cell_size_m~={cell_size_m:.2f} ({cols}x{rows} cells) to keep "
            f"GRD00001.bin a sane size. Accuracy at zone boundaries is "
            f"reduced accordingly — a real single-field flight should not "
            f"hit this path."
        )

    # Rasterize: each cell holds the 1-based TreatmentZoneCode of the zone
    # whose polygon contains that cell's center, or 0 if none (0 = no
    # treatment, matches ISO 11783-10 GridType 1 semantics).
    grid = np.zeros((rows, cols), dtype=GRID_DTYPE)

    zone_codes = {}  # idx -> 1-based TreatmentZoneCode, aligned with TZN.A below
    for idx, row in zones.iterrows():
        rate = float(row.get("rate_l_ha", 0) or 0)
        if rate <= 0:
            continue
        zone_code = idx + 1
        if zone_code > MAX_TREATMENT_ZONE_CODE:
            raise ValueError(
                f"zone index {idx} exceeds MAX_TREATMENT_ZONE_CODE={MAX_TREATMENT_ZONE_CODE} "
                "(GridType 1 TreatmentZoneCode is a single byte, 0-254)"
            )
        zone_codes[idx] = zone_code

        geom = row.geometry
        b_minx, b_miny, b_maxx, b_maxy = geom.bounds

        c_min = max(0, int((b_minx - min_lon) / lon_size))
        c_max = min(cols - 1, int((b_maxx - min_lon) / lon_size))
        r_min = max(0, int((b_miny - min_lat) / lat_size))
        r_max = min(rows - 1, int((b_maxy - min_lat) / lat_size))

        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                cell_lon = min_lon + (c + 0.5) * lon_size
                cell_lat = min_lat + (r + 0.5) * lat_size
                if geom.contains(Point(cell_lon, cell_lat)):
                    grid[r, c] = zone_code

    # Write bin file: south to north (row 0 = min_lat), west to east within
    # each row (col 0 = min_lon). C-order flatten matches this directly.
    bin_path = taskdata_dir / "GRD00001.bin"
    with open(bin_path, "wb") as f:
        f.write(grid.tobytes(order='C'))
    file_length_bytes = bin_path.stat().st_size

    # Boundary PLN -> LSG -> PNT: simple rectangle from the zones' bounds,
    # expanded by a few cells so the field boundary strictly encloses the
    # treatment zones.
    pln = ET.SubElement(pfd, "PLN", A="1")  # 1 = Polygon exterior
    lsg = ET.SubElement(pln, "LSG", A="1")  # 1 = Polygon exterior
    ET.SubElement(lsg, "PNT", A="1", C=f"{min_lat-lat_size*5:.7f}", D=f"{min_lon-lon_size*5:.7f}")
    ET.SubElement(lsg, "PNT", A="1", C=f"{max_lat+lat_size*5:.7f}", D=f"{min_lon-lon_size*5:.7f}")
    ET.SubElement(lsg, "PNT", A="1", C=f"{max_lat+lat_size*5:.7f}", D=f"{max_lon+lon_size*5:.7f}")
    ET.SubElement(lsg, "PNT", A="1", C=f"{min_lat-lat_size*5:.7f}", D=f"{max_lon+lon_size*5:.7f}")

    # TZN (treatment zones) — one per polygon, PDV DDI=0001 (Setpoint
    # Volume Per Area Application Rate). Per the ISO 11783-11 Data
    # Dictionary, DDI 0001's unit is mm^3/m^2 with resolution 0.01, i.e.
    # the raw PDV.B integer equals value_mm3_per_m2 / 0.01. Combined with
    # 1 L/ha == 100 mm^3/m^2 (1 L / 10,000 m^2 = 1,000,000 mm^3 / 10,000
    # m^2 = 100 mm^3/m^2):
    #     raw = (rate_l_ha * 100 mm^3/m^2 per L/ha) / 0.01 = rate_l_ha * 10000
    for idx, row in zones.iterrows():
        if idx not in zone_codes:
            continue
        rate_l_ha = float(row.get("rate_l_ha", 0) or 0)
        pdv_value = int(round(rate_l_ha * 10000))
        zone_code = zone_codes[idx]
        tzn = ET.SubElement(tsk, "TZN", A=str(zone_code))
        ET.SubElement(tzn, "PDV", A="0001", B=str(pdv_value), C="PDT0001")

        geom = row.geometry
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            z_pln = ET.SubElement(tzn, "PLN", A="1")
            z_lsg = ET.SubElement(z_pln, "LSG", A="1")
            for coord in poly.exterior.coords:
                ET.SubElement(z_lsg, "PNT", A="1", C=f"{coord[1]:.7f}", D=f"{coord[0]:.7f}")

    # Grid (GRD) — see module docstring for the attribute mapping and the
    # GridType 1 vs 2 decision.
    ET.SubElement(
        tsk, "GRD",
        A=f"{min_lat:.7f}",
        B=f"{min_lon:.7f}",
        C=f"{lat_size:.7f}",
        D=f"{lon_size:.7f}",
        E=str(cols),
        F=str(rows),
        G="GRD00001",
        H=str(file_length_bytes),
        I=str(GRID_TYPE),
    )

    tree = ET.ElementTree(root)
    xml_path = taskdata_dir / "TASKDATA.XML"

    ET.indent(tree, space="  ", level=0)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    import shutil
    shutil.make_archive(str(output_dir / "TASKDATA"), 'zip', taskdata_dir)

    return xml_path


def read_grid(taskdata_xml_path):
    """Read a GRD00001.bin (GridType 1) back into a 2-D array of
    TreatmentZoneCode bytes, plus the georeferencing attributes from the
    <GRD> element. Used by tests to cross-check the raster against the
    <TZN> polygons that produced it.

    Returns a dict with keys: grid (np.uint8 array, shape (rows, cols)),
    min_lat, min_lon, lat_size, lon_size, cols, rows, grid_type.
    """
    taskdata_xml_path = Path(taskdata_xml_path)
    tree = ET.parse(taskdata_xml_path)
    root = tree.getroot()
    grd = root.find(".//GRD")
    if grd is None:
        return None

    min_lat = float(grd.attrib["A"])
    min_lon = float(grd.attrib["B"])
    lat_size = float(grd.attrib["C"])
    lon_size = float(grd.attrib["D"])
    cols = int(grd.attrib["E"])
    rows = int(grd.attrib["F"])
    filename = grd.attrib["G"]
    grid_type = int(grd.attrib["I"])

    bin_path = taskdata_xml_path.parent / f"{filename}.bin"
    raw = bin_path.read_bytes()
    dtype = np.uint8 if grid_type == 1 else np.uint32
    expected_bytes = rows * cols * np.dtype(dtype).itemsize
    if len(raw) != expected_bytes:
        raise ValueError(
            f"{bin_path.name}: expected {expected_bytes} bytes for a {rows}x{cols} "
            f"GridType {grid_type} grid, got {len(raw)}"
        )
    grid = np.frombuffer(raw, dtype=dtype).reshape((rows, cols))

    return {
        "grid": grid,
        "min_lat": min_lat,
        "min_lon": min_lon,
        "lat_size": lat_size,
        "lon_size": lon_size,
        "cols": cols,
        "rows": rows,
        "grid_type": grid_type,
    }
