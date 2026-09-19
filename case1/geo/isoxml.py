"""ISO 11783-10 TASKDATA export.

This module exports treatment zones to ISOXML (TaskData Version 4) with Grid Type 2.
Since official XSD from AEF requires registration and isn't trivially downloadable,
validation relies on structural checks in our unit tests.
"""
import xml.etree.ElementTree as ET
import struct
import numpy as np
import geopandas as gpd
from pathlib import Path
from pyproj import Transformer

def export_isoxml(zones_geojson, output_dir, cell_size_m=1.0):
    output_dir = Path(output_dir)
    taskdata_dir = output_dir / "TASKDATA"
    taskdata_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Read zones
    if isinstance(zones_geojson, (str, Path)):
        zones = gpd.read_file(zones_geojson)
    else:
        zones = gpd.GeoDataFrame.from_features(zones_geojson["features"], crs="EPSG:4326")
        
    if zones.empty:
        # Generate empty TASKDATA
        pass # TODO
        
    # Project zones to metric to calculate grid
    center_lon, center_lat = zones.iloc[0].geometry.centroid.x, zones.iloc[0].geometry.centroid.y
    aeqd_crs = f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    zones_m = zones.to_crs(aeqd_crs)
    
    minx, miny, maxx, maxy = zones_m.total_bounds
    
    # We want to create a grid. 
    # ISO-XML grid requires North/East positions in WGS84 for the origin.
    # GridMinimumNorthPosition (lat) and GridMinimumEastPosition (lon).
    # GridCellNorthSize and GridCellEastSize are in degrees! 
    # Wait, in ISO-XML, Grid Type 2 has dimensions in degrees.
    # Let's check: "For GridType 2, sizes are in degrees".
    # But usually 1 meter ~ 0.00000898 degrees lat.
    # Let's do it simple: we define the grid in WGS84.
    
    min_lon, min_lat, max_lon, max_lat = zones.total_bounds
    
    # 1m in degrees roughly
    lat_size = 1.0 / 111111.0
    lon_size = 1.0 / (111111.0 * np.cos(np.radians(center_lat)))
    
    cols = int(np.ceil((max_lon - min_lon) / lon_size)) + 1
    rows = int(np.ceil((max_lat - min_lat) / lat_size)) + 1
    
    # Rasterize
    grid = np.zeros((rows, cols), dtype=np.uint32)
    
    # Simple rasterization: check each zone's bounding box and fill
    for idx, row in zones.iterrows():
        rate = int(row.get("rate_l_ha", 0))
        if rate == 0: continue
        
        geom = row.geometry
        b_minx, b_miny, b_maxx, b_maxy = geom.bounds
        
        c_min = max(0, int((b_minx - min_lon) / lon_size))
        c_max = min(cols - 1, int((b_maxx - min_lon) / lon_size))
        r_min = max(0, int((b_miny - min_lat) / lat_size))
        r_max = min(rows - 1, int((b_maxy - min_lat) / lat_size))
        
        # We can just fill the bounding box or accurately check intersection.
        # For small 1m cells, we'll check intersection with the cell centroid
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                cell_lon = min_lon + (c + 0.5) * lon_size
                cell_lat = min_lat + (r + 0.5) * lat_size
                from shapely.geometry import Point
                if geom.contains(Point(cell_lon, cell_lat)):
                    grid[r, c] = rate

    # Write bin file
    # GRD00001.bin: rows are written from minimum north to maximum north, or max to min?
    # ISO 11783-10: "The first value in the file corresponds to the cell at GridMinimumNorthPosition and GridMinimumEastPosition"
    # So r=0, c=0 is min_lat, min_lon.
    # We flatten and pack as 32-bit unsigned integers.
    bin_path = taskdata_dir / "GRD00001.bin"
    with open(bin_path, "wb") as f:
        # grid shape is (rows, cols)
        # flattened in C-order: row0, row1... where row0 is min_lat.
        # Each row goes from min_lon to max_lon.
        f.write(grid.tobytes(order='C'))

    # XML root
    root = ET.Element("ISO11783_TaskData", VersionMajor="4", VersionMinor="3", DataTransferOrigin="1")
    
    # CTR (Customer), FRM (Farm), PFD (Partfield)
    ctr = ET.SubElement(root, "CTR", A="CTR0001", B="AgroVision_Customer")
    frm = ET.SubElement(root, "FRM", A="FRM0001", B="AgroVision_Farm", I="CTR0001")
    pfd = ET.SubElement(root, "PFD", A="PFD0001", C="AgroVision_Field", D="0", E="FRM0001")
    
    # Boundary PLN -> LSG -> PNT
    pln = ET.SubElement(pfd, "PLN", A="1") # 1 = Polygon outside
    lsg = ET.SubElement(pln, "LSG", A="1") # 1 = Polygon exterior
    
    # Build a simple rectangle boundary from field bounds, slightly expanded
    p1 = ET.SubElement(lsg, "PNT", A="1", C=f"{min_lat-lat_size*5:.7f}", D=f"{min_lon-lon_size*5:.7f}")
    p2 = ET.SubElement(lsg, "PNT", A="1", C=f"{max_lat+lat_size*5:.7f}", D=f"{min_lon-lon_size*5:.7f}")
    p3 = ET.SubElement(lsg, "PNT", A="1", C=f"{max_lat+lat_size*5:.7f}", D=f"{max_lon+lon_size*5:.7f}")
    p4 = ET.SubElement(lsg, "PNT", A="1", C=f"{min_lat-lat_size*5:.7f}", D=f"{max_lon+lon_size*5:.7f}")
    
    # Product (PDT)
    pdt = ET.SubElement(root, "PDT", A="PDT0001", B="Herbicide")
    
    # Task (TSK)
    tsk = ET.SubElement(root, "TSK", A="TSK0001", B="Spot Spraying", C="PFD0001", G="1") # G=1 Status=Planned
    
    # TZN
    for idx, row in zones.iterrows():
        rate = int(row.get("rate_l_ha", 0))
        tzn = ET.SubElement(tsk, "TZN", A=str(idx+1))
        pdv = ET.SubElement(tzn, "PDV", A="0001", B=str(rate), C="PDT0001")
        
        # Add polygon for TZN
        geom = row.geometry
        if geom.geom_type == "Polygon":
            polys = [geom]
        else:
            polys = list(geom.geoms)
            
        for p in polys:
            z_pln = ET.SubElement(tzn, "PLN", A="1")
            z_lsg = ET.SubElement(z_pln, "LSG", A="1")
            for coord in p.exterior.coords:
                ET.SubElement(z_lsg, "PNT", A="1", C=f"{coord[1]:.7f}", D=f"{coord[0]:.7f}")

    # Grid (GRD)
    # A = GridMinimumNorthPosition
    # B = GridMinimumEastPosition
    # C = GridCellNorthSize
    # D = GridCellEastSize
    # E = GridMaximumColumn
    # F = GridMaximumRow
    # G = Filename (without .bin)
    # H = FileLength (bytes)
    # I = GridType (2 = grid 2)
    # J = TreatmentZoneCode (0 for now)
    
    grd = ET.SubElement(tsk, "GRD", 
        A=f"{min_lat:.7f}", 
        B=f"{min_lon:.7f}", 
        C=f"{lat_size:.7f}", 
        D=f"{lon_size:.7f}", 
        E=str(cols), 
        F=str(rows), 
        G="GRD00001", 
        I="2"
    )
    
    tree = ET.ElementTree(root)
    xml_path = taskdata_dir / "TASKDATA.XML"
    
    # Pretty print
    ET.indent(tree, space="  ", level=0)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    
    # Zip it up
    import shutil
    shutil.make_archive(str(output_dir / "TASKDATA"), 'zip', taskdata_dir)
    
    return xml_path
