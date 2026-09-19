import re
with open("case1/dashboard/app.py", "r") as f:
    text = f.read()

# 1. Update generate_iso_xml
old_iso_xml = """def generate_iso_xml(df_weeds, task_name="Task_WeedSpray_Kostanay_2026"):
    xml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<ISO11783_TaskFile VersionMajor="4" VersionMinor="3" ManagementSoftwareManufacturer="AgroVision_AI" ManagementSoftwareVersion="2.0">',
        f'  <TSK A="{task_name}" B="SpotSpray_Prescription" G="1">',
        f'    <TZN A="1" B="TargetZone_Weeds">',
    ]
    count = 0
    if not df_weeds.empty:
        for _, row in df_weeds.iterrows():
            lat = row.get("drone_lat")
            lon = row.get("drone_lon")
            act = str(row.get("spray_action", ""))
            rate = 150.0 if act == "spray_weed" else 0.0
            if pd.notna(lat) and pd.notna(lon):
                count += 1
                xml.append(
                    f'      <PNT A="{count}" C="{float(lat):.7f}" D="{float(lon):.7f}" E="0.0">'
                    f'<PDV A="1" B="{rate:.1f}" C="L_per_HA"/>'
                    f'</PNT>'
                )
    xml.extend([
        '    </TZN>',
        '  </TSK>',
        '</ISO11783_TaskFile>'
    ])
    return "\\n".join(xml)"""

new_iso_xml = """def generate_iso_xml(df_weeds, task_name="Task_WeedSpray_Kostanay_2026"):
    from case1.geo.zones import create_treatment_zones
    from case1.geo.isoxml import create_isoxml_taskdata
    detections = []
    if not df_weeds.empty:
        for _, row in df_weeds.iterrows():
            if pd.notna(row.get("drone_lat")) and pd.notna(row.get("drone_lon")):
                detections.append({
                    "lat": float(row["drone_lat"]),
                    "lon": float(row["drone_lon"]),
                    "action": str(row.get("spray_action", "spray_weed")),
                    "species": str(row.get("species", "unknown"))
                })
    geojson_data = create_treatment_zones(detections)
    xml_str, _ = create_isoxml_taskdata(geojson_data, task_name=task_name)
    return xml_str"""

text = text.replace(old_iso_xml, new_iso_xml)

# 2. Update VRA Tab Map
old_map = """            st.subheader("Тепловая карта засорённости участка")
            fig_density = px.density_heatmap(
                df_img,
                x="bbox_x1",
                y="bbox_y1",
                nbinsx=25,
                nbinsy=20,
                color_continuous_scale="YlOrRd",
                title="Плотность очагов на поле (координаты пикселей)"
            )
            fig_density.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_density, width="stretch")"""

new_map = """            st.subheader("Зоны обработки (Полигоны VRA)")
            try:
                from case1.geo.zones import create_treatment_zones
                import geopandas as gpd
                
                det_list = []
                for _, r in df_img.iterrows():
                    if pd.notna(r.get("drone_lat")) and pd.notna(r.get("drone_lon")):
                        det_list.append({
                            "lat": float(r["drone_lat"]),
                            "lon": float(r["drone_lon"]),
                            "action": str(r.get("spray_action", "spray_weed")),
                            "species": str(r.get("species", "unknown"))
                        })
                
                gj = create_treatment_zones(det_list)
                if gj["features"]:
                    gdf = gpd.GeoDataFrame.from_features(gj)
                    fig_map = px.choropleth_mapbox(
                        gdf,
                        geojson=gdf.geometry.__geo_interface__,
                        locations=gdf.index,
                        color="rate_l_ha",
                        color_continuous_scale="Viridis",
                        mapbox_style="carto-positron",
                        zoom=18,
                        center={"lat": gdf.geometry.centroid.y.mean(), "lon": gdf.geometry.centroid.x.mean()},
                        opacity=0.5,
                        title="Полигоны для опрыскивания (WGS84)"
                    )
                    fig_map.update_layout(margin=dict(l=0, r=0, t=35, b=0))
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.info("Нет зон для обработки на этом снимке.")
            except Exception as e:
                st.error(f"Ошибка построения карты зон: {e}")"""

text = text.replace(old_map, new_map)

with open("case1/dashboard/app.py", "w") as f:
    f.write(text)
