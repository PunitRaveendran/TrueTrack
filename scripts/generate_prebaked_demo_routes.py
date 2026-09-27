"""
generate_prebaked_demo_routes.py

Phase 7 — Emergency Demo Fallback (Task 3.5)
Pre-bakes static bundles for 3 representative demo destinations from Raidurg Metro:
  - Route 1: Raidurg Metro -> Cyber Towers
  - Route 2: Raidurg Metro -> Trident Hotel
  - Route 3: Raidurg Metro -> Deloitte Mindspace

For each route, generates:
  1. route_polyline.json (GeoJSON LineString)
  2. corridor_ribbon.geojson (GeoJSON FeatureCollection with buffered boundary and clipped manifolds)
  3. connectivity.csv (Sampled 20m cellular signal strength)
  4. blackout_events.json (Discrete edge transitions and autonomous mode tags)

Bundled directly into:
  - android_app/app/src/main/assets/demo_assets/route_N/
  - web_app/data/demo_assets/route_N/
"""

import csv
import json
import os
import shutil
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, "scripts"))

from corridor_buffering import a_star, build_corridor_ribbon, haversine
from generate_connectivity_heatmap import (
    export_connectivity_csv,
    generate_connectivity_for_route,
    load_local_towers
)
from tag_blackout_states import tag_blackout_events


def main():
    print("==================================================================")
    print("=== PRE-BAKING STATIC DEMO ROUTES FOR ZERO-COMPUTE FALLBACK ===")
    print("==================================================================")

    graph_path = os.path.join(base_dir, "web_app", "data", "demo_road_graph.json")
    pois_path = os.path.join(base_dir, "web_app", "data", "demo_pois.json")

    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    with open(pois_path, "r", encoding="utf-8") as f:
        pois = json.load(f)["pois"]

    origin = next(p for p in pois if p["id"] == "raidurg_metro")

    destinations = [
        ("route_1", "cyber_towers", "Cyber Towers Junction"),
        ("route_2", "trident_hotel", "Trident Hotel Hyderabad"),
        ("route_3", "deloitte_driveway", "Deloitte Mindspace")
    ]

    # Preload towers in corridor area
    bounds = graph["metadata"]["bounds"]
    towers = load_local_towers(base_dir, bounds["minlat"], bounds["minlon"], bounds["maxlat"], bounds["maxlon"])
    print(f"Loaded {len(towers)} towers for corridor area.")

    android_assets = os.path.join(base_dir, "android_app", "app", "src", "main", "assets", "demo_assets")
    web_assets = os.path.join(base_dir, "web_app", "data", "demo_assets")

    manifest = []

    for route_dir_name, dest_id, dest_name in destinations:
        dest = next(p for p in pois if p["id"] == dest_id)
        print(f"\nProcessing {route_dir_name}: Raidurg Metro -> {dest_name}...")

        # 1. A* Route
        path, dist_m = a_star(origin["nearest_node"], dest["nearest_node"], graph["nodes"], graph["adjacency"])
        if not path:
            print(f"  [ERROR] No path found to {dest_id}!")
            continue

        route_coords = [[graph["nodes"][str(nid)]["lon"], graph["nodes"][str(nid)]["lat"]] for nid in path]

        route_geojson = {
            "type": "Feature",
            "properties": {
                "route_id": f"raidurg_to_{dest_id}",
                "origin": origin["name"],
                "destination": dest_name,
                "distance_m": round(dist_m, 1),
                "num_nodes": len(path)
            },
            "geometry": {
                "type": "LineString",
                "coordinates": route_coords
            }
        }

        # 2. Corridor Ribbon
        corridor_geojson, _ = build_corridor_ribbon(route_coords, graph, default_width_m=175.0, junction_width_m=350.0)

        # 3. Connectivity Heatmap CSV
        conn_samples = generate_connectivity_for_route(route_coords, towers, step_m=20.0)

        # Temp save CSV to parse events
        temp_csv = os.path.join(base_dir, f"temp_{dest_id}.csv")
        export_connectivity_csv(conn_samples, temp_csv)

        # 4. Blackout State Events
        blackout_events = tag_blackout_events(temp_csv, route_id=f"raidurg_to_{dest_id}")
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

        # Save to both target directories
        for target_root in [android_assets, web_assets]:
            out_dir = os.path.join(target_root, route_dir_name)
            os.makedirs(out_dir, exist_ok=True)

            with open(os.path.join(out_dir, "route_polyline.json"), "w", encoding="utf-8") as f:
                json.dump(route_geojson, f, indent=2)

            with open(os.path.join(out_dir, "corridor_ribbon.geojson"), "w", encoding="utf-8") as f:
                json.dump(corridor_geojson, f, indent=2)

            export_connectivity_csv(conn_samples, os.path.join(out_dir, "connectivity.csv"))

            with open(os.path.join(out_dir, "blackout_events.json"), "w", encoding="utf-8") as f:
                json.dump(blackout_events, f, indent=2)

        manifest.append({
            "id": route_dir_name,
            "destination_id": dest_id,
            "destination_name": dest_name,
            "distance_m": round(dist_m, 1),
            "num_nodes": len(path),
            "num_blackout_segments": blackout_events["num_blackout_segments"],
            "assets_dir": f"demo_assets/{route_dir_name}/"
        })
        print(f"  [OK] Saved bundle for {route_dir_name} ({dist_m:.1f}m, {len(conn_samples)} connectivity samples, {blackout_events['num_blackout_segments']} dead-zones)")

    # Save manifest
    manifest_payload = {
        "offline_demo_fallback_active": True,
        "origin": "Raidurg Metro Station",
        "routes": manifest
    }
    for target_root in [android_assets, web_assets]:
        with open(os.path.join(target_root, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2)

    print("\n[SUCCESS] Phase 7 Emergency Demo Fallback Bundles Created Successfully!")


if __name__ == "__main__":
    main()
