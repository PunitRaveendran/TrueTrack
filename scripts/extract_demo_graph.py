"""
extract_demo_graph.py

Extracts a tightly bounded, routable road network graph and curates verified POIs
from OpenStreetMap bbox data for on-device A* routing (Android / Teammate B and Web Cockpit).

ROUTING MODE: CAR-STRICT (ZERO CONTRAFLOW)
  - One-way streets (oneway=yes) are strictly enforced as single-direction edges.
  - Two-way streets (residential, service, two-way avenues) allow both directions.
  - Absolutely zero wrong-way routing down one-way streets.

Outputs:
  - android_app/app/src/main/assets/demo_road_graph.json
  - web_app/data/demo_road_graph.json
  - android_app/app/src/main/assets/demo_pois.json
  - web_app/data/demo_pois.json
"""

import os
import json
import heapq
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSM_INPUT = os.path.join(BASE_DIR, "ml_engine", "osm_bbox_data.json")

ASSETS_DIR = os.path.join(BASE_DIR, "android_app", "app", "src", "main", "assets")
WEB_DATA_DIR = os.path.join(BASE_DIR, "web_app", "data")
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(WEB_DATA_DIR, exist_ok=True)

R_EARTH = 6378137.0


def haversine(lat1, lon1, lat2, lon2):
    """Calculates great-circle distance between two points on Earth in meters."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return float(2.0 * R_EARTH * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def a_star_search(start_id, goal_id, nodes, adj):
    """Robust on-device A* implementation with strict one-way constraints."""
    if start_id not in nodes or goal_id not in nodes:
        return None
    pq = [(0.0, 0.0, start_id, [start_id])]
    visited = {}
    while pq:
        f, g, u, path = heapq.heappop(pq)
        if u == goal_id:
            return path, g
        if u in visited and visited[u] <= g:
            continue
        visited[u] = g
        for edge in adj.get(u, []):
            v = edge["target"]
            c = edge["dist_m"]
            g_new = g + c
            if v not in visited or g_new < visited[v]:
                h = haversine(nodes[v]["lat"], nodes[v]["lon"], nodes[goal_id]["lat"], nodes[goal_id]["lon"])
                heapq.heappush(pq, (g_new + h, g_new, v, path + [v]))
    return None


def main():
    print("==================================================================")
    print("=== EXTRACTING BOUNDED ROAD GRAPH (STRICT ZERO CONTRAFLOW) ===")
    print("==================================================================")

    with open(OSM_INPUT, "r", encoding="utf-8") as f:
        osm_data = json.load(f)

    # 1. Parse all node coordinates
    all_nodes = {}
    for elem in osm_data.get("elements", []):
        if elem.get("type") == "node":
            all_nodes[elem["id"]] = {
                "id": elem["id"],
                "lat": round(float(elem["lat"]), 7),
                "lon": round(float(elem["lon"]), 7)
            }

    print(f"Total OSM nodes in bbox: {len(all_nodes)}")

    # 2. Filter drivable ways
    drivable_highways = {
        "primary", "secondary", "tertiary", "trunk",
        "residential", "primary_link", "secondary_link",
        "service", "unclassified"
    }

    raw_edges = []
    graph_node_ids = set()

    for elem in osm_data.get("elements", []):
        if elem.get("type") == "way" and "tags" in elem:
            tags = elem["tags"]
            hw = tags.get("highway")
            if hw in drivable_highways:
                way_nodes = elem.get("nodes", [])
                way_name = tags.get("name", hw.capitalize())
                oneway = tags.get("oneway") in ["yes", "1", "true"]
                oneway_rev = tags.get("oneway") == "-1"

                for u, v in zip(way_nodes[:-1], way_nodes[1:]):
                    if u in all_nodes and v in all_nodes:
                        dist = round(haversine(all_nodes[u]["lat"], all_nodes[u]["lon"],
                                               all_nodes[v]["lat"], all_nodes[v]["lon"]), 2)
                        graph_node_ids.add(u)
                        graph_node_ids.add(v)

                        # Strict legal edges only:
                        # Forward edge
                        if not oneway_rev:
                            raw_edges.append({
                                "from": u, "to": v, "dist_m": dist,
                                "name": way_name, "highway": hw,
                                "oneway": oneway
                            })
                        # Reverse edge (only if road is NOT one-way)
                        if not oneway and not oneway_rev:
                            raw_edges.append({
                                "from": v, "to": u, "dist_m": dist,
                                "name": way_name, "highway": hw,
                                "oneway": False
                            })
                        elif oneway_rev:
                            raw_edges.append({
                                "from": v, "to": u, "dist_m": dist,
                                "name": way_name, "highway": hw,
                                "oneway": True
                            })

    print(f"Parsed {len(raw_edges)} strictly legal directed edges (0 contraflow)")

    # Build adjacency
    strict_adj = {}
    for e in raw_edges:
        strict_adj.setdefault(e["from"], []).append({
            "target": e["to"],
            "dist_m": e["dist_m"],
            "name": e["name"],
            "highway": e["highway"],
            "oneway": e["oneway"]
        })

    # Filter nodes present in edges
    active_nodes = {nid: all_nodes[nid] for nid in strict_adj.keys()}
    for e in raw_edges:
        if e["to"] in all_nodes:
            active_nodes[e["to"]] = all_nodes[e["to"]]

    print(f"Routable graph: {len(active_nodes)} nodes, {len(raw_edges)} directed edges")

    # 3. Curate 8 Verified Demo POIs in HITEC City / Madhapur
    # Snapped to verified nodes along the legal corridor flow
    candidate_pois = [
        {
            "id": "raidurg_metro",
            "name": "Raidurg Metro Station",
            "category": "transit",
            "target_node": 3602303323,
            "lat": 17.442180,
            "lon": 78.377182,
            "description": "Terminal Hyderabad Metro Blue Line station (Corridor Origin)"
        },
        {
            "id": "cyber_gateway",
            "name": "Cyber Gateway",
            "category": "tech_park",
            "target_node": 6536581993,
            "lat": 17.447246,
            "lon": 78.377899,
            "description": "Major IT complex and transit corridor stop"
        },
        {
            "id": "cyber_towers",
            "name": "Cyber Towers Junction",
            "category": "landmark",
            "target_node": 1310963248,
            "lat": 17.450675,
            "lon": 78.380143,
            "description": "Iconic HITEC City roundabout and tech landmark"
        },
        {
            "id": "trident_hotel",
            "name": "Trident Hotel Hyderabad",
            "category": "hotel",
            "target_node": 11306501197,
            "lat": 17.450112,
            "lon": 78.378793,
            "description": "5-star luxury hotel adjacent to HITEC City Road"
        },
        {
            "id": "deloitte_driveway",
            "name": "Deloitte Mindspace",
            "category": "tech_park",
            "target_node": 11980860807,
            "lat": 17.448509,
            "lon": 78.375746,
            "description": "Deloitte campus in Mindspace IT corridor"
        },
        {
            "id": "tcs_deccan_park",
            "name": "TCS Deccan Park",
            "category": "tech_park",
            "target_node": 3602303693,
            "lat": 17.443898,
            "lon": 78.377896,
            "description": "Tata Consultancy Services software campus"
        },
        {
            "id": "paradise_biryani",
            "name": "Paradise Food Court",
            "category": "food",
            "target_node": 3602306714,
            "lat": 17.450632,
            "lon": 78.379287,
            "description": "Famous Hyderabad culinary and dining landmark"
        },
        {
            "id": "medicover_hospital",
            "name": "Medicover Hospitals",
            "category": "hospital",
            "target_node": 827334185,
            "lat": 17.446849,
            "lon": 78.379876,
            "description": "Multi-specialty hospital at Madhapur junction"
        }
    ]

    curated_pois = []
    for poi in candidate_pois:
        nid = poi["target_node"]
        node_lat = active_nodes[nid]["lat"]
        node_lon = active_nodes[nid]["lon"]
        d = haversine(poi["lat"], poi["lon"], node_lat, node_lon)
        poi["nearest_node"] = nid
        poi["dist_to_node_m"] = round(d, 1)
        poi["snapped_lat"] = node_lat
        poi["snapped_lon"] = node_lon
        curated_pois.append(poi)

    # 4. Compute All Strictly Legal Routes between POIs
    print("\n--- Validating Strictly Legal Routes (Zero Contraflow) ---")
    total_pairs = 0
    legal_routes = 0
    for p_from in curated_pois:
        p_from["reachable_destinations"] = []
        for p_to in curated_pois:
            if p_from["id"] != p_to["id"]:
                total_pairs += 1
                res = a_star_search(p_from["nearest_node"], p_to["nearest_node"], active_nodes, strict_adj)
                if res:
                    legal_routes += 1
                    path, dist = res
                    p_from["reachable_destinations"].append({
                        "id": p_to["id"],
                        "name": p_to["name"],
                        "dist_m": round(dist, 1),
                        "num_nodes": len(path)
                    })
                    print(f"  [LEGAL] {p_from['id']} -> {p_to['id']}: {dist:.1f}m ({len(path)} nodes)")

    print(f"\nStrictly Legal Routes: {legal_routes} / {total_pairs} pairs")
    print(f"Raidurg Metro reaches {len(curated_pois[0]['reachable_destinations'])} destinations:")
    for dest in curated_pois[0]['reachable_destinations']:
        print(f"   -> {dest['name']}: {dest['dist_m']}m ({dest['num_nodes']} nodes)")

    # 5. Export demo_road_graph.json
    bounds = osm_data.get("bounds", {
        "minlat": min(n["lat"] for n in active_nodes.values()),
        "minlon": min(n["lon"] for n in active_nodes.values()),
        "maxlat": max(n["lat"] for n in active_nodes.values()),
        "maxlon": max(n["lon"] for n in active_nodes.values())
    })

    graph_payload = {
        "metadata": {
            "name": "HITEC City / Madhapur Bounded Demo Road Graph",
            "version": "1.1",
            "routing_mode": "car_strict_zero_contraflow",
            "bounds": bounds,
            "num_nodes": len(active_nodes),
            "num_directed_edges": len(raw_edges)
        },
        "nodes": active_nodes,
        "adjacency": strict_adj
    }

    for target_dir in [ASSETS_DIR, WEB_DATA_DIR]:
        graph_file = os.path.join(target_dir, "demo_road_graph.json")
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(graph_payload, f, separators=(",", ":"))
        print(f"Exported graph to {graph_file} ({os.path.getsize(graph_file) / 1024:.1f} KB)")

    # 6. Export demo_pois.json
    poi_payload = {
        "metadata": {
            "region": "Hyderabad HITEC City",
            "count": len(curated_pois),
            "routing_mode": "car_strict_zero_contraflow",
            "osm_data_frozen": True
        },
        "pois": curated_pois
    }

    for target_dir in [ASSETS_DIR, WEB_DATA_DIR]:
        poi_file = os.path.join(target_dir, "demo_pois.json")
        with open(poi_file, "w", encoding="utf-8") as f:
            json.dump(poi_payload, f, indent=2)
        print(f"Exported POIs to {poi_file} ({os.path.getsize(poi_file) / 1024:.1f} KB)")

    print("\n[SUCCESS] Strict zero-contraflow road graph & POIs exported successfully!")


if __name__ == "__main__":
    main()
