# TrueTrack — Track 3 Data Contracts Specification

> **Version:** 1.0 (Frozen)  
> **Target Consumers:** Android Native App (`MainActivity.kt`), Web Simulation Cockpit (`navigation_engine.js`), Hardware Bridge (`TelemetryStreamServer.kt`)

---

## 1. Route Polyline Schema (GeoJSON `LineString`)

**File / Payload Type:** Route navigation path  
**Coordinate Standard:** `[longitude, latitude]` (Standard GeoJSON RFC 7946)

```json
{
  "type": "Feature",
  "properties": {
    "route_id": "raidurg_to_cyber_towers",
    "origin_poi": "raidurg_metro",
    "destination_poi": "cyber_towers",
    "distance_m": 1020.2,
    "num_nodes": 47,
    "routing_mode": "car_strict_zero_contraflow"
  },
  "geometry": {
    "type": "LineString",
    "coordinates": [
      [78.3771355, 17.4421954],
      [78.3771892, 17.4442934],
      [78.3778848, 17.4469133],
      [78.3804507, 17.4472041]
    ]
  }
}
```

---

## 2. Corridor Ribbon Geometry Schema (GeoJSON `FeatureCollection`)

**File / Payload Type:** Bounded road ribbon + internal clipped manifolds  
**Buffer Standard:** 175m default half-width, 350m junction half-width (degree $\ge 3$)

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "type": "corridor_ribbon_boundary",
        "default_width_m": 175.0,
        "junction_width_m": 350.0,
        "num_road_segments": 1190
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [78.3755, 17.4421],
            [78.3788, 17.4422],
            [78.3812, 17.4475],
            [78.3755, 17.4421]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "type": "route_centerline",
        "length_m": 1020.2
      },
      "geometry": {
        "type": "LineString",
        "coordinates": [
          [78.3771355, 17.4421954],
          [78.3804507, 17.4472041]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "from": 3602303323,
        "to": 827334124,
        "dist_m": 42.5,
        "name": "HITEC City Main Road",
        "highway": "primary",
        "oneway": true
      },
      "geometry": {
        "type": "LineString",
        "coordinates": [
          [78.3771355, 17.4421954],
          [78.3771897, 17.444148]
        ]
      }
    }
  ]
}
```

---

## 3. Connectivity & Blackout State Schema

### A. Discrete Blackout Events (`blackout_events_<route_id>.json`)
Emitted upon entering or exiting degraded or severed connectivity corridors.

```json
{
  "route_id": "hitec_mindspace_corridor",
  "num_events": 2,
  "num_blackout_segments": 1,
  "segments": [
    {
      "start_m": 340.0,
      "end_m": 880.0,
      "length_m": 540.0,
      "start_coords": [17.4440336, 78.3781295],
      "end_coords": [17.4468515, 78.3811817],
      "mode": "AUTONOMOUS_DEAD_RECKONING"
    }
  ],
  "events": [
    {
      "event": "ENTER_BLACKOUT",
      "distance_along_route_m": 340.0,
      "lat": 17.4440336,
      "lon": 78.3781295,
      "mode": "AUTONOMOUS_DEAD_RECKONING",
      "description": "Entered subterranean underpass. GNSS & Cellular severed. Neural-Inertial NPU engaged."
    },
    {
      "event": "EXIT_BLACKOUT",
      "distance_along_route_m": 880.0,
      "lat": 17.4468515,
      "lon": 78.3811817,
      "mode": "GNSS_REACQUIRED",
      "description": "Exited underpass. GNSS reacquired. Sigmoidal reconciliation active."
    }
  ]
}
```

### B. Continuous Connectivity Heatmap CSV (`connectivity_<route_id>.csv`)
Provides 20m resampled resolution along route corridor:
```csv
lat,lon,distance_along_route_m,estimated_strength_dbm,blackout_flag
17.4421954,78.3771355,0.0,-78.4,0
17.4423500,78.3771500,20.0,-79.1,0
17.4440336,78.3781295,340.0,-122.5,1
17.4468515,78.3811817,880.0,-81.2,0
```

---

## 4. Rerouting State Events Schema (WebSocket / IPC Event Bus)

**Channel:** WebSocket port `8765` / Local Kotlin Listener (`RerouteStateListener`)  
**Payload Example:**

```json
{
  "event_type": "NAV_STATE_CHANGE",
  "state": "DEGRADED_DR_MODE",
  "timestamp_ms": 1774735200000,
  "reason": "status: off_route_no_connectivity. Tracking against last known corridor manifold.",
  "active_corridor_id": "corridor_primary",
  "secondary_corridor_id": null,
  "cross_track_error_m": 62.4,
  "status_payload": {
    "status": "off_route_no_connectivity",
    "dead_reckoning_active": true,
    "snap_to_manifold": false
  }
}
```
