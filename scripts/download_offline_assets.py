"""Fetch the pinned MapLibre browser runtime used by the web cockpit.

OpenStreetMap raster tiles are requested only as the map is viewed. Do not
prefetch or vendor tiles from tile.openstreetmap.org for offline use.
"""

import os
import urllib.request


HEADERS = {
    "User-Agent": "TrueTrack-EdgeAI-Navigation/1.0 (academic prototype)"
}

# Keep the browser runtime reproducible and compatible with index.html.
MAPLIBRE_VERSION = "4.7.1"
MAPLIBRE_BASE = f"https://unpkg.com/maplibre-gl@{MAPLIBRE_VERSION}/dist"


def _valid_asset(path, url):
    try:
        if os.path.getsize(path) <= 500:
            return False
        with open(path, "rb") as asset:
            prefix = asset.read(8192)
    except OSError:
        return False

    if url.endswith(".js"):
        return b"MapLibre GL JS" in prefix
    if url.endswith(".css"):
        return b".maplibregl-map" in prefix
    return False


def download_file(url, dest_path):
    if os.path.exists(dest_path) and _valid_asset(dest_path, url):
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temporary_path = dest_path + ".download"
    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = response.read()
        if len(payload) <= 500:
            raise ValueError("downloaded file is unexpectedly small")

        with open(temporary_path, "wb") as output:
            output.write(payload)
        if not _valid_asset(temporary_path, url):
            raise ValueError("downloaded content does not match the expected MapLibre asset")

        os.replace(temporary_path, dest_path)
        return True
    except Exception as error:
        print(f"Failed {url}: {error}")
        return False
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def bundle_assets():
    print("[1/1] Checking the pinned MapLibre GL JS runtime...")
    lib_dir = os.path.join("web_app", "lib")
    js_url = f"{MAPLIBRE_BASE}/maplibre-gl.js"
    css_url = f"{MAPLIBRE_BASE}/maplibre-gl.css"

    ok_js = download_file(js_url, os.path.join(lib_dir, "maplibre-gl.js"))
    ok_css = download_file(css_url, os.path.join(lib_dir, "maplibre-gl.css"))
    if ok_js and ok_css:
        print("  > MapLibre GL JS and CSS are ready in web_app/lib/.")
    else:
        print("  > MapLibre download incomplete; retry with network access.")

    print("  > The OpenStreetMap basemap loads viewed tiles on demand; this script does not download tiles.")


if __name__ == "__main__":
    bundle_assets()
