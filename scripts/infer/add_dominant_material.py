#!/usr/bin/env python3
"""
Bridge (stage 10): add scalar convenience fields to the enriched GeoJSON written
by `predict_roof_materials.py`, so the green-roof / LST scenario tool can filter
buildings on a single-valued attribute.

Added properties:
  dominant_material      first entry of predicted_roof_materials, as a string
  dominant_material_cov  matching coverage fraction, as a float
  n_materials            number of reported materials

This step is OPTIONAL. `green-roof-scenario` also accepts the list-valued field
directly (`--roof_material_field predicted_roof_materials`), because its value
normaliser takes the first element of a list, which is exactly the dominant
material. Use this script when you prefer an explicit scalar column, e.g. for
QGIS styling or for a GeoPackage export.

Only stdlib is used, so this runs without a GIS stack.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SRC_FIELD = "predicted_roof_materials"
COV_FIELD = "material_cov"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_geojson", required=True, help="Output of predict_roof_materials.py.")
    ap.add_argument("--out_geojson", required=True, help="Where the enriched copy is written.")
    ap.add_argument(
        "--drop_unclassified",
        action="store_true",
        help="Skip features that have no predicted material instead of keeping them unchanged.",
    )
    args = ap.parse_args()

    fc = json.loads(Path(args.in_geojson).read_text(encoding="utf-8"))
    if fc.get("type") != "FeatureCollection":
        raise SystemExit("--in_geojson must be a GeoJSON FeatureCollection.")

    out_feats = []
    hist = Counter()
    n_missing = 0

    for feat in fc.get("features", []):
        props = dict(feat.get("properties") or {})
        mats = props.get(SRC_FIELD)
        covs = props.get(COV_FIELD)

        if not isinstance(mats, (list, tuple)) or len(mats) == 0:
            n_missing += 1
            if args.drop_unclassified:
                continue
            props["dominant_material"] = None
            props["dominant_material_cov"] = None
            props["n_materials"] = 0
        else:
            props["dominant_material"] = str(mats[0])
            if isinstance(covs, (list, tuple)) and len(covs) > 0:
                props["dominant_material_cov"] = float(covs[0])
            else:
                props["dominant_material_cov"] = None
            props["n_materials"] = len(mats)
            hist[str(mats[0])] += 1

        out_feats.append({**feat, "properties": props})

    out = {**fc, "features": out_feats}
    out_path = Path(args.out_geojson)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    sys.stderr.write(
        f"[OK] wrote {len(out_feats)} features -> {out_path} "
        f"({n_missing} without a predicted material)\n"
    )
    for code, n in sorted(hist.items(), key=lambda kv: -kv[1]):
        sys.stderr.write(f"[INFO] dominant_material={code}: {n} buildings\n")


if __name__ == "__main__":
    main()
