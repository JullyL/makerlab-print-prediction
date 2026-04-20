"""
Extracts G-code features from Bambu Lab .gcode.3mf files.

Usage:
    python -m src.parse_3mf --input_dir ./3mf_files --output real_prints_features.csv
"""

import zipfile
import os
import re
import math
import csv
import argparse
from typing import Optional


def parse_gcode(gcode_text: str) -> dict:
    lines = gcode_text.splitlines()

    layer_count        = 0
    max_z              = 0.0
    total_extrusion    = 0.0
    total_travel       = 0.0
    total_toolpath     = 0.0
    print_time_sec     = 0.0

    cur_x, cur_y   = 0.0, 0.0
    prev_x, prev_y = 0.0, 0.0
    min_x = min_y  =  float("inf")
    max_x = max_y  = -float("inf")

    layer_move_lengths:  list[float] = []
    current_layer_moves: list[float] = []
    is_first_layer       = True
    first_layer_toolpath = 0.0

    feedrates:   list[float] = []
    cur_feedrate = 0.0

    retraction_count = 0
    cur_e = prev_e   = 0.0

    layer_re        = re.compile(r';.*\blayer\s*[:=]?\s*(\d+)', re.IGNORECASE)
    bambu_layer_re  = re.compile(r';\s*CHANGE_LAYER', re.IGNORECASE)
    total_layers_re = re.compile(r';\s*total layer number\s*[:=]?\s*(\d+)', re.IGNORECASE)
    z_re    = re.compile(r'Z([\d.]+)',   re.IGNORECASE)
    x_re    = re.compile(r'X([-\d.]+)', re.IGNORECASE)
    y_re    = re.compile(r'Y([-\d.]+)', re.IGNORECASE)
    e_re    = re.compile(r'E([-\d.]+)', re.IGNORECASE)
    f_re    = re.compile(r'F([\d.]+)',  re.IGNORECASE)
    time_re = re.compile(
        r';\s*(?:estimated printing time|total estimated time).*?(\d+)h\s*(\d+)m\s*(\d+)s',
        re.IGNORECASE,
    )
    time_re2 = re.compile(
        r';\s*(?:estimated printing time|total estimated time).*?(\d+)m\s*(\d+)s',
        re.IGNORECASE,
    )
    g92_re = re.compile(r'^G92\s.*E([-\d.]+)', re.IGNORECASE)

    for line in lines:
        line = line.strip()

        tlm = total_layers_re.search(line)
        if tlm:
            layer_count = max(layer_count, int(tlm.group(1)))
            continue

        is_layer_change = bool(bambu_layer_re.search(line))
        if not is_layer_change:
            lm = layer_re.search(line)
            if lm:
                n = int(lm.group(1))
                if n > layer_count:
                    layer_count = n
                is_layer_change = True

        if is_layer_change:
            if current_layer_moves:
                layer_sum = sum(current_layer_moves)
                if is_first_layer:
                    first_layer_toolpath = layer_sum
                    is_first_layer = False
                layer_move_lengths.append(layer_sum)
                current_layer_moves = []
            continue

        tm = time_re.search(line)
        if tm:
            h, m, s = int(tm.group(1)), int(tm.group(2)), int(tm.group(3))
            print_time_sec = h * 3600 + m * 60 + s
            continue
        tm2 = time_re2.search(line)
        if tm2 and print_time_sec == 0:
            m, s = int(tm2.group(1)), int(tm2.group(2))
            print_time_sec = m * 60 + s
            continue

        g92m = g92_re.search(line)
        if g92m:
            cur_e = float(g92m.group(1))
            prev_e = cur_e
            continue

        if not line.startswith(('G0', 'G1', 'G00', 'G01')):
            continue

        xm = x_re.search(line)
        ym = y_re.search(line)
        zm = z_re.search(line)
        em = e_re.search(line)
        fm = f_re.search(line)

        if xm: cur_x = float(xm.group(1))
        if ym: cur_y = float(ym.group(1))
        if zm:
            z_val = float(zm.group(1))
            if z_val > max_z:
                max_z = z_val
        if fm:
            cur_feedrate = float(fm.group(1)) / 60.0  # mm/min → mm/s

        if fm and line.upper().startswith('G1') and cur_feedrate > 0:
            feedrates.append(cur_feedrate)

        if em:
            new_e   = float(em.group(1))
            delta_e = new_e - prev_e
            if delta_e < -0.1:
                retraction_count += 1
            prev_e = cur_e = new_e

        dx   = cur_x - prev_x
        dy   = cur_y - prev_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > 0 and em and float(em.group(1)) > 0 if em else False:
            if cur_x < min_x: min_x = cur_x
            if cur_x > max_x: max_x = cur_x
            if cur_y < min_y: min_y = cur_y
            if cur_y > max_y: max_y = cur_y

        if dist > 0:
            total_toolpath += dist
            current_layer_moves.append(dist)

            if em:
                e_val = float(em.group(1))
                if e_val > 0:
                    total_extrusion += dist
                else:
                    total_travel += dist
            else:
                if line.upper().startswith('G0'):
                    total_travel += dist
                else:
                    total_extrusion += dist

        prev_x, prev_y = cur_x, cur_y

    if current_layer_moves:
        layer_sum = sum(current_layer_moves)
        if is_first_layer:
            first_layer_toolpath = layer_sum
        layer_move_lengths.append(layer_sum)

    extrusion_travel_ratio = (
        round(total_extrusion / total_travel, 4) if total_travel > 0 else 0.0
    )

    if len(layer_move_lengths) > 1:
        mean_lml = sum(layer_move_lengths) / len(layer_move_lengths)
        variance = sum((v - mean_lml) ** 2 for v in layer_move_lengths) / len(layer_move_lengths)
        path_variability = round(math.sqrt(variance), 4)
    else:
        path_variability = 0.0

    bbox_x = round(max_x - min_x, 2) if min_x != float("inf") else 0.0
    bbox_y = round(max_y - min_y, 2) if min_y != float("inf") else 0.0

    footprint    = max(bbox_x, bbox_y)
    aspect_ratio = round(max_z / footprint, 4) if footprint > 0 else 0.0

    max_layer_toolpath = round(max(layer_move_lengths), 2) if layer_move_lengths else 0.0

    if layer_move_lengths:
        mean_lml        = sum(layer_move_lengths) / len(layer_move_lengths)
        sparse_count    = sum(1 for v in layer_move_lengths if v < mean_lml * 0.10)
        sparse_layer_fraction = round(sparse_count / len(layer_move_lengths), 4)
    else:
        sparse_layer_fraction = 0.0

    if feedrates:
        avg_feedrate = round(sum(feedrates) / len(feedrates), 2)
        if len(feedrates) > 1 and avg_feedrate > 0:
            std_f        = math.sqrt(sum((v - avg_feedrate) ** 2 for v in feedrates) / len(feedrates))
            feedrate_cv  = round(std_f / avg_feedrate, 4)
        else:
            feedrate_cv = 0.0
    else:
        avg_feedrate = feedrate_cv = 0.0

    retraction_density = (
        round(retraction_count / layer_count, 4) if layer_count > 0 else 0.0
    )

    return {
        "total_layers":            layer_count,
        "max_z_height_mm":         round(max_z, 3),
        "total_toolpath_mm":       round(total_toolpath, 2),
        "extrusion_travel_ratio":  extrusion_travel_ratio,
        "est_print_duration_sec":  int(print_time_sec),
        "path_variability":        path_variability,
        "bbox_x_mm":               bbox_x,
        "bbox_y_mm":               bbox_y,
        "aspect_ratio":            aspect_ratio,
        "first_layer_toolpath_mm": round(first_layer_toolpath, 2),
        "max_layer_toolpath_mm":   max_layer_toolpath,
        "sparse_layer_fraction":   sparse_layer_fraction,
        "avg_feedrate_mms":        avg_feedrate,
        "feedrate_cv":             feedrate_cv,
        "retraction_count":        retraction_count,
        "retraction_density":      retraction_density,
    }


def read_gcode_from_3mf(zip_path: str) -> Optional[str]:
    """Open a .gcode.3mf (ZIP) and return the raw G-code text, or None on failure."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            for candidate in ("Metadata/plate_1.gcode", "Metadata/plate_1_gcode.gcode"):
                if candidate in names:
                    return zf.read(candidate).decode('utf-8', errors='replace')
            for name in names:
                if name.endswith('.gcode'):
                    return zf.read(name).decode('utf-8', errors='replace')
            print(f"  [warn] No G-code found inside {os.path.basename(zip_path)}")
            return None
    except zipfile.BadZipFile:
        print(f"  [error] Cannot open {zip_path} as ZIP")
        return None


FEATURE_COLS = [
    "filename",
    "total_layers", "max_z_height_mm", "total_toolpath_mm",
    "extrusion_travel_ratio", "est_print_duration_sec", "path_variability",
    "bbox_x_mm", "bbox_y_mm", "aspect_ratio",
    "first_layer_toolpath_mm", "max_layer_toolpath_mm", "sparse_layer_fraction",
    "avg_feedrate_mms", "feedrate_cv",
    "retraction_count", "retraction_density",
]


def extract_features_from_folder(input_dir: str) -> list[dict]:
    rows, skipped = [], []
    candidates = [
        f for f in sorted(os.listdir(input_dir))
        if f.endswith('.3mf') or f.endswith('.gcode.3mf')
    ]
    if not candidates:
        print(f"[warn] No .3mf files found in {input_dir}")
        return rows

    print(f"Found {len(candidates)} .3mf file(s) in '{input_dir}'")
    for fname in candidates:
        print(f"Processing: {fname}")
        gcode_text = read_gcode_from_3mf(os.path.join(input_dir, fname))
        if gcode_text is None:
            skipped.append(fname)
            continue
        features = parse_gcode(gcode_text)
        rows.append({"filename": fname, **features})
        print(
            f"  layers={features['total_layers']}  max_z={features['max_z_height_mm']}mm"
            f"  toolpath={features['total_toolpath_mm']}mm"
            f"  retracts={features['retraction_count']}"
            f"  bbox={features['bbox_x_mm']}×{features['bbox_y_mm']}mm"
        )
    if skipped:
        print(f"\nSkipped {len(skipped)} file(s): {skipped}")
    return rows


def write_csv(rows: list[dict], output_path: str) -> None:
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {output_path}")


def extract_features_from_bytes(file_bytes: bytes) -> dict:
    """Parse features from raw bytes — accepts .gcode text or .gcode.3mf ZIP."""
    import io
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as zf:
            names = zf.namelist()
            for candidate in ("Metadata/plate_1.gcode", "Metadata/plate_1_gcode.gcode"):
                if candidate in names:
                    return parse_gcode(zf.read(candidate).decode('utf-8', errors='replace'))
            for name in names:
                if name.endswith('.gcode'):
                    return parse_gcode(zf.read(name).decode('utf-8', errors='replace'))
    except (zipfile.BadZipFile, Exception):
        pass
    try:
        return parse_gcode(file_bytes.decode('utf-8', errors='replace'))
    except Exception as exc:
        raise ValueError(f"Could not parse uploaded file: {exc}") from exc


def _cli():
    parser = argparse.ArgumentParser(description="Extract G-code features from .gcode.3mf files")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", default="real_prints_features.csv")
    args = parser.parse_args()
    rows = extract_features_from_folder(args.input_dir)
    if rows:
        write_csv(rows, args.output)
    else:
        print("No features extracted.")


if __name__ == "__main__":
    _cli()
