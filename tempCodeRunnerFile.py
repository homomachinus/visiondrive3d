import os
import json
from glob import glob

# ==========================================
# CONFIG
# ==========================================

# Folder annotation JSON
INPUT_JSON_DIR = "ann"

# Folder output label YOLO
OUTPUT_LABEL_DIR = "labels4"

# Mapping class
CLASS_MAP = {
  
 
    "road": 2,


    "car": 11,
  

 

}

# ==========================================
# CREATE OUTPUT DIR
# ==========================================

os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# ==========================================
# CONVERT FUNCTION
# ==========================================

def convert_polygon_to_yolo(points, img_w, img_h):
    """
    Convert polygon points to YOLO segmentation format
    """
    yolo_points = []

    for x, y in points:
        x_norm = x / img_w
        y_norm = y / img_h

        yolo_points.append(f"{x_norm:.6f}")
        yolo_points.append(f"{y_norm:.6f}")

    return " ".join(yolo_points)

# ==========================================
# PROCESS JSON FILES
# ==========================================

json_files = glob(os.path.join(INPUT_JSON_DIR, "*.json"))

for json_file in json_files:

    with open(json_file, "r") as f:
        data = json.load(f)

    img_w = data["size"]["width"]
    img_h = data["size"]["height"]

    output_lines = []

    for obj in data["objects"]:

        # hanya polygon
        if obj["geometryType"] != "polygon":
            continue

        class_name = obj["classTitle"]

        # skip class tidak dikenal
        if class_name not in CLASS_MAP:
            print(f"Skip unknown class: {class_name}")
            continue

        class_id = CLASS_MAP[class_name]

        polygon = obj["points"]["exterior"]

        # minimal polygon valid
        if len(polygon) < 3:
            continue

        yolo_polygon = convert_polygon_to_yolo(
            polygon,
            img_w,
            img_h
        )

        line = f"{class_id} {yolo_polygon}"
        output_lines.append(line)

    # save txt
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    output_txt = os.path.join(OUTPUT_LABEL_DIR, base_name + ".txt")

    with open(output_txt, "w") as f:
        f.write("\n".join(output_lines))

    print(f"Converted: {json_file} -> {output_txt}")

print("Done!")