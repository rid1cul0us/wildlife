##
## convert megaDetector output to a simple version
##
## mega_raw.json : megaDetecter version(a.5.0) threshold 0.15
## mega_raw_simple_raw_simple_fi

import os
import json
from pathlib import Path
from typing import Optional


## threshold >= 0.2
##
def mega_label_rebuild(
    src_path: Path,
    threshold: float = 0.2,
    dump_path: Optional[Path] = None,
) -> dict:
    # input:  'file', 'max_detection_conf', 'detections'
    # output: 'file': true or false

    with open(src_path, "r") as f:
        mega_raw = json.load(f)

    img_y = mega_raw["images"]
    simple_mega = {}

    ## TODO
    threshold = max(threshold, 0.2)

    for info in img_y:
        filename = str(info["file"]).rsplit("/", 1)[-1]
        if info["max_detection_conf"] >= threshold:
            simple_mega[filename] = True
        else:
            simple_mega[filename] = False

    if dump_path is not None:
        if os.path.isdir(dump_path):
            with open(
                os.path.join(dump_path, f"mega_label_p{threshold*100:.0f}.json"), "w"
            ) as f:
                json.dump(simple_mega, f, indent=4)
        else:
            with open(os.path.join(dump_path), "w") as f:
                json.dump(simple_mega, f, indent=4)
    return simple_mega


if __name__ == "__main__":
    mega_label_rebuild(
        src_path="/root/data/dataset/wilds_nop/metadata/mega_raw/mega_raw_p20.json",
        threshold=0.2,
        dump_path="/root/data/dataset/wilds_nop/metadata/mega_label/mega_label_p20_from_r20.json",
    )
