import json
import os
from tqdm import tqdm
import meshio
from config import Config

INPUT_DIR = Config.DATA_DIR / "moldflow_out"
OUTPUT_DIR = Config.DATA_DIR / "raw"


def main():
    studies = sorted([sd for sd in os.listdir(INPUT_DIR) if not sd.startswith(".")])
    for sd in tqdm(studies, desc="clean_data"):
        study_dir = INPUT_DIR / sd
        pat_file = [study_dir / fn for fn in os.listdir(study_dir) if fn.endswith(".pat")][0]
        xml_files = [study_dir / fn for fn in os.listdir(study_dir) if fn.endswith("fill_time.xml")]

        # read moldflow output as mesh
        mesh = meshio.moldflow.read(pat_file, xml_filenames=xml_files)

        # extract values
        pos = mesh.points
        fill_time = mesh.point_data["Füllzeit"]

        # save as json file
        json_content = dict(
            pos=pos.tolist(),
            fill_time=fill_time.tolist()
        )
        json_file_name = f"{sd}.json"
        json_file_path = OUTPUT_DIR / json_file_name
        with open(json_file_path, "w") as json_file:
            json.dump(json_content, json_file)


if __name__ == "__main__":
    main()
