from typing import Iterable, Tuple
import patran_parsing
import lxml.objectify
import os
import pandas as pd
from tqdm import tqdm
from util import parse_to_float_list


def clean_whole_dataset(path_to_dataset: str) -> None:
    """Cleans every study in a dataset."""
    file_names = os.listdir(path_to_dataset)
    file_paths = [os.path.join(path_to_dataset, fn) for fn in file_names]
    study_dir_paths = [dir for dir in file_paths if os.path.isdir(dir)]

    for path_to_study_dir in tqdm(study_dir_paths):
        clean_study(path_to_study_dir)


def clean_study(path_to_study_dir: str) -> None:
    """Extracts node information and study results and writes them into
       a csv file in the same directory
    """
    # get nodes
    file_names = os.listdir(path_to_study_dir)
    pat_file_name = [fn for fn in file_names if fn.endswith(".pat")][0]
    pat_file_path = os.path.join(path_to_study_dir, pat_file_name)
    node_dict = _read_node_information(pat_file_path)

    # get study results
    xml_file_names = [fn for fn in file_names if fn .endswith(".xml")]
    xml_file_paths = [os.path.join(path_to_study_dir, fn) for fn in xml_file_names]
    feature_file_names = [os.path.splitext(fn)[0] for fn in xml_file_names]
    feature_names = ["_".join(ffn.split("_")[-2:]) for ffn in feature_file_names]
    study_result_dicts = [_read_study_result(xfp) for xfp in xml_file_paths]

    # write to DataFrame
    df_study = pd.DataFrame(node_dict.items(), columns=("id", "position"))
    l_df_study_result = [pd.DataFrame(srd.items(), columns=("id", fn)) for fn, srd in zip(feature_names, study_result_dicts)]
    for df_study_result in l_df_study_result:
        df_study = df_study.merge(df_study_result, on="id", how="left", sort=True)

    # save as csv
    csv_file_path = os.path.join(path_to_study_dir, "nodes.csv")
    df_study.to_csv(csv_file_path, index=False)


def _read_node_information(path_to_patran_file: str) -> dict[int, Tuple[float, float, float]]:
    """Extracts node information from a patran file"""
    df_nodes, df_tetras = patran_parsing.parse_patran_file(path_to_patran_file)
    ids = df_nodes.id
    positions = [(x, y, z) for x,y,z in zip(df_nodes.x, df_nodes.y, df_nodes.z)]
    return {k: v for k, v in zip(ids, positions)}


def _read_study_result(path_to_xml_file: str) -> dict[int, float]:
    """Extracts study results from xml file"""
    with open(path_to_xml_file, 'r', encoding="windows-1252") as f:
        xml = f.read()
    root = lxml.objectify.fromstring(xml)
    nodes = root.Dataset.Blocks.Block.Data.getchildren()

    #get value for each node
    ids = [int(n.get("ID")) for n in nodes]
    values = [float(n.DeptValues.text) for n in nodes]
    study_result = {k: v for k, v in zip(ids, values)}
    return study_result


def load_study(path_to_study_dir: str) -> pd.DataFrame:
    """Loads a DataFrame containing for each node: id, position,
       fill_time, weld_line, weld_surface"""
    csv_file_path = os.path.join(path_to_study_dir, "nodes.csv")
    if not csv_file_path in os.listdir(path_to_study_dir):
        clean_study(path_to_study_dir)
    
    df_study = pd.read_csv(csv_file_path)
    df_study.position = df_study.position.apply(lambda x: parse_to_float_list(x))
    return df_study