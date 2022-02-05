import pandas as pd


def parse_header(header_lines):
    header = ''.join(header_lines)
    header = header.replace('       ', ',')
    header = header.replace('      ', ',')
    header = header.replace('     ', ',')
    header = header.replace('    ', ',')
    header = header.replace('   ', ',')
    header = header.replace('  ', ',')
    header = header.replace(' ', ',')
    header = header.replace('\n', ',')
    header = header[:-1]
    header = header.split(',')
    tmp = []
    for x in header:
        try:
            tmp.append(int(x))
        except:
            pass
    header = tmp
    header = [n for n in header if n > 50]
    return header


def parse_nodes(node_lines, num_nodes):
    nodes = ''.join(node_lines)
    nodes = nodes.replace('       ', ',')
    nodes = nodes.replace('      ', ',')
    nodes = nodes.replace('     ', ',')
    nodes = nodes.replace('    ', ',')
    nodes = nodes.replace('   ', ',')
    nodes = nodes.replace('  ', ',')
    nodes = nodes.replace(' ', ',')
    nodes = nodes.replace('\n', ',')
    nodes = nodes.replace(',,', ',')
    nodes = nodes.replace('E-', 'ne')
    nodes = nodes.replace('-', ',-')
    nodes = nodes.replace('ne', 'E-')
    nodes = nodes.split(',')
    nodes = nodes[1:-1]
    nodes = [n for n in nodes if n != '']
    tmp = []
    for x in nodes:
        try:
            tmp.append(int(x))
        except:
            try:
                tmp.append(float(x))
            except:
                tmp.append(x)

    assert num_nodes == len(nodes) / 17, 'Number of parsed nodes must match number of expected nodes'

    nodes = tmp
    nodes = [nodes[i * 17: i * 17 + 18] for i in range(num_nodes)]

    return nodes


def parse_tetras(tetra_lines, num_tetras):
    tetras = ''.join(tetra_lines)

    tetras = tetras.replace('       ', ',')
    tetras = tetras.replace('      ', ',')
    tetras = tetras.replace('     ', ',')
    tetras = tetras.replace('    ', ',')
    tetras = tetras.replace('   ', ',')
    tetras = tetras.replace('  ', ',')
    tetras = tetras.replace(' ', ',')
    tetras = tetras.replace('\n', ',')
    tetras = tetras.replace(',,', ',')

    tetras = tetras.split(',')
    tetras = tetras[1:-1]

    tmp = []
    for t in tetras:
        try:
            tmp.append(int(t))
        except:
            try:
                tmp.append(float(t))
            except:
                tmp.append(t)
    tetras = tmp
    tetras = [tetras[i * 20: i * 20 + 21] for i in range(num_tetras)]

    return tetras


def get_df_nodes(node_list):
    df_nodes = pd.DataFrame(node_list)

    # drop static values
    for c in df_nodes.columns:
        if df_nodes[c].dtype == 'O' or df_nodes[c].std() == 0.0:
            df_nodes.drop(c, axis=1, inplace=True)
    # rename columns
    df_nodes.columns = ["id", "x", "y", "z"]
    return df_nodes


def get_df_tetras(tetra_list):
    df_tetras = pd.DataFrame(tetra_list)

    # drop static values
    for c in df_tetras.columns:
        if df_tetras[c].std() == 0.0:
            df_tetras.drop(c, axis=1, inplace=True)

    # rename columns
    df_tetras.columns = ['id', 'v1', 'v2', 'v3', 'v4']

    # let tetras be indexed by their ids
    df_tetras.set_index('id', inplace=True)

    return df_tetras


def parse_patran_file(path_to_patran_file: str):
    with open(path_to_patran_file, 'r') as file:
        lines = file.readlines()

    header_lines = lines[:4]
    num_nodes, num_tetras, *_ = parse_header(header_lines)

    node_start_index = 4  # the header takes up the first 4 lines so nodes start at index 4
    node_end_index = node_start_index + 3 * num_nodes

    tetra_start_index = node_end_index
    tetra_end_index = tetra_start_index + 3 * num_tetras

    assert tetra_end_index == len(lines) - 1, 'Number of nodes and tetras must match header description'

    node_lines = lines[node_start_index: node_end_index]
    tetra_lines = lines[tetra_start_index: tetra_end_index]

    nodes = parse_nodes(node_lines, num_nodes)
    tetras = parse_tetras(tetra_lines, num_tetras)

    # remove unused information and put into dataframe for simpler usage
    df_nodes = get_df_nodes(nodes)
    df_tetras = get_df_tetras(tetras)

    return df_nodes, df_tetras