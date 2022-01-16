from argparse import ArgumentParser
import os
from time import sleep
from study import generators
from tqdm import tqdm
import pickle


parser = ArgumentParser(
    description='Generates study objects and saves with pickle.'
)

parser.add_argument(
    "study_generator",
    type=str,
    help=f"Generator to create studies. Must be on of these: {generators.keys()}"
    )
parser.add_argument(
    "-n", "--num_samples",
    default=1,
    type=int,
    required=False,
    help="Number of samples to create.",
    dest="num_samples"
)
parser.add_argument(
    "-o", "--output_dir",
    default=os.getcwd(),
    type=str,
    required=False,
    help="Directory to put the generated studies in.",
    dest="output_dir"
)


# verify arguments
args = parser.parse_args()

if not args.study_generator in generators.keys():
    raise ValueError("<study_generator> must be a valid generator.")

if not os.path.isdir(args.output_dir):
    raise ValueError("<output_dir> must be a valid path to an existing directory.")

# generate studies
study_generator = generators[args.study_generator]
print("Generating studies..")
for i in tqdm(range(args.num_samples)):
    s = study_generator.generate_study()
    s_name = s.name
    pickle_file_name = f'{s_name}.pickle'
    pickle_file_path = os.path.join(args.output_dir, pickle_file_name)
    with open(pickle_file_path, "wb") as f:
        pickle.dump(s, f)
