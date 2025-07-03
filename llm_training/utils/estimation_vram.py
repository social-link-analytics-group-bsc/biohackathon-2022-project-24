import argparse
from transformers import AutoModel
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default=None,
    type=str,
    required=True,
    help="Give the config filename to the model to be run. ",
)

parser.add_argument(
    "--num_gpus_per_node",
    type=int,
    default=1,
    help="Number of GPU per node"
)

parser.add_argument(
    "--num_nodes",
    type=int,
    default=1
    help="Number of nodes",
)

args = parser.parse_args()
model = AutoModel.from_pretrained("bigscience/T0_3B")
estimate_zero3_model_states_mem_needs_all_live(args.model, num_gpus_per_node=args.num_gpus_node, num_nodes=args.num_nodes)
