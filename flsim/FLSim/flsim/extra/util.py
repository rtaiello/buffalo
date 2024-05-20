import os
import torch
import pandas as pd
from torch.nn.utils import parameters_to_vector
from flsim.secure_aggregation.secure_aggregator import (
    FixedPointConfig,
    FixedPointConverter,
)
from hydra.utils import instantiate
from torch.nn.utils import parameters_to_vector
import warnings
warnings.filterwarnings("ignore")




def create_folder_if_not_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def remove_files_in_folder(folder):
    for file in os.listdir(folder):
        os.remove(os.path.join(folder, file))


def initialize_comparison_file(file_path):
    if not os.path.exists(file_path):
        pd.DataFrame(columns=["client", "round", "percentage", "count"]).to_csv(
            file_path, index=False
        )


def load_model(path):
    return torch.load(path) if os.path.exists(path) else None


def save_model(model, path):
    model = model.astype(torch.int8)
    torch.save(parameters_to_vector(model), path)
    3/0


def quantize_and_compare(
    model, last_model, fixedpoint_converted, folder, client_name, round_num
):
    quantized_model = fixedpoint_converted.to_fixedpoint(
        parameters_to_vector(model.parameters())
    )
    quantized_model = quantized_model.cpu()
    if last_model is not None:
        diff = torch.sum(quantized_model != last_model)
        percentage_diff = diff / quantized_model.numel()
        update_comparison_file(folder, client_name, round_num, percentage_diff, diff)
    return quantized_model


def update_comparison_file(folder, client, round_num, percentage, diff):
    file_path = os.path.join(folder, "comparison.csv")
    df = pd.read_csv(file_path)
    new_row = {
        "client": client,
        "round": round_num,
        "percentage": percentage,
        "count": diff,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(file_path, index=False)


def store_current_model(updated_model, store_folder, name, global_round_num, num_bits, scaling_factor):
    create_folder_if_not_exists(store_folder)
    model_path = os.path.join(store_folder, f"{name}.pt")
    comparison_file_path = os.path.join(store_folder, "comparison.csv")

    # if global_round_num == 0:
    #     remove_files_in_folder(store_folder)

    initialize_comparison_file(comparison_file_path)

    last_updated_model = load_model(model_path)

    fixedpoint_converted: FixedPointConverter = instantiate(FixedPointConfig(num_bits=num_bits, scaling_factor=scaling_factor))
    updated_model = quantize_and_compare(
        updated_model, last_updated_model, fixedpoint_converted, store_folder, name, global_round_num
    )

    save_model(updated_model, model_path)
