import os
import json

import torch
from torch import load
from torch import nn
from torch import save

from src.damage_detector.CommonPath import CommonPath
from src.damage_detector.ConfigParams import ConfigParams
from src.damage_detector.ParserArguments import ParserArguments
from src.models.AutoencoderGA import Autoencoder
from src.damage_detector.utils import __get_device, build_model_folder_path


def reconstruct_lineal_layer(current_layer: nn.Linear, mascara: torch.Tensor):

    assert mascara.numel() == current_layer.out_features, "La máscara debe tener una longitud igual al número de neuronas de salida."

    active_indexes = torch.nonzero(mascara).squeeze()
    new_dimension = active_indexes.numel()
    
    reconstructed_layer = nn.Linear(current_layer.in_features, new_dimension, bias=(current_layer.bias is not None))
    
    reconstructed_layer.weight.data = current_layer.weight.data[active_indexes, :].clone()
    if current_layer.bias is not None:
        reconstructed_layer.bias.data = current_layer.bias.data[active_indexes].clone()
        
    return reconstructed_layer, active_indexes


def reconstruct_lineal_layer_input(current_layer: nn.Linear, active_indexes: torch.Tensor):
    new_dimension = active_indexes.numel()
    new_layer = nn.Linear(new_dimension, current_layer.out_features, bias=(current_layer.bias is not None))
    
    new_layer.weight.data = current_layer.weight.data[:, active_indexes].clone()
    if current_layer.bias is not None:
        new_layer.bias.data = current_layer.bias.data[active_indexes].clone()
        # new_layer.bias.data = current_layer.bias.data.clone()
        
    return new_layer


def reimplement_layer(prev_layer: nn.Linear, next_layer: nn.Linear, mask: torch.Tensor) -> tuple:
    new_prev_layer, active_indexes = reconstruct_lineal_layer(prev_layer, mask)
    new_next_layer = reconstruct_lineal_layer_input(next_layer, active_indexes)
    return new_prev_layer, new_next_layer


def reimplement_model(model: Autoencoder, layer_to_mask: str, mask: torch.Tensor):
    if layer_to_mask == 'first':
        model.encoder[0], model.encoder[2] = reimplement_layer(model.encoder[0], model.encoder[2], mask)
    elif layer_to_mask == 'bottleneck':
        model.encoder[2], model.decoder[0] = reimplement_layer(model.encoder[2], model.decoder[0], mask)
    elif layer_to_mask == 'decoder':
        model.decoder[0], model.decoder[2] = reimplement_layer(model.decoder[0], model.decoder[2], mask)


def main():
    args = ParserArguments()

    config_params = ConfigParams.load(os.path.join(CommonPath.CONFIG_FILES_FOLDER.value, args.config_filename))

    sequences_length = config_params.get_params('global_variables').get('sequences_length')
    device_to_use = __get_device()
    layer_to_mask = config_params.get_params('ga_params')['to_mask']

    # Load the base model
    base_model_folder = build_model_folder_path(args.model_id, config_params.get_params('id'), 'base_models')
    model_path = os.path.join(base_model_folder, 'model_trained.pth')

    encoder_size, bottleneck_size = list(map(lambda x: int(x), args.model_id.split('_')[1].split('x')))
    decoder_size = encoder_size

    model = Autoencoder(sequences_length, layer_to_mask, encoder_size, bottleneck_size, decoder_size)
    model.load_state_dict(load(model_path))
    model.eval()
    model.to(device_to_use)

    # Load the best mask
    result_file_path = os.path.join(base_model_folder, 'optimization_results.json')
    assert os.path.isfile(result_file_path), "Optimization results file doesn't exists!"

    with open(result_file_path, 'r') as results_file:
        results = json.load(results_file)

    mask = torch.tensor(results['best_individual']).to(device_to_use)

    # Reimplement the model using the mask
    reimplement_model(model, layer_to_mask, mask)

    # Save the reimplemented model
    if args.save:
        pruned_model_folder = build_model_folder_path(args.model_id, config_params.get_params('id'), 'pruned_models')
        os.makedirs(pruned_model_folder, exist_ok=True)

        print("Saving the reimplementation of the model")
        model_path = os.path.join(pruned_model_folder, 'model_reduced.pth')
        save(model.state_dict(), model_path)

        model_dimension_path = os.path.join(pruned_model_folder, 'model_dimensions.json')
        model_dimension = {
            "first": model.encoder[0].out_features,
            "bottleneck": model.encoder[2].out_features,
            "decoder": model.decoder[0].out_features
        }
        with open(model_dimension_path, 'w') as model_dimension_file:
            json.dump(model_dimension, model_dimension_file)


if __name__ == '__main__':
    main()
