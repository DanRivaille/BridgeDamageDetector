import torch
import os
import time

from torch import load
from torch import nn
from torch import save

from src.damage_detector.CommonPath import CommonPath
from src.damage_detector.ConfigParams import ConfigParams
from src.damage_detector.ParserArguments import ParserArguments
from src.models.Autoencoder import Autoencoder
from src.damage_detector.utils import __get_device, build_model_folder_path, load_data


def rearmar_capa_lineal(capa: nn.Linear, mascara: torch.Tensor):

    assert mascara.numel() == capa.out_features, "La máscara debe tener una longitud igual al número de neuronas de salida."

    indices_activos = torch.nonzero(mascara).squeeze()
    nuevo_num_neuronas = indices_activos.numel()
    print(indices_activos)
    
    nueva_capa = nn.Linear(capa.in_features, nuevo_num_neuronas, bias=(capa.bias is not None))
    
    nueva_capa.weight.data = capa.weight.data[indices_activos, :].clone()
    if capa.bias is not None:
        nueva_capa.bias.data = capa.bias.data[indices_activos].clone()
        
    return nueva_capa, indices_activos

def rearmar_capa_lineal_input(capa: nn.Linear, indices_activos: torch.Tensor):
    nuevo_num_entradas = indices_activos.numel()
    nueva_capa = nn.Linear(nuevo_num_entradas, capa.out_features, bias=(capa.bias is not None))
    
    nueva_capa.weight.data = capa.weight.data[:, indices_activos].clone()
    print(capa.weight.data)
    if capa.bias is not None:
        nueva_capa.bias.data = capa.bias.data.clone()
        
    return nueva_capa


if __name__ == '__main__':
    args = ParserArguments()

    config_params = ConfigParams.load(os.path.join(CommonPath.CONFIG_FILES_FOLDER.value, args.config_filename))

    sequences_length = config_params.get_params('global_variables').get('sequences_length')
    device_to_use = __get_device()

    model_folder = build_model_folder_path(args.model_id, config_params.get_params('id'), args.folder_name)
    model_path = os.path.join(model_folder, 'model_trained.pth')

    model = Autoencoder(sequences_length)
    model.load_state_dict(load(model_path))
    model.eval()
    model.to(device_to_use)


    criterion = nn.MSELoss()
    validation_error = []

    start_time = time.time()

    mask = torch.tensor([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1]).to(device_to_use)

    capa = 'Encoder'
    
    if capa =='Encoder':
        new_layer, indices_activos = rearmar_capa_lineal(model.encoder[0], mask)
        model.encoder[0] = new_layer
        print("Después de modificar encoder[0]:")
        print("  encoder[0].out_features =", model.encoder[0].out_features)  


        new_layer = rearmar_capa_lineal_input(model.encoder[2], indices_activos)
        model.encoder[2] = new_layer
        print("Nueva dimensión de entrada para encoder[2]:", model.encoder[2].in_features)
    elif capa == 'Bottleneck':
        new_encoder2, indices_activos_enc2 = rearmar_capa_lineal(model.encoder[2], mask)
        model.encoder[2] = new_encoder2
        print("Después de modificar encoder[2]:")
        print("  encoder[2].out_features =", model.encoder[2].out_features)  

        new_decoder0 = rearmar_capa_lineal_input(model.decoder[0], indices_activos_enc2)
        model.decoder[0] = new_decoder0
        print("Nueva dimensión de entrada para decoder[0]:", model.decoder[0].in_features)
    else:
        new_layer, indices_activos = rearmar_capa_lineal(model.decoder[0], mask)
        model.decoder[0] = new_layer
        print("Después de modificar decoder[0]:")
        print("  decoder[0].out_features =", model.decoder[0].out_features)  

        new_layer = rearmar_capa_lineal_input(model.decoder[2], indices_activos)
        model.decoder[2] = new_layer
        print("Nueva dimensión de entrada para decoder[2]:", model.decoder[2].in_features)


    print(model)

    if args.save:
        model_folder = build_model_folder_path(args.model_id, config_params.get_params('id'), args.folder_name)
        os.makedirs(model_folder, exist_ok=True)

        model_path = os.path.join(model_folder, 'model_reduced.pth')
        save(model.state_dict(), model_path)
