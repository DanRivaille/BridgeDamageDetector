import torch
import torch.nn as nn
import os
import time

from torch import nn
from torch.utils.data import DataLoader
from src.damage_detector.CommonPath import CommonPath
from src.damage_detector.ConfigParams import ConfigParams
from src.damage_detector.ParserArguments import ParserArguments
from src.models.Autoencoder import Autoencoder
from src.models.CustomDataset import CustomDataset
from src.damage_detector.utils import __get_device, build_model_folder_path, load_data

def medir_tiempo_inferencia(modelo, validation_loader, device="cuda"):
    modelo.to(device)
    modelo.eval()

    criterion = nn.MSELoss()

    start_time = time.time()

    for validation_batch in validation_loader:
        validation_signals = validation_batch.to(device_to_use)

        val_output = model(validation_signals)
        validation_loss = criterion(val_output, validation_signals.data)
    
   
    elapsed_time = time.time() - start_time

    return elapsed_time  


args = ParserArguments()
config_params = ConfigParams.load(os.path.join(CommonPath.CONFIG_FILES_FOLDER.value, args.config_filename))
sequences_length = config_params.get_params('global_variables').get('sequences_length')
train_data, validation_data = load_data(config_params, is_train=True)

device_to_use = __get_device()

batch_size = config_params.get_params('train_params')['batch_size']

validation_set = CustomDataset(validation_data)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

model_folder = build_model_folder_path(args.model_id, config_params.get_params('id'), args.folder_name)

"""
Indicar el nombre del modelo a analizar las metricas.
"""
model_path = os.path.join(model_folder, 'model_reduced_ft.pth')

model = Autoencoder(sequences_length)
model.load_state_dict(torch.load(model_path))
model.to(device_to_use)
model.eval()

memoria_antes = torch.cuda.memory_allocated(device_to_use) / (1024 * 1024)  
print(f"Mmemoria al cargar el modelo: {memoria_antes}")
tiempo_inferencia = medir_tiempo_inferencia(model, validation_loader)
memoria_despues = torch.cuda.memory_allocated(device_to_use) / (1024 * 1024) 

print(f"Tiempo en inferencia  {tiempo_inferencia:.2f} Segundos")
print(f"Mmemoria despues de inferencia: {memoria_despues}")