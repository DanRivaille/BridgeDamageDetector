import os
import time
import torch
from torch.nn.utils import prune
from torch import load
from torch import nn
from torch.utils.data import DataLoader
from torch import save


from src.damage_detector.CommonPath import CommonPath
from src.damage_detector.ConfigParams import ConfigParams
from src.damage_detector.ParserArguments import ParserArguments
from src.models.Autoencoder import Autoencoder
from src.models.CustomDataset import CustomDataset
from src.damage_detector.utils import __get_device, build_model_folder_path, load_data





if __name__ == '__main__':
    args = ParserArguments()
    config_params = ConfigParams.load(os.path.join(CommonPath.CONFIG_FILES_FOLDER.value, args.config_filename))

    sequences_length = config_params.get_params('global_variables').get('sequences_length')
    train_data, validation_data = load_data(config_params, is_train=True)

    device_to_use = __get_device()
    batch_size = config_params.get_params('train_params')['batch_size']

    validation_set = CustomDataset(validation_data)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    model_folder = build_model_folder_path(args.model_id, config_params.get_params('id'), args.folder_name)
    model_path = os.path.join(model_folder, 'model_trained.pth')

    model = Autoencoder(sequences_length)
    model.load_state_dict(load(model_path))
    model.eval()
    model.to(device_to_use)

    """
    Indicar la mascara y capa a testear.
    Dejar comentado/borrar en caso de solo testear el modelo.
    """
    mask = torch.tensor([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])
    mask = mask.to(device_to_use)
    mask = mask.unsqueeze(1).expand(-1, 512)
    prune.custom_from_mask(model.encoder[0], name='weight', mask=mask)
    
    criterion = nn.MSELoss()
    validation_error = []
    
    start_time = time.time()
    for validation_batch in validation_loader:  
      validation_signals = validation_batch.to(device_to_use)

      val_output = model(validation_signals)
      validation_loss = criterion(val_output, validation_signals.data)
      validation_error.append(validation_loss.item())
    elapsed_time = time.time() - start_time

    promedio_loss = sum(validation_error)/len(validation_error)
    
    print(f"Tiempo de ejecucion {elapsed_time}")
    print(f"Best validation loss {promedio_loss}")

    # if args.save:
    #     model_folder = build_model_folder_path(args.model_id, config_params.get_params('id'), args.folder_name)
    #     os.makedirs(model_folder, exist_ok=True)

    #     model_path = os.path.join(model_folder, 'model_trained.pth')
    #     save(model.state_dict(), model_path)



