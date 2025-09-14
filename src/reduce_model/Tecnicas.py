import os
import copy
from torch.nn.utils import prune
from torch import load
from torch import nn
from torch.utils.data import DataLoader


from src.damage_detector.CommonPath import CommonPath
from src.damage_detector.ConfigParams import ConfigParams
from src.damage_detector.ParserArguments import ParserArguments
from src.models.Autoencoder import Autoencoder
from src.models.CustomDataset import CustomDataset
from src.damage_detector.utils import __get_device, build_model_folder_path, load_data


def evaluate_model(model_, validation_loader_, device_to_use_, criterion_):
    validation_error = []

    for validation_batch in validation_loader_:
      validation_signals = validation_batch.to(device_to_use_)

      val_output = model_(validation_signals)
      validation_loss = criterion_(val_output, validation_signals.data)
      validation_error.append(validation_loss.item())

    return sum(validation_error) / len(validation_error)


if __name__ == '__main__':
    args = ParserArguments()

    # Load configs
    config_params = ConfigParams.load(os.path.join(CommonPath.CONFIG_FILES_FOLDER.value, args.config_filename))
    batch_size = config_params.get_params('train_params')['batch_size']
    sequences_length = config_params.get_params('global_variables').get('sequences_length')

    # Load data
    train_data, validation_data = load_data(config_params, is_train=True)
    validation_set = CustomDataset(validation_data)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    # Create the model
    device_to_use = __get_device()

    model_folder = build_model_folder_path(args.model_id, config_params.get_params('id'), args.folder_name)
    model_path = os.path.join(model_folder, 'model_trained.pth')

    model = Autoencoder(sequences_length)
    model.load_state_dict(load(model_path))
    model.eval()
    model.to(device_to_use)

    pruned_model_l1 = copy.deepcopy(model).to(device_to_use)
    pruned_model_l2 = copy.deepcopy(model).to(device_to_use)

    # Prune estructurado eliminando filas (neuronas de salida)
    prune.ln_structured(
        module=pruned_model_l1.decoder[0],  # Primera capa
        name='weight',
        amount=0.1,              # Proporción de filas (neuronas) a eliminar
        n=1,                   # Usamos norma L1 para determinar importancia
        dim=0)                   # Filas: Desactiva neuronas completas

    # Prune estructurado eliminando filas (neuronas de salida)
    prune.ln_structured(
        module=pruned_model_l2.decoder[0],  # Primera capa
        name='weight',
        amount=0.1,  # Proporción de filas (neuronas) a eliminar
        n=2,  # Usamos norma L1 para determinar importancia
        dim=0)  # Filas: Desactiva neuronas completas

    criterion = nn.MSELoss()
    base_loss = evaluate_model(model, validation_loader, device_to_use, criterion)
    mean_loss_l1 = evaluate_model(pruned_model_l1, validation_loader, device_to_use, criterion)
    mean_loss_l2 = evaluate_model(pruned_model_l2, validation_loader, device_to_use, criterion)
    print(f"Base validation loss {base_loss}")
    print(f"Best validation loss (L1) {mean_loss_l1}")
    print(f"Best validation loss (L2) {mean_loss_l2}")

    if args.save:
        # Save the results
        # Save the mask
        pass
