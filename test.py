import os
import json
import time

from torch import load, cuda
from torch.utils.data import DataLoader
from torch import nn

from src.damage_detector.AnomalyDetector import AnomalyDetector
from src.damage_detector.CommonPath import CommonPath
from src.damage_detector.ConfigParams import ConfigParams
from src.damage_detector.Metrics import Metrics
from src.damage_detector.ParserArguments import ParserArguments
from src.damage_detector.Results import Results
from src.models.AutoencoderGA import Autoencoder
from src.models.CustomDataset import CustomDataset
from src.damage_detector.utils import __get_device, build_model_folder_path, load_data

if __name__ == '__main__':
    args = ParserArguments()

    # Load configs
    config_params = ConfigParams.load(os.path.join(CommonPath.CONFIG_FILES_FOLDER.value, args.config_filename))
    batch_size = config_params.get_params('train_params')['batch_size']
    sequences_length = config_params.get_params('global_variables').get('sequences_length')

    # Load the data
    test_data, validation_data = load_data(config_params, is_train=False)

    test_set = CustomDataset(test_data)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    validation_set = CustomDataset(validation_data)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    device_to_use = __get_device()

    # Load the trained model
    model_folder = build_model_folder_path(args.model_id, config_params.get_params('id'), args.folder_name)
    model_path = os.path.join(model_folder, 'model_trained.pth')

    model_dimension_path = os.path.join(model_folder, 'model_dimensions.json')
    if os.path.isfile(model_dimension_path):
        with open(model_dimension_path, 'r') as model_dimension_file:
            model_dimension = json.load(model_dimension_file)

        encoder_size = model_dimension['first']
        bottleneck_size = model_dimension['bottleneck']
        decoder_size = model_dimension['decoder']
    else:
        encoder_size, bottleneck_size = list(map(lambda x: int(x), args.model_id.split('_')[1].split('x')))
        decoder_size = encoder_size

    trained_model = Autoencoder(sequences_length, '', encoder_size, bottleneck_size, decoder_size)
    trained_model.load_state_dict(load(model_path))
    trained_model.eval()
    trained_model.to(device_to_use)

    memory_before_inference = cuda.memory_allocated(device_to_use) / (1024 * 1024)

    anomaly_detector = AnomalyDetector(trained_model, config_params, device_to_use)
    results: Results = anomaly_detector.detect_damage(test_loader, validation_loader)

    validation_error = []
    criterion = nn.MSELoss()

    start_time = time.time()
    for validation_batch in validation_loader:
        validation_signals = validation_batch.to(device_to_use)

        val_output = trained_model(validation_signals)
        validation_loss = criterion(val_output, validation_signals.data)

    validation_error.append(validation_loss.item())

    memory_after_inference = cuda.memory_allocated(device_to_use) / (1024 * 1024)
    inference_time = time.time() - start_time
    mean_validation_loss = sum(validation_error) / len(validation_error)

    if args.save:
        metrics: Metrics = Metrics(
            results,
            mean_validation_loss,
            memory_before_inference,
            memory_after_inference,
            inference_time
        )
        results_path = os.path.join(model_folder, 'metrics.json')
        metrics.save(results_path)
