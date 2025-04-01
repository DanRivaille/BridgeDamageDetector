import os

from torch import load
from torch.utils.data import DataLoader

from src.damage_detector.AnomalyDetector import AnomalyDetector
from src.damage_detector.CommonPath import CommonPath
from src.damage_detector.ConfigParams import ConfigParams
from src.damage_detector.ParserArguments import ParserArguments
from src.models.Autoencoder import Autoencoder
from src.models.CustomDataset import CustomDataset
from src.damage_detector.utils import __get_device, build_model_folder_path, load_data

if __name__ == '__main__':
    args = ParserArguments()

    # Load configs
    config_params = ConfigParams.load(os.path.join(CommonPath.CONFIG_FILES_FOLDER.value, args.config_filename))
    batch_size = config_params.get_params('train_params')['batch_size']

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

    trained_model = Autoencoder(config_params.get_params('global_variables').get('sequences_length'))
    trained_model.load_state_dict(load(model_path))
    trained_model.eval()
    trained_model.to(device_to_use)

    anomaly_detector = AnomalyDetector(trained_model, config_params, device_to_use)
    results = anomaly_detector.detect_damage(test_loader, validation_loader)

    if args.save:
        results_path = os.path.join(model_folder, 'test_results.json')
        results.save(results_path)
