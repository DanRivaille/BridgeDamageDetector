import os
import time

from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import save

from src.damage_detector.History import History
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

    sequences_length = config_params.get_params('global_variables').get('sequences_length')
    n_features = 1

    # Load data
    train_data, validation_data = load_data(config_params, is_train=True)

    device_to_use = __get_device()

    # Create the model
    num_epochs = config_params.get_params('train_params')['num_epochs']
    batch_size = config_params.get_params('train_params')['batch_size']
    learning_rate = config_params.get_params('train_params')['learning_rate']

    train_set = CustomDataset(train_data)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    validation_set = CustomDataset(validation_data)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    model = Autoencoder(sequences_length)
    model.to(device_to_use)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()

    train_error = []
    validation_error = []

    for epoch in range(num_epochs):
        for train_batch in train_loader:
            signals = train_batch.to(device_to_use)

            output = model(signals)
            loss = criterion(output, signals.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for validation_batch in validation_loader:
            validation_signals = validation_batch.to(device_to_use)

            val_output = model(validation_signals)
            validation_loss = criterion(val_output, validation_signals.data)

        train_error.append(loss.item())
        validation_error.append(validation_loss.item())

        
        print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.item(): .4f}, valid_loss:{validation_loss.item(): .4f}')
        

    elapsed_time = time.time() - start_time

    print(f"Tiempo de ejecucion {elapsed_time}")
    print(f"Best train loss {min(train_error)}")
    print(f"Best validation loss {min(validation_error)}")

    learning_rate_updating = [learning_rate] * num_epochs
    history_obj = History(train_error, validation_error, learning_rate_updating, elapsed_time)

    if args.save:
        # Save the results
        model_folder = build_model_folder_path(args.model_id, config_params.get_params('id'), args.folder_name)
        os.makedirs(model_folder, exist_ok=True)

        model_path = os.path.join(model_folder, 'model_trained.pth')
        save(model.state_dict(), model_path)

        history_path = os.path.join(model_folder, 'train_history.json')
        history_obj.save(history_path)
