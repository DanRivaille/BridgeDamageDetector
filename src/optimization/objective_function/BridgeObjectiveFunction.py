import copy
from torch.nn.utils import prune
from torch import nn
from torch import sum as torch_sum

from src.optimization.objective_function.ObjectiveFunction import ObjectiveFunction


def bitwise_operation(mask: list) -> int:
  binary_mask = int(0)
  for bit in mask:
    binary_mask = (binary_mask << 1) | bit
  return binary_mask


def compute_key(mask, max_length) -> tuple:
  num_sub_keys = max(1, len(mask) // max_length)

  key = []
  for i in range(num_sub_keys):
    start_index = i * max_length
    end_index = start_index + max_length
    current_key = bitwise_operation(mask[start_index: end_index])
    key.append(current_key)

  return tuple(key)


def evaluate_model(model, validation_loader, device_to_use):
  criterion = nn.MSELoss()
  validation_error = []

  for validation_batch in validation_loader:
    validation_signals = validation_batch.to(device_to_use)

    val_output = model(validation_signals)
    validation_loss = criterion(val_output, validation_signals.data)
    validation_error.append(validation_loss.item())

  return sum(validation_error) / len(validation_error)


class BridgeObjectiveFunction(ObjectiveFunction):
  def __init__(self, is_minimization: bool, model, validation_loader, device_to_use, proportion_rate):
    super().__init__(is_minimization)
    self.model = model
    self.validation_loader = validation_loader
    self.device_to_use = device_to_use
    self.__proportion_rate = proportion_rate
    self.__historical_fitness = dict()

  def evaluate(self, mask) -> float:
    current_key = compute_key(mask, 32)

    if current_key not in self.__historical_fitness:
      self.__historical_fitness[current_key] = self.__compute_fitness(mask)

    return self.__historical_fitness[current_key]

  def __compute_fitness(self, mask):
    if (torch_sum(mask) / (mask.size(dim=0) * 1.0)) > self.__proportion_rate:
      return 100.0

    pruned_model = copy.deepcopy(self.model).to(self.device_to_use)
    self.__apply_pruning(pruned_model, mask)

    return evaluate_model(pruned_model, self.validation_loader, self.device_to_use)

  def __apply_pruning(self, modelo, mask):
    mask = mask.to(self.device_to_use)

    if modelo.layer_to_mask == "first":
      mask = mask.unsqueeze(1).expand(-1, modelo.input_length)
      prune.custom_from_mask(modelo.encoder[0], name='weight', mask=mask)

    elif modelo.layer_to_mask == "bottleneck":
      mask = mask.unsqueeze(1).expand(-1, modelo.encoder[2].in_features)
      prune.custom_from_mask(modelo.encoder[2], name='weight', mask=mask)

    elif modelo.layer_to_mask == "decoder":
      mask = mask.unsqueeze(1).expand(-1, modelo.decoder[0].in_features)
      prune.custom_from_mask(modelo.decoder[0], name='weight', mask=mask)

  def get_history(self) -> dict:
    return dict(sorted(self.__historical_fitness.items(), key=lambda item: item[1], reverse=True))
