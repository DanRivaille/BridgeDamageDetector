from src.optimization.objective_function.ObjectiveFunction import ObjectiveFunction
import copy
import numpy as np
from torch.nn.utils import prune
from torch import nn
from torch.optim import Adam


class BridgeObjectiveFunction(ObjectiveFunction):
  def __init__(self, is_minimization: bool, model, train_loader, validation_loader, learning_rate, num_epochs,
               device_to_use, proportion_rate):
    super().__init__(is_minimization) 
    self.__model = model
    self.__train_loader = train_loader
    self.__validation_loader = validation_loader
    self.__learning_rate = learning_rate
    self.__num_epochs = num_epochs
    self.__device_to_use = device_to_use
    self.__proportion_rate = proportion_rate

  def train_mask_fine_tuning(self, model):
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=self.__learning_rate)

    train_error = []
    validation_error = []

    for epoch in range(self.__num_epochs):
      valid_loss = 0
           
      for train_batch in self.__train_loader:
          
        signals = train_batch.to(self.__device_to_use)

        output = model(signals)
        loss = criterion(output, signals.data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
      for validation_batch in self.__validation_loader:
        validation_signals = validation_batch.to(self.__device_to_use)
        # Revisar - calcular promedio
        val_output = model(validation_signals)
        validation_loss = criterion(val_output, validation_signals.data)
        print(F"VALID LOSS {validation_loss}")
        valid_loss += validation_loss.item()
        
      print()

      train_error.append(loss.item())
      # validation_error.append(validation_loss.item())
      # print(f"validation_loss = {validation_loss.item()}")
      validation_error.append(valid_loss/len(self.__validation_loader))
      print(f"validation_loss = {valid_loss/len(self.__validation_loader)}")

      print(f'epoch [{epoch + 1}/{self.__num_epochs}], loss:{loss.item(): .4f}')
        
    return validation_error[-1]
  
  def train_mask_evaluate_model(self, model):
    criterion = nn.MSELoss()
    validation_error = []

    for validation_batch in self.__validation_loader:  
      validation_signals = validation_batch.to(self.__device_to_use)

      # Revisar - calcular promedio
      val_output = model(validation_signals)
      validation_loss = criterion(val_output, validation_signals.data)
      validation_error.append(validation_loss.item())

    promedio_loss = sum(validation_error)/len(validation_error)
    print(f"validation_loss = {promedio_loss}")

    return promedio_loss

  def evaluate(self, mask) -> float:
    array = mask.numpy()  
    weigh_count = np.sum(array)
    largo = np.size(array)
    if (weigh_count/largo) > self.__proportion_rate:
      print("invalido")
      return 100.0

    modelo_copia = copy.deepcopy(self.__model)
    modelo_copia = modelo_copia.to(self.__device_to_use)

    self.apply_pruning(modelo_copia, mask)

    validation_error = self.train_mask_evaluate_model(modelo_copia)
    
    return validation_error
  
  def apply_pruning(self, modelo, mask):
    mask = mask.to(self.__device_to_use)

    if modelo.layer_to_mask == "first":
      mask = mask.unsqueeze(1).expand(-1, modelo.input_length)
      prune.custom_from_mask(modelo.encoder[0], name='weight', mask=mask)

    elif modelo.layer_to_mask == "bottleneck":
      mask = mask.unsqueeze(1).expand(-1, 32)
      prune.custom_from_mask(modelo.encoder[2], name='weight', mask=mask)

    elif modelo.layer_to_mask == "decoder":
      mask = mask.unsqueeze(1).expand(-1, 16)
      prune.custom_from_mask(modelo.decoder[0], name='weight', mask=mask)


      # mask1 = mask[:32].unsqueeze(1).expand(-1, modelo.input_length)
      # mask2 = mask[32:48].unsqueeze(1).expand(-1, 32)
      # mask3 = mask[48:80].unsqueeze(1).expand(-1, 16)

      # prune.custom_from_mask(modelo.encoder[0], name='weight', mask=mask1)
      # prune.custom_from_mask(modelo.encoder[2], name='weight', mask=mask2)
      # prune.custom_from_mask(modelo.decoder[0], name='weight', mask=mask3)
      
