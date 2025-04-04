from src.damage_detector.Results import Results
from src.optimization.optimization_result.OptimizationResult import OptimizationResult


class Metrics(OptimizationResult):
  def __init__(self,
               test_results: Results,
               validation_loss: float,
               memory_used_before_inference: float,
               memory_used_after_inference: float,
               inference_time: float
               ):
    self.test_results: Results = test_results
    self.validation_loss = validation_loss
    self.memory_used_before_inference = memory_used_before_inference
    self.memory_used_after_inference = memory_used_after_inference
    self.inference_time = inference_time

  def to_json(self) -> dict:
    return {
      'test_results': self.test_results.to_json(),
      'validation_loss': self.validation_loss,
      'memory_before_inference': self.memory_used_before_inference,
      'memory_after_inference': self.memory_used_after_inference,
      'inference_time': self.inference_time
    }
