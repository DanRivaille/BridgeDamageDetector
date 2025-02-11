from abc import ABC, abstractmethod


class Instances(ABC):

    def __init__(self, file_path: str):
        """
        Clase base para representar una instancia genérica.
        """
        self.file_path = file_path

    @abstractmethod
    def load(self):
        """
        Método genérico para cargar una instancia desde un archivo.
        Debe ser implementado por las subclases.
        """
        pass
    