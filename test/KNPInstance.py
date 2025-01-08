
from Instances import Instances

class KnapsackInstance(Instances):
    def __init__(self, file_path: str):
        """
        Extiende la clase base para una instancia del problema Knapsack.
        """
        super().__init__(file_path)
        self.capacity = None
        self.num_items = None
        self.weights = []
        self.values = []
        self.load()

    def load(self):
        """
        Implementación del método para cargar una instancia Knapsack.
        """
        with open(self.file_path, 'r') as file:
            lines = file.readlines()

        self.capacity = int(lines[1].strip())
        self.num_items = int(lines[0].strip())
        

        for line in lines[3:]:
            linea = line.strip()
            linea = linea.split(" ")  
            self.values.append(int(linea[0]))
            self.weights.append(int(linea[1]))
            

    def get_instance_data(self):
        """
        Devuelve los datos específicos de la instancia Knapsack.
        """
        return {
            "capacity": self.capacity,
            "num_items": self.num_items,
            "weights": self.weights,
            "values": self.values,
        }