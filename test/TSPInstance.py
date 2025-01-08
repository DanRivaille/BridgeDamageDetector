
from Instances import Instances

class TSPInstance(Instances):
    def __init__(self, file_path: str):
        """
        Extiende la clase base para una instancia del problema TSP.
        """
        super().__init__(file_path)
        self.tour_size = None
        self.optTourknow = []
        self.optdistance = None
        self.matrix = []
        self.load()

    def load(self):
        """
        Implementación del método para cargar una instancia TSP.
        """
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
        
        self.tour_size = int(lines[0].strip())
        
        self.optTourknow = lines[2].strip().split(" ")
        self.optTourknow = [int(value) for value in self.optTourknow]
        
        self.optdistance = int(lines[4])
        
        for line in lines[6:]:
            tempLine = line.strip().split(" ")
            tempLine = [int(value) for value in tempLine]
            
            self.matrix.append(tempLine)
            

    def get_instance_data(self):
        """
        Devuelve los datos específicos de la instancia TSP.
        """
        return {
            "tour_size": self.tour_size,
            "opt_TourKnow": self.optTourknow,
            "opt_distance": self.optdistance,
            "distance_matrix": self.matrix,
        }