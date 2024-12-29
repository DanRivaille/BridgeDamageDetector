import torch
import copy
import random

from torch import nn
from torch.optim import Adam
from src.Plotter import Plotter
from torch import save


def train_mask(model, train_loader, validation_loader, neuron_mask,learning_rate, num_epochs, device_to_use):

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_error = []
    validation_error = []

    for epoch in range(num_epochs):        
        for train_batch in train_loader:
            signals = train_batch.to(device_to_use)

            output = model(signals, mask=neuron_mask)
            loss = criterion(output, signals.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for validation_batch in validation_loader:
            validation_signals = validation_batch.to(device_to_use)

            val_output = model(validation_signals, mask=neuron_mask)
            validation_loss = criterion(val_output, validation_signals.data)
            

        train_error.append(loss.item())
        validation_error.append(validation_loss.item())
        print(f"validation_loss = {validation_error[-1]}")

        
        print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.item(): .4f}')
        
    return validation_error


def elitist_selection(poblacion, num_padres):
    
    seleccionados = sorted(poblacion, key=lambda x: x[0])
    padres_seleccionados = []
    cont = 0
    for tupla in seleccionados:
        if cont == num_padres:
            break
        padre = tupla[1]
        fitne = tupla[0]
        print(f"Parent {cont+1} {fitne}")
        padres_seleccionados.append(padre)
        cont += 1
      
        
    return padres_seleccionados


def onepoint_crossover(padre1, padre2):
    puntoCruce = random.randint(1, len(padre1) - 1)
    hijo1 = torch.cat((padre1[:puntoCruce], padre2[puntoCruce:]))
    hijo2 = torch.cat((padre2[:puntoCruce], padre1[puntoCruce:]))

    return hijo1, hijo2


def uniform_crossover(padre1, padre2):
    hijo1, hijo2 = [], []

    for gen1, gen2 in zip(padre1, padre2):
        if random.random() < 0.5:
            hijo1.append(gen1)
            hijo2.append(gen2)
        else:
            hijo1.append(gen2)
            hijo2.append(gen1)
    
    hijo1 = torch.tensor(hijo1)
    hijo2 = torch.tensor(hijo2)
    
    return hijo1, hijo2

def bit_flip_mutation(genome, prob_mutacion):
    genome = genome.clone()
    for i in range(len(genome)):
        prob = random.random()
        if prob <= prob_mutacion and genome[i] == 0:
            genome[i] = 1
        elif prob <= prob_mutacion and genome[i] == 1:
            genome[i] = 0 
    
    return genome

def generar_poblacion_inicial(poblacion_size, mask_size):

    poblacion = []

    probabilidad = 0.8 # 80% de 1 y 20% de 0
    for _ in range(poblacion_size):
        
        mask = torch.bernoulli(torch.full((mask_size,), probabilidad)).int()
        
        poblacion.append(mask)

    return poblacion

def calcularFitness(model, train_loader, validation_loader,learning_rate, mask, num_epochs, device):
    modelo_copia = copy.deepcopy(model)
    modelo_copia = modelo_copia.to(device)
            
    mask = mask.to(device)

    validation_error = train_mask(modelo_copia, train_loader, validation_loader, mask, learning_rate, num_epochs, device)

    return min(validation_error)

def calcularFitnessPoblacion(poblacion, fitness_scores, model, train_loader, validation_loader,learning_rate, num_epochs, device):
    mejor_solucion = None
    mejor_fitness = float('inf')
    cont = 1
    for mask in poblacion:

        print(f"\n>{cont} mask a probar, comenzando train_mask\n")

        validation_error = calcularFitness(model, train_loader, validation_loader,learning_rate, mask, num_epochs, device)
        fitness_scores.append(validation_error)

        cont += 1
        if validation_error < mejor_fitness:
            mejor_fitness = validation_error
            mejor_solucion = mask.clone()
    
    return fitness_scores, mejor_fitness, mejor_solucion

def genetic_algorithm(model, train_loader, validation_loader,learning_rate, num_epochs, prob_mutacion, device):

    """
    La idea es que esta funcion llame a la funcion train_mask de train.py, ya que se evaluaria en el modelo las mascaras
    generadas por el algoritmo genetico, pasando por las 30 epochs que requiere el ciclo de entrenamiento establecido en 
    la configuracion.

    Una vez termine el algoritmo genetico la idea es que se encuentre la mejor solucion (mejor vector binario de mask). Con
    este vector binario encontrado, se puede aplicar el pruning especifico que indica el vector binario, donde recien el modelo
    pesaria menos. En un principio las mascaras que indican que una neurona esta "desactivada", no hacen que esta neurona NO este
    en el modelo.
    """
    poblacion_size = 10
    hijos = 5
    num_generaciones = 10

    """
    Si se quiere buscar mejores mascaras para unicamente la capa de cuello de botella, se cambia a un mask_size de 128.
    Cambiando tambien el Autoencoder.py, usando el forward de 128 neuronas para esta capa.
    """
    mask_size = 640 

    poblacion = generar_poblacion_inicial(poblacion_size, mask_size)

    print(f"Poblacion Inicial generada = {poblacion}")
    
    #memo = []
    generacion_track = []
    fitness_scores = []

    fitness_scores, mejor_fitness, mejor_solucion = calcularFitnessPoblacion(poblacion, fitness_scores, model, train_loader, validation_loader, learning_rate, num_epochs, device)
        
            
    print("Poblacion Inicial, Fitness")
    for i in range(poblacion_size):
        print(f"{fitness_scores[i]}")

    pobla = list(zip(fitness_scores, poblacion))

    for generacion in range(num_generaciones):
        
        parents = elitist_selection(pobla, 2)

        next_generation = []
        fitness_hijos = []
        cont_hijos = 0
        while len(next_generation) < hijos:
    
            offspring1, offspring2 = uniform_crossover(parents[0],parents[1])

            h1 = bit_flip_mutation(offspring1, prob_mutacion)
            cont_hijos += 1
            print(h1)
            print(f"\nCalculando fitess Hijo {cont_hijos} . . .")

            fitness_hijos.append(calcularFitness(model, train_loader, validation_loader,learning_rate, h1, num_epochs, device))
            print(f"Fitness Hijo {cont_hijos} = {fitness_hijos[cont_hijos-1]}\n")
            next_generation.append(h1)
            
            
            if len(next_generation) < hijos:
                h2 = bit_flip_mutation(offspring2, prob_mutacion)
                cont_hijos += 1
                print(h2)
                print(f"\nCalculando fitess Hijo {cont_hijos} . . .")
                fitness_hijos.append(calcularFitness(model, train_loader, validation_loader,learning_rate, h2, num_epochs, device))
                print(f"Fitness Hijo {cont_hijos} = {fitness_hijos[cont_hijos-1]}\n")
                next_generation.append(h2)


        for i in range(len(next_generation)):
            print(f"Hijo {i+1} | Fitness = {fitness_hijos[i]}")

        nueva_con_fitness = list(zip(fitness_hijos, next_generation))
        
        """
        Se mezcla la poblacion actual con los hijos generados
        """
        pobla_mix = nueva_con_fitness + pobla

        """
        Se ordena la poblacion mezclada con los hijos, en base al fitness
        """
        pobla_mix.sort(key=lambda x: x[0])

        print("\nPoblacion Mezclada con los Hijos generados")
        for tupla in pobla_mix:
            mask = tupla[1]
            validation_error = tupla[0]

            print(validation_error)
        print("")

        """
        Se corta la poblacion hasta el size de la poblacion original, dejando a los mejores hasta ese size,
        pudiendo quedar fuera tanto hijos nuevos como antiguos.
        """
        pobla = pobla_mix[0:poblacion_size]


        print(f"Poblacion generada en la generacion {generacion+1} ")
        for tupla in pobla:
            mask = tupla[1]
            validation_error = tupla[0]

            print(validation_error)


        for tupla in pobla:
            mask = tupla[1]
            validation_error = tupla[0]

            if validation_error < mejor_fitness:
                mejor_fitness = validation_error
                mejor_solucion = mask.clone()
        generacion_track.append(min(pobla, key=lambda x: x[0])[0])
        
        print(f"\nGeneración {generacion + 1}/{num_generaciones}, Mejor fitness: {mejor_fitness}\n")
            

    print("===============================================")
    for i in range(1, num_generaciones + 1):
        print(f"Generacion {i} | Mejor Fitness = {generacion_track[i-1]}")
    print()


    return mejor_solucion
