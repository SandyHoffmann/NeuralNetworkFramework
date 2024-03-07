from data_loader_two_by_two import get_data_sets 
from nn_framework.framework import ANN

def run_framework():
    training_set, evaluation_set = get_data_sets()

    exemplo = next(training_set())
    
    numero_pixels = exemplo.shape[0] * exemplo.shape[1]
    numero_nodes = [numero_pixels, numero_pixels]

    print(numero_pixels, numero_nodes)

    autoencoder = ANN()

    autoencoder.train(training_set)
    autoencoder.evaluate(evaluation_set)

    return numero_pixels, numero_nodes


run_framework()
