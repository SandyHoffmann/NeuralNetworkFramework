from data_loader_two_by_two import get_data_sets 
from nn_framework.framework import ANN
from nn_framework.layer import Dense
def run_framework():
    training_set, evaluation_set = get_data_sets()

    exemplo = next(training_set())
    
    numero_pixels = exemplo.shape[0] * exemplo.shape[1]
    numero_nodes = [numero_pixels, numero_pixels]
    model = [Dense(n_input=numero_nodes[0], n_output=numero_nodes[1])]
    print(model[0].weights)
    print(numero_pixels, numero_nodes)

    autoencoder = ANN(model=model, range_values=[0, 1])

    # autoencoder.train(training_set)
    # autoencoder.evaluate(evaluation_set)

    return numero_pixels, numero_nodes


run_framework()
