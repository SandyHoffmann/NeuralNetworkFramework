from data_loader_two_by_two import get_data_sets 
from nn_framework.framework import ANN
from nn_framework.layer import Dense
from nn_framework.activation import Tahn
def run_framework():
    N_NODES = [7, 4, 6]

    training_set, evaluation_set = get_data_sets()

    exemplo = next(training_set())
    
    numero_pixels = exemplo.shape[0] * exemplo.shape[1]
    numero_nodes = [numero_pixels]+  N_NODES + [numero_pixels]
    model = []
    for n in range(len(numero_nodes) - 1):
        model.append(
            Dense(n_input=numero_nodes[n], n_output=numero_nodes[n+1], activation=Tahn())
        )
    autoencoder = ANN(model=model, range_values=[0, 1])
    # print(model[0].forward_prop(exemplo))
    autoencoder.train(training_set)
    # autoencoder.evaluate(evaluation_set)

    return numero_pixels, numero_nodes


run_framework()
