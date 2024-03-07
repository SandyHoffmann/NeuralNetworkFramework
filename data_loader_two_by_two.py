import numpy as np
import random
def get_data_sets():
    examples = [
        # np.array([[0, 0], [0, 0]]),
        # np.array([[0, 0], [0, 1]]),
        # np.array([[0, 0], [1, 0]]),
        # np.array([[0, 0], [1, 1]]),
        # np.array([[0, 1], [0, 0]]),
        # np.array([[0, 1], [0, 1]]),
        # np.array([[0, 1], [1, 0]]),
        # np.array([[0, 1], [1, 1]]),
        # np.array([[1, 0], [0, 0]]),
        # np.array([[1, 0], [0, 1]]),
        # np.array([[1, 0], [1, 0]]),
        # np.array([[1, 0], [1, 1]]),
        # np.array([[1, 1], [0, 0]]),
        # np.array([[1, 1], [0, 1]]),
        # np.array([[1, 1], [1, 0]]),
        # np.array([[1, 1], [1, 1]])
    ]

    for i in range(16):
        examples.append(np.array([[i%2, i//2%2], [i//4%2, i//8%2]]))
    
    random.shuffle(examples)
    print(examples)

    #80% dos dados para treino
    # * Yield retorna apenas um objeto por vez, e quando for pedido
    def training_set():
        while True:
            index = random.randint(0, len(examples) - 1)
            yield examples[index]

    #20% dos dados para evaluação
    def evaluation_set():
        while True:
            index = random.randint(0, len(examples) - 1)
            yield examples[index]

    return training_set, evaluation_set
