import time


def time_dataset(dataset, epochs=1):
    st = time.time()
    for sample in dataset.repeat(epochs):
        pass

    end_time = time.time() - st
    return end_time
