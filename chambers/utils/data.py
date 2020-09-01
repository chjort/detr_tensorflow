import time


def time_dataset(dataset, epochs=1, n=None):
    st = time.time()
    for i, sample in dataset.repeat(epochs).enumerate():
        print("{}/{}".format(i+1, n), end="\r")

    end_time = time.time() - st
    return end_time
