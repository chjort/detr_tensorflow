import time


def time_dataset(dataset, epochs=1, n=None):
    epochs_times = []
    for i in range(epochs):
        st = time.time()
        for j, sample in dataset.enumerate():
            print("{}/{}".format(j+1, n), end="\r")
        end_time = time.time() - st
        print()
        print(" -", end_time)

        epochs_times.append(end_time)

    print(sum(epochs_times) / len(epochs_times))
    return epochs_times
