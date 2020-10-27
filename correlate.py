import sys
from os import listdir, path

import numpy as np
import time
import math
from multiprocessing import cpu_count, Pool
import functools
from numba import jit
import pandas as pd

SQUARE_DIP_INDEX_INPUT = 3
SQUARE_DIR_INDEX_INPUT = 4
SQUARE_DIP_INDEX_RESULT = 0
SQUARE_DIR_INDEX_RESULT = 1


def do_correlate(target_dataframe, correlation_target_dataframe, thread_pool=None):
    start_time = time.time()

    target_dataframe_as_array = np.asarray(target_dataframe.to_numpy()[:,0:5], dtype=float)
    correlation_dataframe_as_array = np.asarray(correlation_target_dataframe.to_numpy(), dtype=float)

    # Execute the function in parallel
    result = thread_pool.map(functools.partial(correlate_row, correlation_target=correlation_dataframe_as_array), target_dataframe_as_array)
    result = np.array(result)

    result_with_mute_flag = result[target_dataframe[target_dataframe['Mute Flag'] == 'M'].index]
    result_without_mute_flag = result[target_dataframe[target_dataframe['Mute Flag'] != 'M'].index]

    mse_dip = np.average(result_without_mute_flag[:, 0])
    mse_dir = np.average(result_without_mute_flag[:, 1])
    mspe_dip = np.average(result_with_mute_flag[:, 0])
    mspe_dir = np.average(result_with_mute_flag[:, 1])

    print_result_to_console(mse_dip, mse_dir, mspe_dip, mspe_dir,
                            len(target_dataframe), len(correlation_target_dataframe),
                            time.time() - start_time)

    return [mse_dip, mse_dir, mspe_dip, mspe_dir]


def print_result_to_console(mse_dip, mse_dir, mspe_dip, mspe_dir, size_of_target, size_of_correlation, time_taken):
    print('Mean Square Error - Dip: {0}'.format(mse_dip))
    print('Mean Square Error - Dir: {0}'.format(mse_dir))
    print('Mean Square Predictive Error - Dip: {0}'.format(mspe_dip))
    print('Mean Square Predictive Error - Dir: {0}'.format(mspe_dir))
    print("--- Correlated %d x %d points in %.4f seconds ---" % (size_of_target, size_of_correlation, time_taken))


# Use Numba to JIT compile the below function for a considerable performance boost.
@jit
def correlate_row(current_row, correlation_target):
    min_distance = None
    current_min = None
    for point in correlation_target:
        # Take the dif_x, dif_y, dif_z out as we will need these later
        # when we expand the below calculation to include other data points.
        dif_x = point[0] - current_row[0]
        dif_y = point[1] - current_row[1]
        dif_z = point[2] - current_row[2]

        # Take the Euclidean distance of the two dot points - this is also the == l2 norm of current - target
        # distance = np.linalg.norm(current_row - target[0:3])
        distance = math.sqrt(
            (dif_x ** 2) +
            (dif_y ** 2) +
            (dif_z ** 2)
        )
        if min_distance is None or distance < min_distance:
            min_distance = distance
            current_min = point

    square_dip = (current_row[SQUARE_DIP_INDEX_INPUT] - current_min[SQUARE_DIP_INDEX_INPUT]) ** 2
    square_dir = (current_row[SQUARE_DIR_INDEX_INPUT] - current_min[SQUARE_DIR_INDEX_INPUT]) ** 2

    return [square_dip, square_dir]


def main():
    if len(sys.argv) != 4:
        print("Usage: correlate.py /path/to/input.csv /path/to/intepolation/dir /path/to/output.csv")
        exit(-1)

    start_time = time.time()
    # Create a thread pool (number of CPU threads, -1 spare for everything else)
    thread_pool=Pool(cpu_count() - 1)
    interpolation_dir = sys.argv[1]
    output = sys.argv[3]

    target_dataframe = pd.read_csv(interpolation_dir, header=0)
    result = []
    for file in listdir(sys.argv[2]):
        correlation_target_dataframe = pd.read_csv(path.join(sys.argv[2], file), header=0)
        file_result = do_correlate(target_dataframe, correlation_target_dataframe, thread_pool)
        result.append([file] + file_result)

    print("\n--- Correlated %d files in %.4f seconds ---" % (len(listdir(sys.argv[2])), time.time() - start_time))
    pd.DataFrame(result, columns=['file', 'mse_dip', 'mse_dir', 'mspe_dip', 'mspe_dir']).to_csv(output, index=False)


if __name__ == '__main__':
    main()
