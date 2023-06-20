# # import argparse
# # import easydict
# # import numpy as np
# # import pandas as pd
# # import tensorflow as tf
# # from tensorflow import keras
# # from tensorflow.keras.models import load_model
# # from matplotlib import pyplot as plt
# # from plotread import *
# # import shap
# # import pickle
# # # import pickle
# # from collections import Counter

# # import tensorflow._api.v2.compat.v1 as tf
# # from tensorflow.compat.v1.keras.backend import get_session
# # tf.compat.v1.disable_v2_behavior()
# # tf.compat.v1.disable_eager_execution()
# # from tensorflow.python.ops.numpy_ops import np_config
# # np_config.enable_numpy_behavior()

# # def main():
# #     ###########################################################################
# #     # Parser Definition
# #     ###########################################################################
# #     opt = easydict.EasyDict({
# #         "model": "GRU",
# #         "datapath": "Dataset1/",
# #         "savepath": "Experiment1/GRU/8/Flatten/Runx/",
# #         "extension": ".dat",
# #         "batch_size": 32,
# #         "plots_in": False,
# #         "plots_out": True
# #     })

# #     ###########################################################################
# #     # Variables Definition
# #     ###########################################################################
# #     nin = ['time', 'PLML2', 'PLMR', 'AVBL', 'AVBR']
# #     nout = ['time', 'DB1', 'LUAL', 'PVR', 'VB1']
# #     neurons = ['time','DB1','LUAL','PVR','VB1','PLML2','PLMR','AVBL','AVBR']

# #     ###########################################################################
# #     # Read data
# #     ###########################################################################
# #     files = getdata(opt.datapath, opt.extension)
# #     train, valid, test = splitdata(files)
# #     trainx, trainy = readdata(opt.datapath, train, neurons, nin, nout)
# #     validx, validy = readdata(opt.datapath, valid, neurons, nin, nout)
# #     testx, testy = readdata(opt.datapath, test, neurons, nin, nout)
# #     if opt.plots_in:
# #         plotdata(trainx, '/train_data', '/x', opt.model, opt.savepath)
# #         plotdata(trainy, '/train_data', '/y', opt.model, opt.savepath)
# #         plotdata(validx, '/valid_data', '/x', opt.model, opt.savepath)
# #         plotdata(validy, '/valid_data', '/y', opt.model, opt.savepath)
# #         plotdata(testx, '/test_data', '/x', opt.model, opt.savepath)
# #         plotdata(testy, '/test_data', '/y', opt.model, opt.savepath)
        
# #     ###########################################################################
# #     # Load Model and Evaluate
# #     ###########################################################################
# #     output_size = 4
# #     global model
# #     model = load_model(opt.savepath + 'model.h5')
# #     model.summary()

# #     print("Starting to explain...")
# #     X_train = np.array(trainx)
# #     X_test = np.array(testx)

# #     nin_names = nin[1:len(nin)]

# #     return X_train, X_test, nin_names


# # X_train, X_test, nin_names = main()

# # background = X_train[np.random.choice(X_train.shape[0], 1, replace=False)]

# # explainer = shap.DeepExplainer(model, background)

# # shap_values = explainer.shap_values(X_test[:1,:,:])

# # print(shap_values)

# # with open("shap_values.pkl", 'wb') as file:
# #     pickle.dump(shap_values, file)

















# import argparse
# import easydict
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import load_model
# from matplotlib import pyplot as plt
# from plotread import *
# import shap
# import pickle
# from collections import Counter
# from multiprocessing import Pool

# import tensorflow._api.v2.compat.v1 as tf
# from tensorflow.compat.v1.keras.backend import get_session
# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()

# def main():
#     ###########################################################################
#     # Parser Definition
#     ###########################################################################
#     opt = easydict.EasyDict({
#         "model": "GRU",
#         "datapath": "Dataset1/",
#         "savepath": "Experiment1/GRU/8/Flatten/Runx/",
#         "extension": ".dat",
#         "batch_size": 32,
#         "plots_in": False,
#         "plots_out": True
#     })

#     ###########################################################################
#     # Variables Definition
#     ###########################################################################
#     nin = ['time', 'PLML2', 'PLMR', 'AVBL', 'AVBR']
#     nout = ['time', 'DB1', 'LUAL', 'PVR', 'VB1']
#     neurons = ['time','DB1','LUAL','PVR','VB1','PLML2','PLMR','AVBL','AVBR']

#     ###########################################################################
#     # Read data
#     ###########################################################################
#     files = getdata(opt.datapath, opt.extension)
#     train, valid, test = splitdata(files)
#     trainx, trainy = readdata(opt.datapath, train, neurons, nin, nout)
#     validx, validy = readdata(opt.datapath, valid, neurons, nin, nout)
#     testx, testy = readdata(opt.datapath, test, neurons, nin, nout)
#     if opt.plots_in:
#         plotdata(trainx, '/train_data', '/x', opt.model, opt.savepath)
#         plotdata(trainy, '/train_data', '/y', opt.model, opt.savepath)
#         plotdata(validx, '/valid_data', '/x', opt.model, opt.savepath)
#         plotdata(validy, '/valid_data', '/y', opt.model, opt.savepath)
#         plotdata(testx, '/test_data', '/x', opt.model, opt.savepath)
#         plotdata(testy, '/test_data', '/y', opt.model, opt.savepath)
        
#     ###########################################################################
#     # Load Model and Evaluate
#     ###########################################################################
#     output_size = 4
#     global model
#     model = load_model(opt.savepath + 'model.h5')
#     model.summary()

#     print("Starting to explain...")
#     X_train = np.array(trainx)
#     X_test = np.array(testx)

#     nin_names = nin[1:len(nin)]

#     return X_train, X_test, nin_names


# # def calculate_shap_values(X_test_sample):
# #     explainer = shap.DeepExplainer(model, background)
# #     return explainer.shap_values(X_test_sample)


# # if __name__ == '__main__':
# #     X_train, X_test, nin_names = main()

# #     # Create a background sample for SHAP explanation
# #     background = X_train[np.random.choice(X_train.shape[0], 1, replace=False)]

# #     # Parallelize the computation of SHAP values
# #     with Pool() as pool:
# #         shap_values = pool.map(calculate_shap_values, X_test[:1, :, :])

# #     print(shap_values)


# def calculate_shap_values(background, X_test_sample):
#     explainer = shap.DeepExplainer(model, background)
#     return explainer.shap_values(X_test_sample)


# if __name__ == '__main__':
#     X_train, X_test, nin_names = main()

#     # Create a background sample for SHAP explanation
#     background = X_train[np.random.choice(X_train.shape[0], 1, replace=False)]

#     # Parallelize the computation of SHAP values
#     with Pool() as pool:
#         shap_values = pool.starmap(calculate_shap_values, [(background, x_sample) for x_sample in X_test[:1, :, :]])

#     print(shap_values)










import argparse
import easydict
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
from plotread import *
import shap
import pickle
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import tensorflow._api.v2.compat.v1 as tf
from tensorflow.compat.v1.keras.backend import get_session
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


def main():
    ###########################################################################
    # Parser Definition
    ###########################################################################
    opt = easydict.EasyDict({
        "model": "GRU",
        "datapath": "Dataset1/",
        "savepath": "Experiment1/GRU/8/Flatten/Runx/",
        "extension": ".dat",
        "batch_size": 32,
        "plots_in": False,
        "plots_out": True
    })

    ###########################################################################
    # Variables Definition
    ###########################################################################
    nin = ['time', 'PLML2', 'PLMR', 'AVBL', 'AVBR']
    nout = ['time', 'DB1', 'LUAL', 'PVR', 'VB1']
    neurons = ['time','DB1','LUAL','PVR','VB1','PLML2','PLMR','AVBL','AVBR']

    ###########################################################################
    # Read data
    ###########################################################################
    files = getdata(opt.datapath, opt.extension)
    train, valid, test = splitdata(files)
    trainx, trainy = readdata(opt.datapath, train, neurons, nin, nout)
    validx, validy = readdata(opt.datapath, valid, neurons, nin, nout)
    testx, testy = readdata(opt.datapath, test, neurons, nin, nout)
    if opt.plots_in:
        plotdata(trainx, '/train_data', '/x', opt.model, opt.savepath)
        plotdata(trainy, '/train_data', '/y', opt.model, opt.savepath)
        plotdata(validx, '/valid_data', '/x', opt.model, opt.savepath)
        plotdata(validy, '/valid_data', '/y', opt.model, opt.savepath)
        plotdata(testx, '/test_data', '/x', opt.model, opt.savepath)
        plotdata(testy, '/test_data', '/y', opt.model, opt.savepath)
        
    ###########################################################################
    # Load Model and Evaluate
    ###########################################################################
    output_size = 4
    global model
    model = load_model(opt.savepath + 'model.h5')
    model.summary()

    print("Starting to explain...")
    X_train = np.array(trainx)
    X_test = np.array(testx)

    nin_names = nin[1:len(nin)]

    return X_train, X_test, nin_names


def run_shap_explanation(X_test, model, background):
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_test[:1, :, :])
    return shap_values


if __name__ == "__main__":

    X_train, X_test, nin_names = main()

    print(X_train)
    print(X_test)
    print(nin_names)
    print(X_train.shape)
    print(X_test.shape)

    background = X_train[np.random.choice(X_train.shape[0], 1, replace=False)]

    print(background)
    print(background.shape)

    # Create a ThreadPoolExecutor with the desired number of threads
    num_threads = 4  # Adjust this value based on your system's capabilities
    executor = ThreadPoolExecutor(max_workers=num_threads)

    # Submit the tasks to the executor and collect the futures
    futures = []
    for _ in range(X_test.shape[0]):
        print("Beginning for loop")
        print(_)
        future = executor.submit(run_shap_explanation, X_test[_:_+1, :, :], model, background)
        print("In the middle for loop")
        print(_)
        futures.append(future)
        print("Exiting for loop")
        print(_)

    # Retrieve the results from the futures
    shap_values = [future.result() for future in futures]

    print(shap_values)

    with open("shap_values.pkl", 'wb') as file:
        pickle.dump(shap_values, file)