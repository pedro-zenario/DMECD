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
import pickle
from collections import Counter

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
        "savepath": "Experiment1/GRU/4/Flatten/VB1/adam_001/",
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
#     output_size = 4
    output_size = 1
    global model
    model = load_model(opt.savepath + 'model.h5')
    model.summary()

    print("Starting to explain...")
    X_train = np.array(trainx)
    X_test = np.array(testx)
        
    y_train = np.array(trainy)
    y_test = np.array(testy)
    
    nin_names = nin[1:len(nin)]
    nout_names = nout[1:len(nout)]

    return X_train, X_test, nin_names, nout_names, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, nin_names, nout_names, y_train, y_test = main()
    
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#     background = X_train[np.random.choice(X_train.shape[0], 1, replace=False)]
    
#     print(background)
#     print(background.shape)

#     explainer_background = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    explainer_X_train = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), X_train)
    
#     print(explainer_background)
    print(explainer_X_train)
    

#     shap_values_1_background = explainer_background.shap_values(X_test[:1,:,:])
# #     shap_values_10_X_train = explainer_X_train.shap_values(X_test)
# #     with open("shap_values_1_background.pkl", 'wb') as file:
#     with open("/home/zenario/DMECD/DMECD/Experiment1/GRU/4/Flatten/Run0/pickles/shap_values_1_background.pkl", 'wb') as file:
#         pickle.dump(shap_values_1_background, file)
# #     print(shap_values_1_background)
#     print(np.array(shap_values_1_background).shape)
        
#     shap_values_1_X_train = explainer_X_train.shap_values(X_test[:1,:,:])
# #     with open("shap_values_1_X_train.pkl", 'wb') as file:
#     with open("/home/zenario/DMECD/DMECD/Experiment1/GRU/32/Flatten/Run3/pickles/shap_values_1_X_train.pkl", 'wb') as file:
#         pickle.dump(shap_values_1_X_train, file)
#     print(shap_values_1_X_train)
#     print(np.array(shap_values_1_X_train).shape)
    
#     shap_values_10_background = explainer_background.shap_values(X_test[:,:,:])
# #     with open("shap_values_10_background.pkl", 'wb') as file:
#     with open("/home/zenario/DMECD/DMECD/Experiment1/GRU/32/Flatten/Run3/pickles/shap_values_10_background.pkl", 'wb') as file:
#         pickle.dump(shap_values_10_background, file)    
#     print(shap_values_10_background)
#     print(np.array(shap_values_10_background).shape)
    
    shap_values_10_X_train = explainer_X_train.shap_values(X_test)
#     with open("shap_values_10_X_train.pkl", 'wb') as file:
    with open("/home/zenario/DMECD/DMECD/Experiment1/GRU/4/Flatten/VB1/adam_001/pickles/shap_values_10_X_train.pkl", 'wb') as file:
        pickle.dump(shap_values_10_X_train, file)
#     print(shap_values_10_X_train)
    print(np.array(shap_values_10_X_train).shape)







#     gpu_device = tf.config.list_physical_devices('GPU')[0]
#     tf.config.experimental.set_memory_growth(gpu_device, True)
    
# #     with tf.device('/cpu:0'):
# #         print("Inside cpu...")
# # #         X_train, X_test, nin_names, nout_names, y_train, y_test = main()
        
# # #         print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# # #         background = X_train[np.random.choice(X_train.shape[0], 1, replace=False)]
        
# # #         print(background)
# # #         print(background.shape)

# # #         explainer_background = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
# # # #         explainer_X_train = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), X_train)

#     # with tf.device('/GPU:0'):
#     with tf.device('/device:GPU:0'):
#         print("Inside gpu...")

#         # get the start time
#         st_1 = time.time()
        
#         X_train, X_test, nin_names, nout_names, y_train, y_test = main()
#         print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        
#         # get the end time
#         et_1 = time.time()
        
#         # get the execution time
#         elapsed_time_1 = et_1 - st_1
#         print('Execution time 1:', elapsed_time_1, 'seconds')



#         # get the start time
#         st_2 = time.time()

#         background = X_train[np.random.choice(X_train.shape[0], 1, replace=False)]
#         print(background)
#         print(background.shape)

#         # get the end time
#         et_2 = time.time()
        
#         # get the execution time
#         elapsed_time_2 = et_2 - st_2
#         print('Execution time 2:', elapsed_time_2, 'seconds')



#         # get the start time
#         st_3 = time.time()

#         explainer_background = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
# #         explainer_X_train = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), X_train)

#         # get the end time
#         et_3 = time.time()
        
#         # get the execution time
#         elapsed_time_3 = et_3 - st_3
#         print('Execution time 3:', elapsed_time_3, 'seconds')




#         # get the end time
#         st_4 = time.time()

#         print("Going to calculate shap values now...")
#         shap_values_1_background = explainer_background.shap_values(X_test[:1,:,:])

#         # get the end time
#         et_4 = time.time()
        
#         # get the execution time
#         elapsed_time_4 = et_4 - st_4
#         print('Execution time 4:', elapsed_time_4, 'seconds')



#         # get the end time
#         st_5 = time.time()

#     # with open("shap_values_1_background.pkl", 'wb') as file:
#         # with open("/home/zenario/DMECD/DMECD/Experiment1/GRU/4/Flatten/VB1/adam_001/pickles/shap_values_1_background_gpu.pkl", 'wb') as file:
#         with open("/home/guests2/pze/Experiment1/GRU/4/Flatten/DB1/adam/pickles/shap_values_1_background_gpu.pkl", 'wb') as file:
#             pickle.dump(shap_values_1_background, file)
#         print(shap_values_1_background)
#         print(np.array(shap_values_1_background).shape)

#         # get the end time
#         et_5 = time.time()
        
#         # get the execution time
#         elapsed_time_5 = et_5 - st_5
#         print('Execution time 5:', elapsed_time_5, 'seconds')











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
# import pickle
# from collections import Counter
# import concurrent.futures

# # import tensorflow._api.v2.compat.v1 as tf
# # from tensorflow.compat.v1.keras.backend import get_session
# # tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()
# # from tensorflow.python.ops.numpy_ops import np_config
# # np_config.enable_numpy_behavior()


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


# def run_shap(X_test, model, background):
#     explainer = shap.DeepExplainer(model, background)
#     shap_values = explainer.shap_values(X_test[:, :, :])
#     return shap_values

# if __name__ == '__main__':
#     X_train, X_test, nin_names = main()

#     background = X_train[np.random.choice(X_train.shape[0], 1, replace=False)]

#     # Create a ThreadPoolExecutor with 128 threads
#     with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
#         # Create a list to store the future objects
#         futures = []

#         # Submit the tasks to the executor
#         for i in range(X_test.shape[0]):
#             future = executor.submit(run_shap, X_test[i:i+1, :, :], model, background)
#             futures.append(future)

#         # Retrieve the results as they become available
#         results = []
#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 result = future.result()
#                 results.append(result)
#             except Exception as e:
#                 print(f"Error occurred: {e}")

#     print(results)

#     with open("shap_values.pkl", 'wb') as file:
#         pickle.dump(results, file)













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


# def run_shap(X_test, model, background):
#     print("Inside run_shap")
#     explainer = shap.DeepExplainer(model, background)
#     print("Middle run_shap")
#     shap_values = explainer.shap_values(X_test[:, :, :])
#     print("Returning run_shap")
#     return shap_values


# if __name__ == '__main__':

#     X_train, X_test, nin_names = main()

#     background = X_train[np.random.choice(X_train.shape[0], 1, replace=False)]

#     # Define the number of processes to use
#     num_processes = 10  # Adjust this number based on your system's capacity

#     # Create a Pool of processes
#     pool = Pool(processes=num_processes)

#     # Split the X_test data into chunks for parallel processing
#     chunk_size = len(X_test) // num_processes
#     print(chunk_size)
#     X_test_chunks = [X_test[i:i+chunk_size, :, :] for i in range(0, len(X_test), chunk_size)]
#     print(X_test_chunks)
#     print(np.array(X_test_chunks).shape)


#     # Execute the main function in parallel
#     results = pool.map(run_shap(X_test, model, background), X_test_chunks)

#     # Combine the results
#     combined_results = np.concatenate(results, axis=0)

#     print(combined_results)

#     with open("shap_values.pkl", 'wb') as file:
#         pickle.dump(combined_results, file)
