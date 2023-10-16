import argparse
import numpy as np
from keras.models import load_model
from plotread import *

def evaluateresults(model, opt, trainx, trainy, validx, validy, testx, testy, output_size):

    #Evaluate on training, validation and test data
    train_l = model.evaluate(np.array(trainx), np.array(trainy), batch_size=opt.batch_size,
        verbose=1, workers=12, use_multiprocessing=True)
    valid_l = model.evaluate(np.array(validx), np.array(validy), batch_size=opt.batch_size,
        verbose=1, workers=12, use_multiprocessing=True)
    test_l = model.evaluate(np.array(testx), np.array(testy), batch_size=opt.batch_size,
        verbose=1, workers=12, use_multiprocessing=True)
    
    trainPredict = model.predict(np.array(trainx), batch_size=opt.batch_size, verbose=1,
        workers=12, use_multiprocessing=True)
    plotresults(opt, trainy, trainPredict, opt.model, "/train", opt.savepath, output_size)
    validPredict = model.predict(np.array(validx), batch_size=opt.batch_size, verbose=1,
        workers=12, use_multiprocessing=True)
    plotresults(opt, validy, validPredict, opt.model, "/valid", opt.savepath, output_size)
    testPredict = model.predict(np.array(testx), batch_size=opt.batch_size, verbose=1,
        workers=12, use_multiprocessing=True)
    plotresults(opt, testy, testPredict, opt.model, "/test", opt.savepath, output_size)

    return train_l, valid_l, test_l

def plotresults(opt, real, predicted, folder, string, path, out_size):
#     cols = ['DB1_Predicted', 'LUAL_Predicted', 'PVR_Predicted', 'VB1_Predicted']
#     cols = ['LUAL_Predicted', 'PVR_Predicted', 'VB1_Predicted']
#     cols = ['PVR_Predicted', 'VB1_Predicted']
    cols = ['VB1_Predicted']
    
#     real = np.reshape(real, (len(real), 1000, 4), order='F')
#     real = real.tolist()
    
    i=0
    for frame in predicted:
        frame = np.reshape(frame, (frame.shape[0], 1))
        pos = real[i].shape[1]
        for j in range(out_size):
            real[i].insert(
                loc = pos,
                column = cols[j],
                value = frame[:,j]
            )
            pos = pos + 1
        
        real[i].iloc[:, 0].plot(kind='line')
        real[i].iloc[:, 1].plot(kind='line', linestyle='--')

#         real[i].iloc[:, 0].plot(kind='line')
#         real[i].iloc[:, 1].plot(kind='line')
#         real[i].iloc[:, 2].plot(kind='line')
#         real[i].iloc[:, 3].plot(kind='line')
#         real[i].iloc[:, 4].plot(kind='line', linestyle='--')
#         real[i].iloc[:, 5].plot(kind='line', linestyle='--')
#         real[i].iloc[:, 6].plot(kind='line', linestyle='--')
#         real[i].iloc[:, 7].plot(kind='line', linestyle='--')
        
        Path(path + "results_files").mkdir(parents=True, exist_ok=True)
        real[i].to_csv(
            path + "results_files" + string + str(i) + '_data.dat', sep=' ', header=False)
        if opt.plots_out:
            plt.legend(loc='upper right')
            Path(path + "results_plots").mkdir(parents=True, exist_ok=True)
            plt.savefig(path + "results_plots" + string + str(i) + '_response.pdf')
            plt.close()

        i = i + 1



# def plotresults(opt, real, predicted, folder, string, path, out_size):
#     cols = ['DB1_Predicted', 'LUAL_Predicted', 'PVR_Predicted', 'VB1_Predicted']
        
#     # output_size=4
#     i = 0
#     for frame in predicted:
        
# #         print(type(real))
        
#         real_nparray = np.array(real)
        
# #         print(real)
# #         print(real_nparray)
# #         print(real_nparray.shape)
#         print(real_nparray.ndim)
        
#         if (real_nparray.ndim==3):
#             pos = real[i].shape[1]
#             for j in range(out_size):
#                 real[i].insert(
#                     loc = pos,
#                     column = cols[j],
#                     value = frame[:,j]
#                 )
#                 pos += 1
                
# #         print(real[i])
        
#         df = pd.DataFrame(real[i])
#         df.plot(kind='line')
        
# #         print(df)
        
#         Path(path + "results_files").mkdir(parents=True, exist_ok=True)
#         df.to_csv(path + "results_files" + string + str(i) + '_data.dat', sep=' ', header=False)
        
#         if opt.plots_out:
#             plt.legend(loc='upper right')
#             Path(path + "results_plots").mkdir(parents=True, exist_ok=True)
#             plt.savefig(path + "results_plots" + string + str(i) + '_response.pdf')
#             plt.close()

#         i += 1



# #     opt, real,   predicted,    folder,    string,   path,         out_size
# #     opt, trainy, trainPredict, opt.model, "/train", opt.savepath, output_size
 
# #     print(type(predicted))
#     predicted = np.reshape(predicted, (predicted.shape[0], 1000, 4), order='F')
# #     print(predicted.shape)
# #     predicted = predicted.tolist()
# #     print(type(predicted))
    
#     # output_size=1    
#     i = 0
#     for frame in predicted:
        
#         print(type(frame))
#         print(frame.shape)     # (1000, 4)
# #         print(len(frame))    # 1000
# #         print(len(frame[0])) # 4
# #         frame = np.reshape(frame, (frame.shape[0], 1))

# #         print(type(real))    # list
# #         print(len(real))     # 20
# #         print(len(real[0]))  # 4000
#         real = np.reshape(real, (len(real), 1000, 4), order='F')
# #         print(real.shape)

#         pos = real[i].shape[1]
# #         pos = len(real[i][0])
#         print(pos)
# #         print(len(real[i]))
# #         print(type(real))
# #         print(type(real[i]))
# #         real = real.tolist()
# #         print(type(real))
# #         real = pd.DataFrame(real)
# #         print(type(real))
#         print(real[i])     # (1000, 4)
#         print(frame[:, 0]) # (1000,)
#         for j in range(out_size):
#             print(type(real[i]))
#             print(real[i])
#             real[i] = pd.DataFrame(real[i], loc = pos,
#                 column = cols[j],
#                 value = frame[:,j])
#             print(type(real[i]))
#             print(real[i])
#             real[i].insert(
#                 loc = pos,
#                 column = cols[j],
#                 value = frame[:,j]
#             )
#             pos = pos + 1
#         real[i].plot(kind='line')
#         Path(path + "results_files").mkdir(parents=True, exist_ok=True)
#         real[i].to_csv(
#             path + "results_files" + string + str(i) + '_data.dat', sep=' ', header=False)
#         if opt.plots_out:
#             plt.legend(loc='upper right')
#             Path(path + "results_plots").mkdir(parents=True, exist_ok=True)
#             plt.savefig(path + "results_plots" + string + str(i) + '_response.pdf')
#             plt.close()

#         i = i + 1



















































# def plotresults(opt, real, predicted, folder, string, path, out_size):
#     cols = ['DB1_Predicted', 'LUAL_Predicted', 'PVR_Predicted', 'VB1_Predicted']
    
#     i = 0
#     for frame in predicted:
#         real_nparray = np.array(real)
        
# #         print(real)
# #         print(real_nparray)
# #         print(real_nparray.ndim)
        
#         if (real_nparray.ndim==3):
#             pos = real[i].shape[1]
#             for j in range(out_size):
#                 real[i].insert(
#                     loc = pos,
#                     column = cols[j],
#                     value = frame[:,j]
#                 )
#                 pos += 1
                
# #         print(real[i])
        
#         df = pd.DataFrame(real[i])
#         df.plot(kind='line')
        
# #         print(df)
        
#         Path(path + "results_files").mkdir(parents=True, exist_ok=True)
#         df.to_csv(path + "results_files" + string + str(i) + '_data.dat', sep=' ', header=False)
        
#         if opt.plots_out:
#             plt.legend(loc='upper right')
#             Path(path + "results_plots").mkdir(parents=True, exist_ok=True)
#             plt.savefig(path + "results_plots" + string + str(i) + '_response.pdf')
#             plt.close()

#         i += 1

# def plotresults(opt, real, predicted, folder, string, path, out_size):
#     cols = ['DB1_Predicted', 'LUAL_Predicted', 'PVR_Predicted', 'VB1_Predicted']
    
#     i=0
#     for frame in predicted:
        
#         print(frame)
#         print(frame.shape)
#         print("real[i].shape:")
#         print(real[i].shape)
#         print("real.shape:")
#         print(real.shape)
#         print(type(real))
        
#         real = np.reshape(real, (real.shape[0], real.shape[1], 1), order='F')
#         frame = np.reshape(frame, (frame.shape[0], 1), order='F')
        
#         print("real[i].shape:")
#         print(real[i].shape)
#         print("real.shape:")
#         print(real.shape)
#         print(type(real))
        
# #         pos = real[i].shape[1]
# #         print(pos)
# #         pos = 0
        
# #         real_nparray = np.array(real)
# #         print("real_nparray:")
# #         print(real_nparray)
# #         print("real_nparray.ndim:")
# #         print(real_nparray.ndim)
# #         print("real_nparray.shape:")
# #         print(real_nparray.shape)
        
#         for j in range(out_size):
#             pos = real[i].shape[1]
#             print("real[i].shape:")
#             print(real[i].shape)
#             print("real.shape:")
#             print(real.shape)
#             print(type(real))
#             real = np.reshape(real, (real.shape[1], real.shape[0]), order='F')
#             print("real[i].shape:")
#             print(real[i].shape)
#             print("real.shape:")
#             print(real.shape)
#             print(type(real))
#             real = pd.DataFrame(real)
#             print(real)
#             print("real[i].shape:")
#             print(real[i].shape)
#             print("real.shape:")
#             print(real.shape)
#             print(type(real))
#             real[i].insert(
#                 loc = pos,
#                 column = cols[j],
#                 value = frame[:,j]
#             )
#             pos = pos + 1
#         real[i].plot(kind='line')
#         Path(path + "results_files").mkdir(parents=True, exist_ok=True)
#         real[i].to_csv(
#             path + "results_files" + string + str(i) + '_data.dat', sep=' ', header=False)
#         if opt.plots_out:
#             plt.legend(loc='upper right')
#             Path(path + "results_plots").mkdir(parents=True, exist_ok=True)
#             plt.savefig(path + "results_plots" + string + str(i) + '_response.pdf')
#             plt.close()

#         i = i + 1


def main():
    ###########################################################################
    # Parser Definition
    ###########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-model', choices=['LSTM', 'GRU', 'RNN'], default='RNN')
    parser.add_argument('-datapath', type=str, default="Dataset1/")
    parser.add_argument('-savepath', type=str, default="Experiment1/RNN/16/Run1/")
    parser.add_argument('-extension', type=str, default=".dat")
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-plots_in', type=bool, default=False)
    parser.add_argument('-plots_out', type=bool, default=True)
    opt = parser.parse_args()

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
    
#     print("I am going to print in MAIN")
#     print(type(trainx), type(trainy))
#     print(np.array(trainx).shape, np.array(trainy).shape)

#     trainy = np.reshape(trainy, (20, 4000), order='F')
#     validy = np.reshape(validy, (10, 4000), order='F')
#     testy = np.reshape(testy, (10, 4000), order='F')
    
#     print("I am going to print in MAIN 2")
#     print(type(trainx), type(trainy))
#     print(np.array(trainx).shape, np.array(trainy).shape)
    
#     trainy = trainy.tolist()
#     validy = validy.tolist()
#     testy = testy.tolist()
    
#     print(type(trainx), type(trainy))
#     print(np.array(trainx).shape, np.array(trainy).shape)

    
    
    
    
    
#     print(type(trainy))
    
#     trainy = np.reshape(trainy, (20, 1000), order='F')
#     validy = np.reshape(validy, (10, 1000), order='F')
#     testy = np.reshape(testy, (10, 1000), order='F')
    
#     print(type(trainy))

#     print(trainy)
    
    
#     print(np.array(trainy).shape)

#     print(type(trainy))

#     trainy = np.reshape(trainy, (20, 4000), order='F')
#     validy = np.reshape(validy, (10, 4000), order='F')
#     testy = np.reshape(testy, (10, 4000), order='F')

    if opt.plots_in:
        plotdata(trainx, '/train_data', '/x', opt.model, opt.savepath)
        plotdata(trainy, '/train_data', '/y', opt.model, opt.savepath)
        plotdata(validx, '/valid_data', '/x', opt.model, opt.savepath)
        plotdata(validy, '/valid_data', '/y', opt.model, opt.savepath)
        plotdata(testx, '/test_data', '/x', opt.model, opt.savepath)
        plotdata(testy, '/test_data', '/y', opt.model, opt.savepath)
        
#     trainy = np.reshape(trainy, (20, 4000), order='F')
#     validy = np.reshape(validy, (10, 4000), order='F')
#     testy = np.reshape(testy, (10, 4000), order='F')
    
    ###########################################################################
    # Load Model and Evaluate
    ###########################################################################
#     output_size = 4
    output_size = 1
    model = load_model(opt.savepath + 'model.h5')
    model.summary()
    
#     print("I am going to print in MAIN 3")
#     print(type(trainx), type(trainy))
#     print(np.array(trainx).shape, np.array(trainy).shape)
    
    lt, lv, ltt = evaluateresults(
        model, opt, trainx, trainy, validx, validy, testx, testy, output_size)
    print("Training Loss: " + str(lt))
    print("Validation Loss: " + str(lv))
    print("Test Loss: " + str(ltt))

if __name__ == "__main__":
    main()
