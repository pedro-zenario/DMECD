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


def plotresults(opt, real, predicted, folder, string, path, out_size):
    cols = ['DB1_Predicted', 'LUAL_Predicted', 'PVR_Predicted', 'VB1_Predicted']
    
    i=0
    for frame in predicted:
#         print(type(frame))
#         print(frame.shape)
        frame = np.reshape(frame, (frame.shape[0], 1))
#         print(real)
#         print(real.shape)
        pos = real[i].shape[0]
        for j in range(out_size):
            real[i].insert(
                loc = pos,
                column = cols[j],
                value = frame[:,j]
            )
            pos = pos + 1
        real[i].plot(kind='line')
        Path(path + "results_files").mkdir(parents=True, exist_ok=True)
        real[i].to_csv(
            path + "results_files" + string + str(i) + '_data.dat', sep=' ', header=False)
        if opt.plots_out:
            plt.legend(loc='upper right')
            Path(path + "results_plots").mkdir(parents=True, exist_ok=True)
            plt.savefig(path + "results_plots" + string + str(i) + '_response.pdf')
            plt.close()

        i = i + 1


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
    
#     print(type(trainy))
    
#     trainy = np.reshape(trainy, (20, 1000), order='F')
#     validy = np.reshape(validy, (10, 1000), order='F')
#     testy = np.reshape(testy, (10, 1000), order='F')
    
#     print(type(trainy))

#     print(trainy)
    
    
#     print(np.array(trainy).shape)

    trainy = np.reshape(trainy, (20, 4000), order='F')
    validy = np.reshape(validy, (10, 4000), order='F')
    testy = np.reshape(testy, (10, 4000), order='F')

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
    model = load_model(opt.savepath + 'model.h5')
    model.summary()
    lt, lv, ltt = evaluateresults(
        model, opt, trainx, trainy, validx, validy, testx, testy, output_size)
    print("Training Loss: " + str(lt))
    print("Validation Loss: " + str(lv))
    print("Test Loss: " + str(ltt))

if __name__ == "__main__":
    main()
