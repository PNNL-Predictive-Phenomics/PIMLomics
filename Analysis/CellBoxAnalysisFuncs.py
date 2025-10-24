

import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import ast 
import warnings
from matplotlib.lines import Line2D

def PlotFigure2ForRun(index,resultDirectory,trialData,root,noilist):
    
    loss = pd.read_csv('results/'+str(resultDirectory)+'/seed_'+str(index)+'/record_eval.csv',usecols=range(8),header=None)

    if isinstance(loss.loc[0,0],str):
        # print('is string')
        loss = pd.read_csv('results/'+str(resultDirectory)+'/seed_'+str(index)+'/record_eval.csv',usecols=range(8))
        
    
    noi_index = np.genfromtxt(root+'/CorrectedData/'+str(trialData)+'/node_index.csv', 
                              dtype = str)[noilist]
    # Combined plot
    f, axes = plt.subplots(figsize = [12, 11])
    plt.title('index')
    # Panel A
    ax = plt.subplot2grid((5, 2), (0, 0), rowspan=2)
    nma = 10
    
    # idx = np.where([x!='None' for x in loss['train_mse']])[0]
    idx = np.where([x!='None' for x in loss.iloc[:,4]])[0]

    def lsplit(lists):
        result = []
        temp_list = []
        for i in lists:
            if i == '0':
                #temp_list.append(i)
                result.append(temp_list)
                temp_list = []
            elif i != 'None':
                temp_list.append(i)
        result.append(temp_list)
        return result

    split_iter = lsplit(loss.iloc[:,1])

    sizes1 = 0
    sizes2 = len(split_iter[1])+sizes1


    # losssnip_t = loss['train_mse'][idx]
    # losssnip_v = loss['valid_mse'][idx]
    losssnip_t = loss.iloc[:,4][idx]
    losssnip_v = loss.iloc[:,5][idx]

    plt.plot(np.arange(sizes2-sizes1-nma+1), 
             moving_average(np.array([float(x) for x in losssnip_t[sizes1:sizes2]]),n=nma), 
             alpha = 0.8, color="black",linestyle="dashed")
    plt.plot(np.arange(sizes2-sizes1-nma+1), 
             moving_average(np.array([float(x) for x in losssnip_v[sizes1:sizes2]]),n=nma), 
             alpha = 0.8, color="black")

    plt.xlabel('Training iterations')
    plt.ylabel('Mean Squared Error')
    plt.title('Seed '+str(index))
    custom_lines = [Line2D([0], [0], color='black',linestyle="dashed"), 
                    Line2D([0], [0], color='black')]
    legend = plt.legend(custom_lines, ['Training', 'Validation'], loc='upper right',
                        frameon=False)
    ax.add_artist(legend)
    plt.text(-0.13,1.02,'A', weight='bold',transform=ax.transAxes)

    # Panel B
    ax = plt.subplot2grid((5, 2), (0, 1), rowspan=2)
    
    
    nlength=106
    
    
    trace, real, substageNum = b(root,resultDirectory,trialData,noilist,noi_index,index = index, condition = 0, nT = 500, nlength = nlength)

    plt.xlabel('ODE simulation steps')
    plt.ylabel('Cell Response', labelpad=-4)


    trace_end=trace[-1,:]
    print(trace.shape)


    # Panel C
    y_hat = pd.read_csv(glob.glob(str(substageNum)+'_best.y_hat*csv')[0], index_col = 0)#.values

    y = pd.read_csv(root+"/CorrectedData/"+str(trialData)+"/expr_matr.csv", header = None)

    ax = plt.subplot2grid((14, 2), (7, 0), rowspan=8)
    x_all = y.iloc[0,0:98].values.flatten()
    # y_all = y_hat.iloc[0:9,1:99].values.flatten()
    y_all = trace_end[1:99]

    print('y_all')
    print(y_all.shape)

    #for t4, 0 to 40 and then 41 to 98
    # x_prot = y.iloc[0:9,0:40]
    # y_prot = y_hat.iloc[0:9,1:41]
    # x_mod = y.iloc[0:9,41:98]
    # y_mod = y_hat.iloc[0:9,42:99]

    x_prot = y.iloc[0,0:40]
    y_prot = y_all[0:40]
    x_mod = y.iloc[0,41:98]
    y_mod = y_all[41:98]

    print('y_mod')
    print(y_mod.shape)
    print('x_mod')
    print(x_mod.shape)
    

    plt.scatter(x_prot, y_prot, s = 15, alpha = 0.7, color="#74A6D1",zorder=3)
    plt.scatter(x_mod, y_mod, s = 15, alpha = 0.7, color="#3D6CA3",zorder=4)
    plt.legend(["Molecular (protein) nodes","Modulon nodes"], loc="upper right", frameon=False,
              handletextpad=0.1, fontsize=7.5)
    #plt.plot([-10, 10], [-10, 10], c = 'white', alpha = 0, ls = '--')
    sns.regplot(x=x_all, y=y_all, scatter_kws={'s': 15, 'alpha': 0},line_kws={'color': '#1B406C', 'alpha': 1})

    lower = np.min([x_all, y_all])
    upper = np.max([x_all, y_all])

    plt.xlim([lower*1.2, upper*1.2])
    plt.ylim([lower*1.2, upper*1.2])
    r = np.corrcoef(x_all, y_all)[0][1]
    plt.text(x = 12, y= 10, s='Pearson\'s ρ: \n ρ = %1.3f'%r,size = 8.5)
    plt.xlabel('Experimental response')
    plt.ylabel('Predicted response')
    #plt.title("Correlation between predictions and \n experiments across all conditions", weight='bold', size=15)

    plt.text(-0.13,1.06,'C', weight='bold',transform=ax.transAxes)
    plt.show()
    
    # Panel D (across different conditions)
    ax = plt.subplot2grid((14, 2), (7, 1), rowspan=8)
    x_all = y.values
    y_all = y_hat.values
    rs = [np.corrcoef(x_all[i], y_all[i])[0][1] for i in range(y_hat.shape[0])]
    plt.hist(rs, bins = 22, color = 'grey', alpha = 0.6, rwidth=0.93)
    plt.axvline(x = r, linewidth=2, label = 'Median', color="#1B406C")
    plt.xlabel('Experiment-prediction correlation')
    plt.ylabel('Number of conditions')
    #plt.text(0.62,33,"correlation for \nall conditions", color="#1B406C",size = 15)
#     plt.text(-0.13,1.06,'D',transform=ax.transAxes)

    return r

# Panel B
nlength = 102
def _simu(t_mu, W, alpha, eps, x_0 = np.zeros([nlength]), dT=0.1, n_T=100):
    def _dXdt(x):
        dXdt = eps[:, 0] * np.tanh(np.matmul(W, x) + t_mu) - alpha[:, 0] * x
        return dXdt
    
    x = x_0
    trace = x_0
    for i in range(n_T):
        """ Integrate with Heun's Method """
        dXdt_current = _dXdt(x)
        dXdt_next = _dXdt(x + dT * dXdt_current)
        x = x + dT * 0.5 * (dXdt_current + dXdt_next)
        trace = np.append(trace,x)
    print(trace.shape)
    trace = np.reshape(trace, [n_T+1, nlength])
    return trace
def b(root,resultDirectory,trialData,noilist,noi_index,index = '000', condition = 0, nT = 400, nlength=106):

    #os.chdir(root+'/b11_'+index)
    os.chdir(root+'/results/'+str(resultDirectory)+'/seed_'+str(index)+'/')

#     alpha = pd.read_csv(glob.glob('*%d*json.4*alpha*csv'%nT)[0], index_col = 0).values
#     eps = pd.read_csv(glob.glob('*%d*json.4*eps*csv'%nT)[0], index_col = 0).values
#     w = pd.read_csv(glob.glob('*%d*json.4*W*csv'%nT)[0], index_col = 0).values
    substageNum=6
    isNotFound=True
    while(substageNum>0 and isNotFound):
        try:

            alpha = pd.read_csv(glob.glob(str(substageNum)+'_best.alpha*csv')[0], index_col = 0).values
            eps = pd.read_csv(glob.glob(str(substageNum)+'_best.eps*csv')[0], index_col = 0).values
            w = pd.read_csv(glob.glob(str(substageNum)+'_best.W*csv')[0], index_col = 0).values
            isNotFound = False
        except:
            substageNum -=1
    print('substageNum: '+str(substageNum))        
    pert = np.genfromtxt('../../../CorrectedData/'+str(trialData)+'/pert_matr.csv', dtype = np.float32, delimiter = ',')
    expr = np.genfromtxt('../../../CorrectedData/'+str(trialData)+'/expr_matr.csv', dtype = np.float32, delimiter = ',')
    pos = np.genfromtxt('random_pos.csv')

    noi = noilist

    trace = _simu(pert[condition], w, alpha, eps, x_0 = np.zeros([nlength]), dT=0.1, n_T = int(nT))
    trace_subset = trace[:,noi].transpose()
    xs = np.linspace(0, nT/10, int(nT)+1)
    real = expr[condition, noi]

    for t, trace_i in enumerate(trace_subset):
        plt.axhline(y = real[t], xmax = 0.98, ls="dashed",  alpha = 0.8,
                    color = sns.color_palette("deep")[t])
                    
        plt.plot(xs, trace_i, color = sns.color_palette("deep")[t], 
                 label = noi_index[t], alpha = 0.8)


    #plt.axvline(x = nT/10, color="black", ls="dashed", alpha = 0.8, linewidth=2)
    return trace, real,substageNum

def b2(root,resultDirectory,trialData,noilist,noi_index,index = '000', condition = 0, nT = 400, nlength=106):

    #os.chdir(root+'/b11_'+index)
    os.chdir(root+str(resultDirectory))

#     alpha = pd.read_csv(glob.glob('*%d*json.4*alpha*csv'%nT)[0], index_col = 0).values
#     eps = pd.read_csv(glob.glob('*%d*json.4*eps*csv'%nT)[0], index_col = 0).values
#     w = pd.read_csv(glob.glob('*%d*json.4*W*csv'%nT)[0], index_col = 0).values
    substageNum=6
    isNotFound=True

    alpha = pd.read_csv(glob.glob('best.alpha*csv')[0], index_col = 0).values
    eps = pd.read_csv(glob.glob('best.eps*csv')[0], index_col = 0).values
    w = pd.read_csv(glob.glob('best.W*csv')[0], index_col = 0).values
 
     
    pert = np.genfromtxt('../../../CellBox-master/cyano_rna_tests/Cellbox_t4/pert_matr_t4.csv', dtype = np.float32, delimiter = ',')
    expr = np.genfromtxt('../../../CellBox-master/cyano_rna_tests/Cellbox_t4/expr_matr_t4.csv', dtype = np.float32, delimiter = ',')
    # pos = np.genfromtxt('random_pos.csv')

    noi = noilist

    trace = _simu(pert[condition], w, alpha, eps, x_0 = np.zeros([nlength]), dT=0.1, n_T = int(nT))
    trace_subset = trace[:,noi].transpose()
    xs = np.linspace(0, nT/10, int(nT)+1)
    real = expr[condition, noi]

    for t, trace_i in enumerate(trace_subset):
        plt.axhline(y = real[t], xmax = 0.98, ls="dashed",  alpha = 0.8,
                    color = sns.color_palette("deep")[t])
                    
        plt.plot(xs, trace_i, color = sns.color_palette("deep")[t], 
                 label = noi_index[t], alpha = 0.8)


    #plt.axvline(x = nT/10, color="black", ls="dashed", alpha = 0.8, linewidth=2)
    return trace, real,substageNum


def moving_average(sequence, n=5) :
    ret = np.cumsum(sequence, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
