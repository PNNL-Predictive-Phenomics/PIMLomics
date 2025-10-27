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
import json
import re
from scipy import stats
import statsmodels.stats.multitest as smt
import matplotlib.gridspec as gridspec
import networkx as nx 
from scipy.stats import linregress
import scipy as sc


def _simu(t_mu, W, alpha, eps, x_0, dT=0.1, n_T=100):
    
    nlength=len(x_0)
    
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

    trace = np.reshape(trace, [n_T+1, nlength])
    return trace

def PlotBestDataAllConditions(mainRoot,pert_path,expr_path,node_index_path,alpha_file,eps_file,W_file,y_hat_file,noilist, json_path="", nT = 400):
    

    alpha = pd.read_csv(glob.glob(alpha_file)[0], index_col = 0).values
    eps = pd.read_csv(glob.glob(eps_file)[0], index_col = 0).values
    w = pd.read_csv(glob.glob(W_file)[0], index_col = 0).values
    y_hat = pd.read_csv(glob.glob(y_hat_file)[0], index_col = 0)
    
    
    pert = np.genfromtxt(pert_path, dtype = np.float32, delimiter = ',')
    expr = np.genfromtxt(expr_path, dtype = np.float32, delimiter = ',')
    noi_index = np.genfromtxt(node_index_path, dtype = str)[noilist]
    node_index =np.genfromtxt(node_index_path, dtype = str)
    noi = noilist


    setOfPredictions=[]
    for condition in np.arange(pert.shape[0]):
        x_0 = np.zeros(pert.shape[1])

        trace = _simu(pert[condition], w, alpha, eps, x_0, dT=0.1, n_T = int(nT))
        trace_subset = trace[:,noi].transpose()
        xs = np.linspace(0, nT/10, int(nT)+1)
        real = expr[condition, noi]
        for t, trace_i in enumerate(trace_subset):
            plt.axhline(y = real[t], xmax = 0.98, ls="dashed",  alpha = 0.8,
                        color = sns.color_palette("deep")[t])

            plt.plot(xs, trace_i, color = sns.color_palette("deep")[t], 
                     label = noi_index[t], alpha = 0.8)
        plt.xlabel('Perturbation relaxation time')
        plt.ylabel('Perturbation expression level')
        plt.title('Expression perturbation trace for condition '+str(condition))
        plt.show()
        trace_end=trace[-1,:]
        setOfPredictions.append(trace_end)
    y=pd.read_csv(expr_path, header = None)

    setOfPredictions = np.array(setOfPredictions)

    setOfPredictionsDf = pd.DataFrame(setOfPredictions)

    display(setOfPredictionsDf)

    x_all = y.values
    x_all = np.delete(x_all,np.s_[-3:], axis=1)
    x_all = x_all.flatten()
    
    y_all = setOfPredictionsDf.values
    y_all = np.delete(y_all,np.s_[-3:], axis=1)
    y_all = y_all.flatten()
    
    # separate the protein and modulon nodes from the perturbation nodes
    json_path
    if json_path != "":
        json_data = json.load(open(json_path))
        n_protein_nodes = int(json_data['n_protein_nodes'])
        n_activity_nodes = int(json_data['n_activity_nodes'])
        n_x = json_data['n_x']

    x_prot = y.iloc[:,0:n_protein_nodes]
    y_prot = setOfPredictionsDf.iloc[:,0:n_protein_nodes]
    x_mod = y.iloc[:,n_protein_nodes+1:n_activity_nodes]
    y_mod = setOfPredictionsDf.iloc[:,n_protein_nodes+1:n_activity_nodes]
    

    plt.scatter(x_prot, y_prot, s = 15, alpha = 0.7, color="#74A6D1",zorder=3)
    plt.scatter(x_mod, y_mod, s = 15, alpha = 0.7, color="#3D6CA3",zorder=4)
    plt.legend(["Molecular (Protein) nodes","Phenotype (Modulon) nodes"], loc="upper right", frameon=False,
              handletextpad=0.1, fontsize=7.5)
    sns.regplot(x=x_all, y=y_all, scatter_kws={'s': 15, 'alpha': 0},line_kws={'color': 'r', 'alpha': 1})


    concateArray = np.concatenate((x_all[x_all<50],y_all[y_all<50]), axis=0)
    lower = np.min(concateArray)
    upper = np.max(concateArray)

    plt.xlim([lower*1.2, upper*1.2])
    plt.ylim([lower*1.2, upper*1.2])
    
    r = np.corrcoef(x_all, y_all)[0][1]
    
    plt.xlabel('Experimental measured response')
    plt.ylabel('Machine predicted response')
    plt.title("Correlation between predictions and \n experiments for the lowest MSE model across all conditions")

    
    plt.show()
    print('Pearson\'s ρ: \n ρ = %1.3f'%r)

    plt.title("Correlation between predictions and \n experiments for each condition")
    x_all = y.values
    y_all = setOfPredictionsDf.values

    rs = [np.corrcoef(x_all[i], y_all[i])[0][1] for i in range(setOfPredictionsDf.shape[0])]

    plt.hist(rs, bins = 100, color = 'grey', alpha = 0.6, rwidth=0.93)
    plt.axvline(x = r, linewidth=2, label = 'Median', color="#1B406C")
    plt.xlabel('Experiment-prediction correlation')
    plt.ylabel('Number of perturbation conditions')
    plt.show()


    isCircadian = True
    if isCircadian:

        try:
            pertDf= pd.DataFrame(pert)
            conditionDf=pd.DataFrame()
            # display(pertDf)
            print('pertDf shape')
            print(pertDf.shape)
            print(os.getcwd())

            if pertDf.shape[0] == 29:
                conditionDf['light_itensity']=pertDf.iloc[:,pertDf.shape[1]-3]
                conditionDf['sin_hours']=pertDf.iloc[:,pertDf.shape[1]-2]
                conditionDf['cos_hours']=pertDf.iloc[:,pertDf.shape[1]-1]
                print('conditionDf')
                display(conditionDf)
                conditionDf['circadian_time']=[0.5,2,4,6,8,9,10,12,10,4,6,8,9,10,12,8,8.25,8.5,9,9.25,9.5,10,8,8.25,8.5,9,9.25,9.5,10]
                alternateConditionDf = conditionDf.copy()

                segmentList=['lowlight_dawn','lowlight_dawn','lowlight_dawn','lowlight_dawn','lowlight_dawn','lowlight_dawn','lowlight_dawn','lowlight_dawn','clearday_dawn','clearday_dawn','clearday_dawn','clearday_dawn','clearday_dawn','clearday_dawn','clearday_dawn','highlight_pulse','highlight_pulse','highlight_pulse','highlight_pulse','highlight_pulse','highlight_pulse','highlight_pulse','shade_pulse','shade_pulse','shade_pulse','shade_pulse','shade_pulse','shade_pulse','shade_pulse']

                alternateConditionDf['Segment']=segmentList
                alternateConditionDf.at[8,'circadian_time']=2
            
            else:

                sin_cos_encoding = pd.read_csv("Light_Circadian_Oshea/Perturbations/Light_Sin_Cos.csv")

                display(sin_cos_encoding)
                sin_cos_encoding.drop(columns=['SRA','Condition_pi'],inplace=True)

                for i in range(sin_cos_encoding.shape[0]):

                    sin_cos_encoding.at[i,'Condition'] = str(sin_cos_encoding.at[i,'Condition']).split('wt_')[1]


                sin_cos_encoding.columns = ['Segment','circadian_time','light_itensity','sin_hours','cos_hours']

            
                display(sin_cos_encoding)
                alternateConditionDf=sin_cos_encoding.copy()
            print()
            print('-------------- lowlight_dawn --------------')
            print()
            segment='lowlight_dawn'
            count=0
            for i in range(setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].shape[1]):
                linOut=linregress(alternateConditionDf[alternateConditionDf['Segment']==segment]['circadian_time'],setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].iloc[:,i])

                if linOut.pvalue < 0.9:
                    print(linOut)
                    print('Range: '+str(np.max(setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].iloc[:,i])-np.min(setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].iloc[:,i])))
                    plt.scatter(x=alternateConditionDf[alternateConditionDf['Segment']==segment]['circadian_time'],y=np.exp2(setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].iloc[:,i]))
                    plt.xlabel('Circadian Time')
                    plt.ylabel(node_index[i])
                    plt.show()
                    count+=1
            print()
            print('-------------- clearday_dawn --------------')
            print()
            segment='clearday_dawn'
            count=0
            print(alternateConditionDf[alternateConditionDf['Segment']==segment].index)
            display(setOfPredictionsDf)
            print(setOfPredictionsDf.shape)
            print(setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].shape[1])
            
            for i in range(setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].shape[1]):
                linOut=linregress(alternateConditionDf[alternateConditionDf['Segment']==segment]['circadian_time'],setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].iloc[:,i])

                if linOut.pvalue < 0.9:
                    print(linOut)
                    print('Range: '+str(np.max(setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].iloc[:,i])-np.min(setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].iloc[:,i])))
                    plt.scatter(x=alternateConditionDf[alternateConditionDf['Segment']==segment]['circadian_time'],y=setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].iloc[:,i])
                    plt.xlabel('Circadian Time')
                    plt.ylabel(node_index[i])
                    plt.show()
                    count+=1



            print()
            print('-------------- highlight_pulse --------------')
            print()
            segment='highlight_pulse'
            for i in range(setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].shape[1]):
                linOut=linregress(alternateConditionDf[alternateConditionDf['Segment']==segment]['circadian_time'],setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].iloc[:,i])

                if linOut.pvalue < 0.9:
                    print(linOut)
                    print('Range: '+str(np.max(setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].iloc[:,i])-np.min(setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].iloc[:,i])))
                    plt.scatter(x=alternateConditionDf[alternateConditionDf['Segment']==segment]['circadian_time'],y=setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].iloc[:,i])
                    plt.xlabel('Circadian Time')
                    plt.ylabel(node_index[i])
                    plt.show()

            print()
            print('-------------- shade_pulse --------------')
            print()

            segment='shade_pulse'
            for i in range(setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].shape[1]):
                linOut=linregress(alternateConditionDf[alternateConditionDf['Segment']==segment]['circadian_time'],setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].iloc[:,i])

                if linOut.pvalue < 0.9:
                    print(linOut)
                    print('Range: '+str(np.max(setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].iloc[:,i])-np.min(setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].iloc[:,i])))
                    plt.scatter(x=alternateConditionDf[alternateConditionDf['Segment']==segment]['circadian_time'],y=setOfPredictionsDf.iloc[alternateConditionDf[alternateConditionDf['Segment']==segment].index].iloc[:,i])
                    plt.xlabel('Circadian Time')
                    plt.ylabel(node_index[i])
                    plt.show()
        except:
            print('Circadian error')

    
    return (x_all,y_all)

def PlotBestData(pert_path,expr_path,node_index_path,alpha_file,eps_file,W_file,y_hat_file,noilist, condition = 0, nT = 400, nlength=106):
    alpha = pd.read_csv(glob.glob(alpha_file)[0], index_col = 0).values
    eps = pd.read_csv(glob.glob(eps_file)[0], index_col = 0).values
    w = pd.read_csv(glob.glob(W_file)[0], index_col = 0).values
    y_hat = pd.read_csv(glob.glob(y_hat_file)[0], index_col = 0)
    
    
    pert = np.genfromtxt(pert_path, dtype = np.float32, delimiter = ',')
    expr = np.genfromtxt(expr_path, dtype = np.float32, delimiter = ',')
    noi_index = np.genfromtxt(node_index_path, dtype = str)[noilist]
    
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
    plt.show()
    
def PlotFigure(root,resultsDir,trialDir,noilist):
    
    loss = pd.read_csv(str(resultsDir)+str(trialDir)+'/record_eval.csv',usecols=range(8),header=None)

    if isinstance(loss.loc[0,0],str):

        loss = pd.read_csv(str(resultsDir)+str(trialDir)+'/record_eval.csv',usecols=range(8))#,header=None)

    display(loss)
    
    display(loss[loss["epoch"]==-1])
    idx = np.where([x!='None' for x in loss[loss["epoch"]==-1]['test_mse']])[0]
    print(idx)

    minVal = float(loss[loss["epoch"]==-1]['test_mse'].values[idx[0]])
    minIdx=idx[0]
    previousIndexRow=np.where([x==loss[loss["epoch"]==-1]['test_mse'].values[idx[0]] for x in loss['test_mse']])[0][0]

    for i in idx:
        if float(loss[loss["epoch"]==-1]['test_mse'].values[i]) < minVal:
            minVal=float(loss[loss["epoch"]==-1]['test_mse'].values[i])
            previousIndexRow=np.where([x==loss[loss["epoch"]==-1]['test_mse'].values[minIdx] for x in loss['test_mse']])[0][0]
            minIdx=i
            
    minIndexRow=np.where([x==loss[loss["epoch"]==-1]['test_mse'].values[minIdx] for x in loss['test_mse']])[0][0]
    print('MinVal: '+str(minVal))
    print('minIdx: '+str(minIdx))
    print(minIndexRow)
    

    subsetLoss = loss[previousIndexRow+1:minIndexRow+1]
    
    display(subsetLoss)
    print(idx)
    print('substage: '+str(np.where(x==minIdx for x in idx)))
    
    return


def ProduceLossDataFrame(mainroot,trialDir,numberOfSeeds=1000):

    print("ProduceLossDataFrame")
    dataMatrix=[]
    for index_num in range(numberOfSeeds):
        print('Loss for '+str(index_num))
        if len(str(index_num))==3:
            index = str(index_num)
        elif len(str(index_num))==2:
            index = "0"+str(index_num)
        else:
            index = "00"+str(index_num)
        try:
            
            os.chdir(str(trialDir)+"/seed_"+str(index)+"/")
           
            loss = pd.read_csv('record_eval.csv',usecols=range(8),header=None)
            if isinstance(loss.loc[0,0],str):
                loss = pd.read_csv('record_eval.csv',usecols=range(8))#,header=None)

            idx = np.where([x!='None' for x in loss[loss["epoch"]==-1]['test_mse']])[0]
            substageNum=[]
            substageLoss=[]
            sN=1
            for val in idx:
                if float(loss[loss["epoch"]==-1]['valid_mse'].values[val-1]) < 1000:
                    substageNum.append(sN)
                    #substage only completed and exported is test_MSe < 1000
                    sN+=1
                    substageLoss.append(loss[loss["epoch"]==-1]['test_mse'].values[val])

            for file in glob.glob("*_best.alpha.loss.*.csv"):
                try:
                    substageNum=file.split("_")[0]
                    validLoss = file.split("loss.")[1].split(".csv")[0]

                    alphaName=str(substageNum)+"_best.alpha.loss."+str(validLoss)+".csv"
                    epsName=str(substageNum)+"_best.eps.loss."+str(validLoss)+".csv"
                    WName=str(substageNum)+"_best.W.loss."+str(validLoss)+".csv"
                    y_hatName=str(substageNum)+"_best.y_hat.loss."+str(validLoss)+".csv"

                    data = ['seed_'+str(index),substageNum,validLoss,substageLoss[int(substageNum)-1],alphaName,epsName,WName,y_hatName]
                    dataMatrix.append(data)
                except:
                    a=1
        except:
            print('Error with seed '+str(index)+' may not exist')
        
        os.chdir(mainroot)
        
    trialDataFrame=pd.DataFrame(dataMatrix,columns=['seed','substage','valid_loss','test_loss','alpha_file','eps_file','W_file','y_hat_file'])
    return trialDataFrame

    
def AnalyseTrialDataFrame(trialDataFrame,ifVerbose=False):
    print("AnalyseTrialDataFrame")
    if ifVerbose:

        validFloat=[float(x) for x in trialDataFrame['valid_loss'].values.tolist()]
        plt.figure()
        plt.scatter(x=np.arange(trialDataFrame.shape[0]),y=validFloat,s=20)
        plt.ylabel('Loss during validation')
        plt.show()

        testFloat=[float(x) for x in trialDataFrame['test_loss'].values.tolist()]
        plt.figure()
        plt.scatter(x=np.arange(trialDataFrame.shape[0]),y=testFloat,s=20)
        plt.ylabel('Loss during test')
        plt.show()

    minVal = float(trialDataFrame['test_loss'].values[0])
    minIdx=0
    for i in range(len(trialDataFrame['test_loss'].values)):
        val=float(trialDataFrame['test_loss'].values[i])
        if minVal>val:
            minVal=val
            minIdx=i
    print('Min test_loss: '+str(minVal))
    print('Min test_loss index: '+str(minIdx))
    print(trialDataFrame.iloc[minIdx])
    print()
    minTestSeed = trialDataFrame.iloc[minIdx]['seed']
    minTestSubstage = trialDataFrame.iloc[minIdx]['substage']
    minTestLoss = trialDataFrame.iloc[minIdx]['test_loss']

    minVal = float(trialDataFrame['valid_loss'].values[0])
    minIdx=0
    for i in range(len(trialDataFrame['valid_loss'].values)):
        val=float(trialDataFrame['valid_loss'].values[i])
        if minVal>val:
            minVal=val
            minIdx=i
    print('Min valid_loss: '+str(minVal))
    print('Min valid_loss index: '+str(minIdx))
    print(trialDataFrame.iloc[minIdx])
    print()

    minValidSeed = trialDataFrame.iloc[minIdx]['seed']
    minValidSubstage = trialDataFrame.iloc[minIdx]['substage']
    minValidLoss = trialDataFrame.iloc[minIdx]['valid_loss']

    return minTestSeed,minTestSubstage,minTestLoss,minValidSeed,minValidSubstage,minValidLoss


def plotBestModel(mainRoot,trialDataFrame,resultsDir,dataDir,trialOutputDir,seed,substage,noilist):
    
    for i in range(trialDataFrame.shape[0]):
        if trialDataFrame.iloc[i]['seed']==seed:
            if trialDataFrame.iloc[i]['substage'] == substage:
                print('Test seed: '+str(seed))
                alpha_file=trialDataFrame.iloc[i]["alpha_file"]
                eps_file=trialDataFrame.iloc[i]["eps_file"]
                W_file=trialDataFrame.iloc[i]["W_file"]
                y_hat_file=trialDataFrame.iloc[i]["alpha_file"]
                seed = trialDataFrame.iloc[i]["seed"]
                os.chdir(mainRoot)
                os.chdir(str(resultsDir)+"/"+str(trialOutputDir)+"/"+str(seed)+"/")

                pert_path='../../../'+str(dataDir)+'/pert_matr.csv'
                expr_path='../../../'+str(dataDir)+'/expr_matr.csv'
                node_index_path='../../../'+str(dataDir)+'/node_index.csv'
                json_path='../../../'+str(dataDir)+'/config.json'

                (x_all,y_all) =PlotBestDataAllConditions(mainRoot,pert_path,expr_path,node_index_path,alpha_file,eps_file,W_file,y_hat_file,noilist,json_path)

                return 
    print('Error: best model not found')
    return 



def compile_W_to_3d_array(folder_path):
    """
    Given multiple Wij matricies in a directory compile them into a single 3D matrix
    Input: Directory
    Output: index - names of nodes in network
            array_3d - compiled weights
    """

    # Filter only the desired CSV files
    csv_files = []

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for file in filenames:
            # regex pattern to match files starting with number, underscore, 'best.W.rank_' and ending with '.csv'
            if re.match(r'^\d+_best\.W\.loss.*', file):
                csv_files.append(os.path.join(dirpath, file))

    # Read and store each CSV file in a list
    dataframes = [pd.read_csv(csv_file, index_col=0) for csv_file in csv_files]
    
    # Get the shape and inidices 
    shape = dataframes[0].shape
    index = dataframes[0].index

    # Create a 3D array of zeros with the dimensions (rows, columns, number_of_files)
    array_3d = np.zeros((shape[0], shape[1], len(dataframes)))

    # Fill the 3D array with the values from the DataFrames
    for i, df in enumerate(dataframes):
        array_3d[:, :, i] = df.values

    return index, array_3d

def calc_interaction_metrics(three_d_array, index):
    """
    Calculate statistics across the matrix 
    one sided student's t-test to test if weights are significantly different than 0
    False discovery correction using benjamini-hochberg correction
    Input: Outputs from compile_W_to_3d_array
    Output: A matrix of weight statistics for each potential connection
    """

    rows, columns, _ = three_d_array.shape
    p_values = []
    t_values = []
    DoF_values = []
    mean_values = []
    std_values = []
    q1_values = []
    q2_values = []
    q3_values = []
    i_values = []
    j_values = []

    # Perform one-sample t-tests and collect statistics
    for i in range(rows):
        for j in range(columns):
            samples = three_d_array[i, j]
            t_value, p_value = stats.ttest_1samp(samples, 0)
            p_values.append(p_value)
            t_values.append(t_value)
            DoF_values.append(len(samples)-1)
            
            mean = np.mean(samples)
            mean_values.append(mean)
            
            std_dev = np.std(samples, ddof=1)
            std_values.append(std_dev)
            
            quantile_25 = np.percentile(samples, 25)
            q1_values.append(quantile_25)
            
            quantile_50 = np.percentile(samples, 50)
            q2_values.append(quantile_50)
            
            quantile_75 = np.percentile(samples, 75)
            q3_values.append(quantile_75)
            
            i_values.append(i)
            j_values.append(j)


    # Create a dataframe
    interaction_metrics_df = pd.DataFrame({
        'i': i_values,
        'j': j_values,
        'mean': mean_values,
        'std_dev': std_values,
        'Q1': q1_values,
        'median': q2_values,
        'Q3': q3_values,
        't_value': t_values,
        'p_value': p_values,
        'DoF':DoF_values
    })
    
    
    interaction_metrics_df[['effector', 'target']] = [(index[val1], index[val2]) for val1, val2 in 
                                                      zip(interaction_metrics_df['j'], interaction_metrics_df['i'])]
    
    valid_metrics_df = interaction_metrics_df.dropna().copy()

    rejected, corrected_p_values = smt.multipletests(valid_metrics_df['p_value'].values, method='fdr_bh')[:2]
    valid_metrics_df['q_value'] = corrected_p_values
    
    #Return all values in the datframe with adjusted fdr p values
    final_df = valid_metrics_df[['effector', 'target', 'mean', 'std_dev', 'Q1', 'median', 'Q3', 't_value', 'p_value', 'q_value','DoF']]
    
    return final_df

def FromTrialDirectoryProduceNetwork(resultsDir,tDir):
    
    folder = resultsDir+"/"+tDir+"/"

    indicies, Wij = compile_W_to_3d_array(folder)

    metrics_df = calc_interaction_metrics(Wij, indicies)
    statistics = metrics_df

    display(metrics_df)
    Wij_index = indicies
    statistics['effector_gene'] = statistics['effector'].str.strip("TF_")
    statistics['target_gene'] = statistics['target'].str.strip(r"TF_|gene_")
    statistics['t_value_abs'] = abs(statistics['t_value'])

    TRN = pd.read_csv("../Data/annotation/TRN_complete.csv", index_col=0)

    metadata = pd.read_csv('../Data/metadata/Metadata_Perturbation_Passing_QC.csv', index_col=0)
    metadata.rename(columns={"Project_tag":"project", "Condition":"condition"}, inplace=True)

    statistics_TRN = statistics.merge(TRN, left_on=['effector_gene', 'target_gene'], 
                                      right_on=['regulatoryGene', 'targetGene'], how='left')
    statistics_TRN_known = statistics.merge(TRN, left_on=['effector_gene', 'target_gene'], 
                                            right_on=['regulatoryGene', 'targetGene'], how='inner')

    top2 = []
    for effector, targets in statistics_TRN.groupby('effector'):
        targets_sorted = targets.sort_values('t_value_abs', ascending=False)
        top2.append(targets_sorted.head(2))

    top2_interactions_df = pd.concat(top2)

    effector_new=[]
    for gene in top2_interactions_df['effector'].tolist():
        
        if "TF_" in gene:
            effector_new.append(gene.split('TF_')[1])
        else:
            effector_new.append(gene)
            
    target_new=[]
    for gene in top2_interactions_df['target'].tolist():
        
        if "TF_" in gene:
            target_new.append(gene.split('TF_')[1])
        else:
            target_new.append(gene)
            
    top2_interactions_df['effector'] = effector_new
    top2_interactions_df['target'] = target_new
    
    GN = nx.DiGraph()
    # GN.add_weighted_edges_from([tuple(x) for x in top2_interactions_df[['effector', 'target', 'mean']].values])
    
    print('statistics[effector_gene]')
    print(statistics['effector_gene'].tolist())
    print()

    print('statistics[target_gene]')
    print(statistics['target_gene'].tolist())
    print()

    print('statistics[mean]')
    print(statistics['mean'].tolist())
    print()

    edgeTupleList=[]
    for i in range(len(statistics['effector_gene'].tolist())):
        edgeTupleList.append((statistics['effector_gene'].tolist()[i],statistics['target_gene'].tolist()[i],statistics['mean'].tolist()[i]))
    display(edgeTupleList)    
    GN.add_weighted_edges_from(edgeTupleList)

    # add a label attribute to each node based on the node function
    for node in GN.nodes():
        # if node.startswith('gene'):
        #     GN.nodes[node]['label'] = 'pheno'
        #     GN.nodes[node]['trophic_lvl'] = 2
        # elif node.startswith('TF'):
        #     GN.nodes[node]['label'] = 'molecular'
        #     GN.nodes[node]['trophic_lvl'] = 1
        # else:
        #     GN.nodes[node]['label'] = 'pert'
        #     GN.nodes[node]['trophic_lvl'] = 0
        if node.startswith('Hour'):
            GN.nodes[node]['label'] = 'pert'
            GN.nodes[node]['trophic_lvl'] = 0
        elif node.startswith('Light'):
            GN.nodes[node]['label'] = 'pert'
            GN.nodes[node]['trophic_lvl'] = 0
        elif node.startswith('TF'):
            GN.nodes[node]['label'] = 'molecular'
            GN.nodes[node]['trophic_lvl'] = 1
        else:
            GN.nodes[node]['label'] = 'pheno'
            GN.nodes[node]['trophic_lvl'] = 2

    node_attrs = [(node, data['label'], data['trophic_lvl']) for node, data in GN.nodes.data()]
    node_labels = pd.DataFrame(node_attrs, columns=['node', 'label', 'level'])
    node_labels['label'].value_counts()
    
    return GN,statistics_TRN_known,statistics_TRN,top2_interactions_df,Wij_index,metrics_df,Wij

    
def ProduceCombinedNetwork(G,H):
    
    # color key
    # red if in only G
    # blue if in only H
    # green is shared by G and H
    
    F = nx.compose(G,H)
    edge_color=[]
    node_color=[]
    for edge in F.edges():
        # print(edge)

        # for edge in G.edges():
        #     print(edge)
        

        ifFound=False

        if edge in G.edges() and edge in H.edges():
            F[edge[0]][edge[1]]['color'] =2
            ifFound=True
        elif edge in G.edges():
            F[edge[0]][edge[1]]['color'] =1
            ifFound=True
        elif edge in H.edges():
            F[edge[0]][edge[1]]['color'] =0
            ifFound=True

        if not ifFound:
            print('Error edge not found: ')
            print(edge)

    for node in F.nodes(data=True):
        # print(node)
        # print(node[0])
        # print(F.nodes[node[0]])

        # for node in G.nodes():
        #     print(node)

        # if node in G.nodes():
        #     print('here')

        ifFound=False

        if node[0] in H.nodes(data=True) and node[0] in G.nodes(data=True):
            F.nodes[node[0]]['color']  =2
            ifFound=True

        elif node[0] in G.nodes(data=True):
            F.nodes[node[0]]['color'] =1
            ifFound=True

        elif node[0] in H.nodes(data=True):
            F.nodes[node[0]]['color']  =0
            ifFound=True

        if not ifFound:
            print('Error node not found: ')
            print(node)
  
    for edge in F.edges(data=True):
        # print(edge)
        if 1 == F[edge[0]][edge[1]]['color']:
            edge_color.append('r')
        elif 2 == F[edge[0]][edge[1]]['color']:
            edge_color.append('g')
        elif 0 == F[edge[0]][edge[1]]['color']:
            edge_color.append('b')

    for node in F.nodes(data=True):
        if 1 == node[1]['color']:
            node_color.append('r')
        elif 2 == node[1]['color']:
            node_color.append('g')
        elif 0 == node[1]['color']:
            node_color.append('b')

    return F, node_color, edge_color


def ProduceIntersectionNetwork(G,H):

    # color key
    # red if in only G
    # blue if in only H
    # green is shared by G and H
    
    I = nx.intersection(G,H)
    # I.remove_nodes_from(list(nx.isolates(I)))
    
    for edge in I.edges():

        ifFound=False

        if edge in G.edges() and edge in H.edges():
            I[edge[0]][edge[1]]['color'] =2
            ifFound=True
        elif edge in G.edges():
            I[edge[0]][edge[1]]['color'] =1
            ifFound=True
        elif edge in H.edges():
            I[edge[0]][edge[1]]['color'] =0
            ifFound=True

        if not ifFound:
            print('Error edge not found: ')
            print(edge)


    for node in I.nodes(data=True):

        ifFound=False

        if node[0] in H.nodes(data=True) and node in G.nodes(data=True):
            I.nodes[node[0]]['color']  =2
            ifFound=True

        elif node[0] in G.nodes(data=True):
            I.nodes[node[0]]['color'] =1
            ifFound=True

        elif node[0] in H.nodes(data=True):
            I.nodes[node[0]]['color']  =0
            ifFound=True

        if not ifFound:
            print('Error node not found: ')
            print(node)
    
    node_color=[]    
    for node in I.nodes(data=True):

        if node in H.nodes(data=True) and node in G.nodes(data=True):
            I.nodes[node[0]]['color']  =2

        elif node in G.nodes(data=True):
            I.nodes[node[0]]['color'] =1

        elif node in H.nodes(data=True):
            I.nodes[node[0]]['color']  =0

    edge_color=[]
    for edge in I.edges(data=True):

        if 1 == I[edge[0]][edge[1]]['color']:
            edge_color.append('r')
        elif 2 == I[edge[0]][edge[1]]['color']:
            edge_color.append('g')
        elif 0 == I[edge[0]][edge[1]]['color']:
            edge_color.append('b')
            
            
    return I, node_color, edge_color



def ProduceIntersectionMappedToPrimaryNetwork(G,H):
    
    # highlight the section of network G that is also in H

    # color key
    # red if in only G
    # blue if in only H
    # green is shared by G and H

    # G = nx.path_graph(3)
    # H = nx.path_graph(5)
    # I = G.copy()
    # I.remove_nodes_from(n for n in G if n not in H)
    # I.remove_edges_from(e for e in G.edges if e not in H.edges)
    
    I = nx.intersection_all([G,H])
    # I.remove_nodes_from(list(nx.isolates(I)))
    
    J = nx.compose(G,I)


    for edge in J.edges():
    # for edge in J.edges():

        ifFound=False

        if edge in G.edges() and edge in I.edges():
            # print(edge)
            J[edge[0]][edge[1]]['color'] =2
            ifFound=True
        elif edge in G.edges():
            J[edge[0]][edge[1]]['color'] =1
            ifFound=True
        elif edge in I.edges():
            J[edge[0]][edge[1]]['color'] =0
            ifFound=True

        if not ifFound:
            print('Error edge not found: ')
            print(edge)

    # for edge in J.edges(data=True):
    #     print(edge)
    node_color=[]    
    # for node in J.nodes(data=True):
    for node in J.nodes(data=True):

        # for j in J.nodes():
        #     print(j)
        if node[0] in I.nodes(data=True) and node[0] in G.nodes(data=True):
            J.nodes[node[0]]['color']  =2
            ifFound=True

        elif node[0] in G.nodes(data=True):
            J.nodes[node[0]]['color'] =1
            ifFound=True

        elif node[0] in I.nodes(data=True):
            J.nodes[node[0]]['color']  =0
            ifFound=True



        if not ifFound:
            print('Error node not found: ')
            print(node)

        # if node in I.nodes(data=True) and node in G.nodes(data=True):
        #     J.nodes[node[0]]['color']  =2

        # elif node in G.nodes(data=True):
        #     J.nodes[node[0]]['color'] =1

        # elif node in I.nodes(data=True):
        #     J.nodes[node[0]]['color']  =0

    print()
    print()
    print()


    edge_color=[]
    for edge in J.edges(data=True):

        if 1 == J[edge[0]][edge[1]]['color']:
            edge_color.append('tab:orange')
            # print(edge)
        elif 2 == J[edge[0]][edge[1]]['color']:
            
            edge_color.append('tab:blue')
        elif 0 == J[edge[0]][edge[1]]['color']:
            edge_color.append('tab:green')

    for node in J.nodes(data=True):
        if 1 == node[1]['color']:
            node_color.append('tab:orange')
        elif 2 == node[1]['color']:
            node_color.append('tab:blue')
        elif 0 == node[1]['color']:
            node_color.append('tab:green')

    return J, node_color, edge_color


def ProduceIntersectionMappedToThresholdPrimaryNetwork(G_original,H,weightThreshold=0):
    
    maxWeight=1
    for edge in G_original.edges(data="weight"):
        # if np.abs(edge[2]) >maxWeight:
        if edge[2] >maxWeight:
            maxWeight=edge[2]

    reducedEdges=[]
    for edge in G_original.edges(data="weight"):
        # if np.abs(edge[2]) > weightThreshold:
        if edge[2] > weightThreshold:
            reducedEdges.append(edge)

    # print(reducedEdges)
    nodesToRemove=[]
    for node in G_original.nodes(data=True):
        # print(node)
        isFound=False
        for edge in reducedEdges:
            # print(node[0])
            if node[0] in edge:
                isFound=True
                break
        if not isFound:
            # print('here')
            nodesToRemove.append(node)

    G = G_original.copy()

    for node in nodesToRemove:
        # print(node[0])
        G.remove_node(node[0])
        
    
    G_mapped, node_color, edge_color = ProduceIntersectionMappedToPrimaryNetwork(G,H)    
        
      
    
    
    return G_mapped, node_color, edge_color, reducedEdges,maxWeight    

    
    
    