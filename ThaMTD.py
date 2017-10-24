from FuncParcel import *
import numpy as np
import nibabel as nib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from dFC_graph import coupling
from matplotlib import colors
import pickle as pickle


#where ts data are
TDSigEI_TS = '/home/despoB/TRSEPPI/TDSigEI/TS_1Ds/'
TRSE_TS = '/home/despoB/TRSEPPI/TRSEPPI/TS_1Ds/'
Data='/home/despoB/kaihwang/bin/ThaMTD/Data/'


def coupling(data,window):
    """
        creates a functional coupling metric from 'data'
        data: should be organized in 'time x nodes' matrix
        smooth: smoothing parameter for dynamic coupling score
        # from PD
        #By default, the result is set to the right edge of the window. 
        This can be changed to the center of the window by setting center=True.
    """
    
    #define variables
    [tr,nodes] = data.shape
    der = tr-1
    td = np.zeros((der,nodes))
    td_std = np.zeros((der,nodes))
    data_std = np.zeros(nodes)
    mtd = np.zeros((der,nodes,nodes))
    sma = np.zeros((der,nodes*nodes))
    
    #calculate temporal derivative
    for i in range(0,nodes):
        for t in range(0,der):
            td[t,i] = data[t+1,i] - data[t,i]
    
    
    #standardize data
    for i in range(0,nodes):
        data_std[i] = np.std(td[:,i])
    
    td_std = td / data_std
   
   
    #functional coupling score
    for t in range(0,der):
        for i in range(0,nodes):
            for j in range(0,nodes):
                mtd[t,i,j] = td_std[t,i] * td_std[t,j]


    #temporal smoothing
    temp = np.reshape(mtd,[der,nodes*nodes])
    sma = pd.rolling_mean(temp,window, center=True)
    sma = np.reshape(sma,[der,nodes,nodes])
    
    return (mtd, sma)


def get_MTD(input1, input2, window):
	MTD = coupling(np.array([input1,input2]).T, window)[1][:,0,1]
	MTD[np.isnan(MTD)] = 0
	MTD = np.insert(MTD,0,0)

	return MTD


def censor_ts(ffa_ts, ppa_ts, vc_ts, tha_ts):
	'''3dDeconvolve outputed censor data with 0, make sure those are removed before any calculations''' 
	ffa_zeros = ffa_ts==0
	ppa_zeros = ppa_ts==0
	vc_zeros = vc_ts==0
	tha_zeros = tha_ts==0	

	zeros = np.sum(tha_zeros,axis=1)+ffa_zeros+vc_zeros+ppa_zeros
	zeros = zeros ==0
	ffa_ts = ffa_ts[zeros]
	ppa_ts = ppa_ts[zeros]
	vc_ts = vc_ts[zeros]
	tha_ts = tha_ts[zeros]

	return ffa_ts, ppa_ts, vc_ts, tha_ts


def fit_linear_model(y,x):
	'''multiple linear regression using OLS in statsmodel, shorthand function'''

	#add constant
	x = sm.add_constant(x)   
	est = sm.OLS(y, x).fit() #OLS fit
	return est 



def run_reg_model(subject, condition, window, single = True):
	
	fn = Data + '%s_FIR_%s_FFA.ts' %(subject, condition)
	ffa_ts = np.loadtxt(fn)
	
	fn = Data + '%s_FIR_%s_PPA.ts' %(subject, condition)
	ppa_ts = np.loadtxt(fn)
	
	fn = Data + '%s_FIR_%s_VC.ts' %(subject, condition)
	vc_ts = np.loadtxt(fn)
	
	fn = Data + '%s_FIR_%s_THA.ts' %(subject, condition)
	tha_ts = np.loadtxt(fn)		

	ffa_ts, ppa_ts, vc_ts, tha_ts = censor_ts(ffa_ts, ppa_ts, vc_ts, tha_ts)

	ffa_vc_mtd = get_MTD(ffa_ts, vc_ts, window)
	ppa_vc_mtd = get_MTD(ppa_ts, vc_ts, window)

	ffa_e = fit_linear_model(ffa_vc_mtd, tha_ts)
	ppa_e = fit_linear_model(ppa_vc_mtd, tha_ts)

	if single:
		ffa_e=np.zeros(15)
		ppa_e=np.zeros(15)
		for i in range(15):
			e = fit_linear_model(ffa_vc_mtd, tha_ts[:,i])
			ffa_e[i] = e.tvalues[1]
			e = fit_linear_model(ppa_vc_mtd, tha_ts[:,i])
			ppa_e[i] = e.tvalues[1]

	return ffa_e, ppa_e


def loop_regressions(Subjects, Conditions, window):

	Nuclei = ['AN','VM', 'VL', 'MGN', 'MD', 'PuA', 'LP', 'IL', 'VA', 'Po', 'LGN', 'PuM', 'PuI', 'PuL', 'VP']
	df = pd.DataFrame(columns=('Subject', 'Thalamic Nuclei', 'Condition', 'FFA-VC', 'PPA-VC'), dtype=float) 

	for subject in Subjects:

		sdf =  pd.DataFrame(columns=('Subject', 'Thalamic Nuclei', 'Condition', 'FFA-VC', 'PPA-VC'), dtype=float)  

		i=0
		for condition in Conditions:
			ffa_e, ppa_e= run_reg_model(subject, condition, window)

			for ii, n in enumerate(Nuclei):
				sdf.loc[i, 'Subject'] = subject
				sdf.loc[i, 'Thalamic Nuclei'] = n
				sdf.loc[i, 'Condition'] = condition
				sdf.loc[i, 'FFA-VC'] = ffa_e[ii]
				sdf.loc[i, 'PPA-VC'] = ppa_e[ii]
				i=i+1
		df = pd.concat([df, sdf])	
	
	return df		


if __name__ == "__main__":

	#TDSigEI
	Subjects = np.loadtxt(Data+'TDSigEI_subject', dtype=int)
	Conditions =['FH','HF', 'Fp', 'Hp', 'Fo', 'Ho']
	window = 15
	TD_df = loop_regressions(Subjects, Conditions, window)


	#TRSE
	Subjects = np.loadtxt(Data+'TRSE_subject', dtype=int)
	Conditions =['FH','HF', 'CAT', 'BO']
	window = 15
	TR_df = loop_regressions(Subjects, Conditions, window)







