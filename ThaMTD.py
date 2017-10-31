from __future__ import division 
from FuncParcel import *
import numpy as np
import nibabel as nib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
#from dFC_graph import coupling
from matplotlib import colors
import itertools
from scipy.stats import norm
from math import exp,sqrt
Z = norm.ppf

#where ts data are
TDSigEI_TS = '/home/despoB/TRSEPPI/TDSigEI/TS_1Ds/'
TRSE_TS = '/home/despoB/TRSEPPI/TRSEPPI/TS_1Ds/'
Data='/home/despoB/kaihwang/bin/ThaMTD/Data/'
Nuclei = ['AN', 'MD', 'VL', 'Pu', 'CD', 'PT', 'PA']

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


def censor_ts(ffa_ts, ppa_ts, vc_ts, tha_ts, bg_ts):
	'''3dDeconvolve outputed censor data with 0, make sure those are removed before any calculations''' 
	ffa_zeros = ffa_ts==0
	ppa_zeros = ppa_ts==0
	vc_zeros = vc_ts==0
	tha_zeros = tha_ts==0	
	bg_zeros = bg_ts==0

	zeros = np.sum(tha_zeros,axis=1)+ffa_zeros+vc_zeros+ppa_zeros+bg_zeros
	zeros = zeros ==0
	ffa_ts = ffa_ts[zeros]
	ppa_ts = ppa_ts[zeros]
	vc_ts = vc_ts[zeros]
	tha_ts = tha_ts[zeros]
	bg_ts = bg_ts[zeros]

	return ffa_ts, ppa_ts, vc_ts, tha_ts, bg_ts


def fit_linear_model(y,x):
	'''multiple linear regression using OLS in statsmodel, shorthand function'''

	#add constant
	x = sm.add_constant(x)   
	est = sm.OLS(y, x).fit() #OLS fit
	return est 



def load_ts(subject, condition):
	fn = Data + '%s_FIR_%s_FFA.ts' %(subject, condition)
	ffa_ts = np.loadtxt(fn)
	
	fn = Data + '%s_FIR_%s_PPA.ts' %(subject, condition)
	ppa_ts = np.loadtxt(fn)
	
	fn = Data + '%s_FIR_%s_VC.ts' %(subject, condition)
	vc_ts = np.loadtxt(fn)
	
	fn = Data + '%s_FIR_%s_THA.ts' %(subject, condition)
	tha_ts = np.loadtxt(fn)	

	fn = Data + '%s_FIR_%s_BG.ts' %(subject, condition)
	bg_ts = np.loadtxt(fn)	

	return ffa_ts, ppa_ts, vc_ts, tha_ts, bg_ts



def run_reg_model(subject, condition, window, single = True, smooth=False):
	''' regression models to regress MTD estimates onto thalamus timecourses'''

	ffa_ts, ppa_ts, vc_ts, tha_ts, bg_ts = load_ts(subject, condition)

	if smooth:
		tha_ts = pd.rolling_mean(tha_ts, window)
		tha_ts[np.isnan(tha_ts)] = 0

	ffa_ts, ppa_ts, vc_ts, tha_ts, bg_ts = censor_ts(ffa_ts, ppa_ts, vc_ts, tha_ts, bg_ts)

	ffa_vc_mtd = get_MTD(ffa_ts, vc_ts, window)
	ppa_vc_mtd = get_MTD(ppa_ts, vc_ts, window)

	tha_ts = np.hstack((tha_ts,bg_ts))
	
	e = fit_linear_model(ffa_vc_mtd, tha_ts)
	ffa_e = np.zeros(len(Nuclei))
	for i in range(len(Nuclei)):
		ffa_e[i] = e.tvalues[i+1]

	e = fit_linear_model(ppa_vc_mtd, tha_ts)
	ppa_e = np.zeros(len(Nuclei))
	for i in range(len(Nuclei)):
		ppa_e[i] = e.tvalues[i+1]


	if single:
		ffa_e=np.zeros(len(Nuclei))
		ppa_e=np.zeros(len(Nuclei)) #np.zeros(15)
		for i in range(len(Nuclei)):
			e = fit_linear_model(ffa_vc_mtd, tha_ts[:,i])
			ffa_e[i] = e.params[1] #e#.tvalues[1]
			e = fit_linear_model(ppa_vc_mtd, tha_ts[:,i])
			ppa_e[i] = e.params[1] #e#.tvalues[1]

	return ffa_e, ppa_e


def loop_regressions(Subjects, Conditions, window, dset = 'TDSigEI'):
	''' wrap regression models to regress MTD estimates onto thalamus timecourses'''

	#Nuclei = ['AN', 'MD', 'VL', 'Pu']
	df = pd.DataFrame(columns=('Subject', 'Thalamic Nuclei', 'T-P', 'D-P', 'T', 'D', 'P', 'T-D'), dtype=float) 

	for subject in Subjects:

		sdf =  pd.DataFrame(columns=('Subject', 'Thalamic Nuclei', 'T-P', 'D-P', 'T', 'D', 'P', 'T-D'), dtype=float)  

		
		ffa_e={}
		ppa_e={}
		for condition in Conditions:
			ffa_e[condition], ppa_e[condition]= run_reg_model(subject, condition, window)
		i=0	
		for ii, n in enumerate(Nuclei):
			sdf.loc[i, 'Subject'] = subject
			sdf.loc[i, 'Thalamic Nuclei'] = n
			#sdf.loc[i, 'Condition'] = condition
			
			if dset == 'TDSigEI':
				sdf.loc[i, 'T-P'] = ffa_e['FH'][ii] - ffa_e['Fp'][ii] + ppa_e['HF'][ii] - ppa_e['Hp'][ii]#.[ii+1]
				sdf.loc[i, 'D-P'] = ffa_e['HF'][ii] - ffa_e['Fp'][ii] + ppa_e['FH'][ii] - ppa_e['Hp'][ii]
				sdf.loc[i, 'T-D'] = ffa_e['FH'][ii] - ffa_e['HF'][ii] + ppa_e['HF'][ii] - ppa_e['FH'][ii]
				sdf.loc[i, 'T'] = (ffa_e['FH'][ii] + ppa_e['HF'][ii])/2 
				sdf.loc[i, 'D'] = (ffa_e['HF'][ii] + ppa_e['FH'][ii])/2
				sdf.loc[i, 'P'] = (ffa_e['Fp'][ii] + ppa_e['Hp'][ii])/2
				sdf.loc[i, 'PPA-T'] = ppa_e['HF'][ii] 
				sdf.loc[i, 'PPA-D'] = ppa_e['FH'][ii]
				sdf.loc[i, 'PPA-P'] = ppa_e['Hp'][ii]
				sdf.loc[i, 'FFA-T'] = ffa_e['FH'][ii] 
				sdf.loc[i, 'FFA-D'] = ffa_e['HF'][ii]
				sdf.loc[i, 'FFA-P'] = ffa_e['Fp'][ii]
				sdf.loc[i, 'PPA-T-p'] = ppa_e['HF'][ii] -  ppa_e['Hp'][ii]
				sdf.loc[i, 'PPA-D-p'] = ppa_e['FH'][ii] -  ppa_e['Hp'][ii]
				sdf.loc[i, 'PPA-T-D'] = ppa_e['HF'][ii] -  ppa_e['FH'][ii]
				sdf.loc[i, 'FFA-T-p'] = ffa_e['FH'][ii] - ffa_e['Fp'][ii]
				sdf.loc[i, 'FFA-D-p'] = ffa_e['HF'][ii] - ffa_e['Fp'][ii]
				sdf.loc[i, 'FFA-T-D'] = ffa_e['FH'][ii] - ffa_e['HF'][ii]     
				
			if dset == 'TRSE':
				sdf.loc[i, 'T-P'] = ffa_e['FH'][ii] - ffa_e['CAT'][ii] + ppa_e['HF'][ii] - ppa_e['CAT'][ii]#.[ii+1]
				sdf.loc[i, 'D-P'] = ffa_e['HF'][ii] - ffa_e['CAT'][ii] + ppa_e['FH'][ii] - ppa_e['CAT'][ii]
				sdf.loc[i, 'T-D'] = ffa_e['FH'][ii] - ffa_e['HF'][ii] + ppa_e['HF'][ii] - ppa_e['FH'][ii]
				sdf.loc[i, 'T'] = (ffa_e['FH'][ii] + ppa_e['HF'][ii])/2 
				sdf.loc[i, 'D'] = (ffa_e['HF'][ii] + ppa_e['FH'][ii])/2
				sdf.loc[i, 'P'] = ffa_e['CAT'][ii]
				sdf.loc[i, 'PPA-T'] = ppa_e['HF'][ii] 
				sdf.loc[i, 'PPA-D'] = ppa_e['FH'][ii]
				sdf.loc[i, 'PPA-P'] = ppa_e['CAT'][ii]
				sdf.loc[i, 'FFA-T'] = ffa_e['FH'][ii] 
				sdf.loc[i, 'FFA-D'] = ffa_e['HF'][ii]
				sdf.loc[i, 'FFA-P'] = ffa_e['CAT'][ii]
				sdf.loc[i, 'PPA-T-p'] = ppa_e['HF'][ii] -  ppa_e['CAT'][ii]
				sdf.loc[i, 'PPA-D-p'] = ppa_e['FH'][ii] -  ppa_e['CAT'][ii]
				sdf.loc[i, 'PPA-T-D'] = ppa_e['HF'][ii] -  ppa_e['FH'][ii]
				sdf.loc[i, 'FFA-T-p'] = ffa_e['FH'][ii] - ffa_e['CAT'][ii]
				sdf.loc[i, 'FFA-D-p'] = ffa_e['HF'][ii] - ffa_e['CAT'][ii]
				sdf.loc[i, 'FFA-T-D'] = ffa_e['FH'][ii] - ffa_e['HF'][ii]  

			i=i+1
		df = pd.concat([df, sdf])	
	
	return df	


def run_ttests(df):
	'''run t-test on thalamus MTD regressors from loop_regressions function'''

	tdf = pd.DataFrame(columns=('Thalamic Nuclei', 'Condition', 't', 'p'), dtype=float) 

	#Nuclei = ['AN','MD', 'VL', 'Pu']
	i=0
	for condition in ['T-P', 'D-P', 'T-D', 'T', 'D', 'P', 'PPA-T', 'PPA-D', 'PPA-P', 'FFA-T', 'FFA-D', 'FFA-P', 'PPA-T-p', 'PPA-D-p', 'PPA-T-D', 'FFA-T-p', 'FFA-D-p', 'FFA-T-D']:
		for n in Nuclei:			
			tdf.loc[i, 'Thalamic Nuclei'] = n
			tdf.loc[i, 'Condition'] = condition
			tdf.loc[i, 't'] = stats.ttest_1samp(df[df['Thalamic Nuclei']==n][condition].values,0)[0] 
			tdf.loc[i, 'p'] = stats.ttest_1samp(df[df['Thalamic Nuclei']==n][condition].values,0)[1] 
			i=i+1
	return tdf
	


def test_MTD_task_modulation(Subjects, Conditions, window):			
	'''wrapper script to test MTD connectivity strength bewteen VC-FFA/PPA'''
	
	#collect MTD
	MTDdf = pd.DataFrame(columns=('Subject', 'Condition', 'FFA-VC', 'PPA-VC'), dtype=float)
	for i, (subject, condition) in enumerate(itertools.product(Subjects, Conditions)):

		ffa_ts, ppa_ts, vc_ts, tha_ts = load_ts(subject, condition)
		ffa_ts, ppa_ts, vc_ts, tha_ts = censor_ts(ffa_ts, ppa_ts, vc_ts, tha_ts)

		MTDdf.loc[i, 'Subject'] = subject
		MTDdf.loc[i, 'Condition'] = condition
		MTDdf.loc[i, 'FFA-VC'] = np.nanmean(get_MTD(ffa_ts, vc_ts, window))
		MTDdf.loc[i, 'PPA-VC'] = np.nanmean(get_MTD(ppa_ts, vc_ts, window))	
	
	#ttest
		#blah, to be written. 

	return MTDdf		


def test_tha_task_connectivity(Subjects, Conditions, window):			
	'''wrapper script test thal nuclei <-> FFA/PPA/VC connectivity strength'''

	Cdf = pd.DataFrame(columns=('Subject', 'Condition', 'FFA-AN', 'PPA-AN', 'VC-AN', 'FFA-MD', 'PPA-MD', 'VC-MD',  'FFA-VL', 'PPA-VL', 'VC-VL',  'FFA-Pu', 'PPA-Pu', 'VC-Pu'), dtype=float)
	for i, (subject, condition) in enumerate(itertools.product(Subjects, Conditions)):
		ffa_ts, ppa_ts, vc_ts, tha_ts = load_ts(subject, condition)
		ffa_ts, ppa_ts, vc_ts, tha_ts = censor_ts(ffa_ts, ppa_ts, vc_ts, tha_ts)

		Cdf.loc[i, 'Subject'] = subject
		Cdf.loc[i, 'Condition'] = condition

		#AN, MD, VL, Pu cross with V1/FFA/PPS
		ffa_e = fit_linear_model(ffa_ts, tha_ts)
		ppa_e = fit_linear_model(ppa_ts, tha_ts)
		vc_e = fit_linear_model(vc_ts, tha_ts)
		
		Cdf.loc[i, 'FFA-AN'] =  np.nanmean(get_MTD(ffa_ts, tha_ts[:,0], window))  #np.corrcoef(ffa_ts[:], tha_ts[:,0])[0,1]		##ffa_e.params[1]#	 
		Cdf.loc[i, 'PPA-AN'] =  np.nanmean(get_MTD(ppa_ts, tha_ts[:,0], window))  #np.corrcoef(ppa_ts[:], tha_ts[:,0])[0,1]		##ppa_e.params[1]#	 
		Cdf.loc[i, 'VC-AN']  =  np.nanmean(get_MTD(vc_ts, tha_ts[:,0], window))   #np.corrcoef(vc_ts[:], tha_ts[:,0])[0,1]		##vc_e.params[1]# 	
		Cdf.loc[i, 'FFA-MD'] =  np.nanmean(get_MTD(ffa_ts, tha_ts[:,1], window))  #np.corrcoef(ffa_ts[:], tha_ts[:,1])[0,1] 	##ffa_e.params[2]# 	
		Cdf.loc[i, 'PPA-MD'] =  np.nanmean(get_MTD(ppa_ts, tha_ts[:,1], window))  #np.corrcoef(ppa_ts[:], tha_ts[:,1])[0,1] 	##ppa_e.params[2]# 	
		Cdf.loc[i, 'VC-MD']  =  np.nanmean(get_MTD(vc_ts, tha_ts[:,1], window))   #np.corrcoef(vc_ts[:], tha_ts[:,1])[0,1]		##vc_e.params[2]# 	
		Cdf.loc[i, 'FFA-VL'] =  np.nanmean(get_MTD(ffa_ts, tha_ts[:,2], window))  #np.corrcoef(ffa_ts[:], tha_ts[:,2])[0,1] 	##ffa_e.params[3]# 	
		Cdf.loc[i, 'PPA-VL'] =  np.nanmean(get_MTD(ppa_ts, tha_ts[:,2], window))  #np.corrcoef(ppa_ts[:], tha_ts[:,2])[0,1] 	##ppa_e.params[3]# 	
		Cdf.loc[i, 'VC-VL']  =  np.nanmean(get_MTD(vc_ts, tha_ts[:,2], window))   #np.corrcoef(vc_ts[:], tha_ts[:,2])[0,1]		##vc_e.params[3]# 	
		Cdf.loc[i, 'FFA-Pu'] =  np.nanmean(get_MTD(ffa_ts, tha_ts[:,3], window))  #np.corrcoef(ffa_ts[:], tha_ts[:,3])[0,1] 	##ffa_e.params[4]# 	
		Cdf.loc[i, 'PPA-Pu'] =  np.nanmean(get_MTD(ppa_ts, tha_ts[:,3], window))  #np.corrcoef(ppa_ts[:], tha_ts[:,3])[0,1] 	##ppa_e.params[4]# 	
		Cdf.loc[i, 'VC-Pu']  =  np.nanmean(get_MTD(vc_ts, tha_ts[:,3], window))   #np.corrcoef(vc_ts[:], tha_ts[:,3])[0,1] 		##vc_e.params[4]#  	
		#np.nanmean(get_MTD(ffa_ts, tha_ts[:,0], window))
		#np.corrcoef(ffa_ts[:], tha_ts[:,0])[0,1] 
		

	return Cdf


def get_AUC(Subjects, Conditions):
	''' wrapper for getting area under the curve for FIR timecourses
		FIR beta data location: '/home/despoB/kaihwang/bin/TDSigEI/Data/FIRbeta.csv, which was generated by iphone notebook for FIR plotting in TDSigEI'''
	
	firdf = pd.read_csv('Data/FIRbeta.csv')
	aucdf = pd.DataFrame(columns=('Subject', 'Condition'), dtype=float)
	for i, (subject, condition) in enumerate(itertools.product(Subjects, Conditions)):
		
		aucdf.loc[i, 'Subject'] = subject
		aucdf.loc[i, 'Condition'] = condition
		for roi in ['FFA', 'PPA']:  #firdf['ROI'].unique()
			aucdf.loc[i,roi+'_AUC'] = np.trapz(firdf[(firdf['Condition']==condition) & (firdf['Subj']==subject) & (firdf['ROI']==roi)]['Beta'])
	return aucdf
		

def consolidate_EI_df(Subjects, TD_auc_df, TD_thacon_df, TD_MTD_df, behav_df):
	'''wrapper script to gather relevant metrics into a signel dataframe'''

	EI_df = pd.DataFrame(dtype=float)
	
	for i, subject in enumerate(Subjects):

		EI_df.loc[i, 'Subject'] = subject

		EI_df.loc[i, 'AUC_T'] = TD_auc_df[(TD_auc_df['Condition']=='FH') &(TD_auc_df['Subject']==subject)]['FFA_AUC'].values + \
								TD_auc_df[(TD_auc_df['Condition']=='HF') &(TD_auc_df['Subject']==subject)]['PPA_AUC'].values #- \
								#TD_auc_df[(TD_auc_df['Condition']=='Fp') &(TD_auc_df['Subject']==subject)]['FFA_AUC'].values - \
								#TD_auc_df[(TD_auc_df['Condition']=='Hp') &(TD_auc_df['Subject']==subject)]['PPA_AUC'].values 	

		EI_df.loc[i, 'AUC_D'] = TD_auc_df[(TD_auc_df['Condition']=='FH') &(TD_auc_df['Subject']==subject)]['PPA_AUC'].values + \
								TD_auc_df[(TD_auc_df['Condition']=='HF') &(TD_auc_df['Subject']==subject)]['FFA_AUC'].values #- \
								#TD_auc_df[(TD_auc_df['Condition']=='Hp') &(TD_auc_df['Subject']==subject)]['PPA_AUC'].values + \
								#TD_auc_df[(TD_auc_df['Condition']=='Fp') &(TD_auc_df['Subject']==subject)]['FFA_AUC'].values
		
		EI_df.loc[i, 'AUC_0bk'] = TD_auc_df[(TD_auc_df['Condition']=='Hp') &(TD_auc_df['Subject']==subject)]['PPA_AUC'].values + \
								TD_auc_df[(TD_auc_df['Condition']=='Fp') &(TD_auc_df['Subject']==subject)]['FFA_AUC'].values


		EI_df.loc[i, 'MD_T'] = 	TD_thacon_df[(TD_thacon_df['Condition']=='FH') & (TD_thacon_df['Subject']==subject)]['FFA-MD'].values + \
								TD_thacon_df[(TD_thacon_df['Condition']=='HF') & (TD_thacon_df['Subject']==subject)]['PPA-MD'].values #- \
								#TD_thacon_df[(TD_thacon_df['Condition']=='Fp') & (TD_thacon_df['Subject']==subject)]['FFA-MD'].values - \
								#TD_thacon_df[(TD_thacon_df['Condition']=='Hp') & (TD_thacon_df['Subject']==subject)]['PPA-MD'].values
		
		EI_df.loc[i, 'Pu_T'] = 	TD_thacon_df[(TD_thacon_df['Condition']=='FH') & (TD_thacon_df['Subject']==subject)]['FFA-Pu'].values + \
								TD_thacon_df[(TD_thacon_df['Condition']=='HF') & (TD_thacon_df['Subject']==subject)]['PPA-Pu'].values #- \
								#TD_thacon_df[(TD_thacon_df['Condition']=='Fp') & (TD_thacon_df['Subject']==subject)]['FFA-Pu'].values - \
								#TD_thacon_df[(TD_thacon_df['Condition']=='Hp') & (TD_thacon_df['Subject']==subject)]['PPA-Pu'].values

		EI_df.loc[i, 'Pu_D'] = 	TD_thacon_df[(TD_thacon_df['Condition']=='FH') & (TD_thacon_df['Subject']==subject)]['PPA-Pu'].values + \
								TD_thacon_df[(TD_thacon_df['Condition']=='HF') & (TD_thacon_df['Subject']==subject)]['FFA-Pu'].values #- \
								#TD_thacon_df[(TD_thacon_df['Condition']=='Hp') & (TD_thacon_df['Subject']==subject)]['PPA-Pu'].values - \
								#TD_thacon_df[(TD_thacon_df['Condition']=='Fp') & (TD_thacon_df['Subject']==subject)]['FFA-Pu'].values

		EI_df.loc[i, 'MD_D'] = 	TD_thacon_df[(TD_thacon_df['Condition']=='FH') & (TD_thacon_df['Subject']==subject)]['PPA-MD'].values + \
								TD_thacon_df[(TD_thacon_df['Condition']=='HF') & (TD_thacon_df['Subject']==subject)]['FFA-MD'].values #- \
								#TD_thacon_df[(TD_thacon_df['Condition']=='Hp') & (TD_thacon_df['Subject']==subject)]['PPA-MD'].values - \
								#TD_thacon_df[(TD_thacon_df['Condition']=='Fp') & (TD_thacon_df['Subject']==subject)]['FFA-MD'].values

		EI_df.loc[i, 'Pu_0bk'] = 	TD_thacon_df[(TD_thacon_df['Condition']=='Fp') & (TD_thacon_df['Subject']==subject)]['FFA-Pu'].values + \
								TD_thacon_df[(TD_thacon_df['Condition']=='Hp') & (TD_thacon_df['Subject']==subject)]['PPA-Pu'].values #- \
								#TD_thacon_df[(TD_thacon_df['Condition']=='Fp') & (TD_thacon_df['Subject']==subject)]['FFA-Pu'].values - \
								#TD_thacon_df[(TD_thacon_df['Condition']=='Hp') & (TD_thacon_df['Subject']==subject)]['PPA-Pu'].values

		EI_df.loc[i, 'MD_0bk'] = TD_thacon_df[(TD_thacon_df['Condition']=='Hp') & (TD_thacon_df['Subject']==subject)]['PPA-MD'].values + \
								TD_thacon_df[(TD_thacon_df['Condition']=='Fp') & (TD_thacon_df['Subject']==subject)]['FFA-MD'].values #- \
								#TD_thacon_df[(TD_thacon_df['Condition']=='Hp') & (TD_thacon_df['Subject']==subject)]['PPA-Pu'].values - \
								#TD_thacon_df[(TD_thacon_df['Condition']=='Fp') & (TD_thacon_df['Subject']==subject)]['FFA-Pu'].values						
								

		EI_df.loc[i, 'MTD_T'] =	TD_MTD_df[(TD_MTD_df['Condition']=='FH') & (TD_MTD_df['Subject']==subject)]['FFA-VC'].values + \
								TD_MTD_df[(TD_MTD_df['Condition']=='HF') & (TD_MTD_df['Subject']==subject)]['PPA-VC'].values

		EI_df.loc[i, 'MTD_D'] =	TD_MTD_df[(TD_MTD_df['Condition']=='HF') & (TD_MTD_df['Subject']==subject)]['FFA-VC'].values + \
								TD_MTD_df[(TD_MTD_df['Condition']=='FH') & (TD_MTD_df['Subject']==subject)]['PPA-VC'].values	

		EI_df.loc[i, 'MTD_T_D'] =TD_MTD_df[(TD_MTD_df['Condition']=='HF') & (TD_MTD_df['Subject']==subject)]['FFA-VC'].values + \
								TD_MTD_df[(TD_MTD_df['Condition']=='FH') & (TD_MTD_df['Subject']==subject)]['PPA-VC'].values - 	\
								TD_MTD_df[(TD_MTD_df['Condition']=='HF') & (TD_MTD_df['Subject']==subject)]['FFA-VC'].values - \
								TD_MTD_df[(TD_MTD_df['Condition']=='FH') & (TD_MTD_df['Subject']==subject)]['PPA-VC'].values					

		EI_df.loc[i, 'MTD_0bk'] =	TD_MTD_df[(TD_MTD_df['Condition']=='Hp') & (TD_MTD_df['Subject']==subject)]['FFA-VC'].values + \
								TD_MTD_df[(TD_MTD_df['Condition']=='Fp') & (TD_MTD_df['Subject']==subject)]['PPA-VC'].values												

		EI_df.loc[i, 'FH_Dprime'] = behav_df[behav_df['Subject'] == subject]['D_FH'].values
		EI_df.loc[i, 'HF_Dprime'] = behav_df[behav_df['Subject'] == subject]['D_HF'].values

		EI_df.loc[i, 'Dprime'] = (EI_df.loc[i, 'FH_Dprime']  + EI_df.loc[i, 'HF_Dprime'] )/2

	return EI_df


def cal_behav(Subjects):
	'''calculate dprime from subject behavior
	behavior file located at Data/BehavData.csv'''

	bdf = pd.read_csv('Data/BehavData.csv')
	behav_df = pd.DataFrame(dtype=float)
	
	for i, subject in enumerate(Subjects):
		hits = sum(bdf[(bdf['Subj']==('S' + str(subject)))&(bdf['Condition']=='HF')&(bdf['Match']==1)]['Accu'])
		crs = sum(bdf[(bdf['Subj']==('S' + str(subject)))&(bdf['Condition']=='HF')&(bdf['Match']==0)]['Accu'])
		misses = len(bdf[(bdf['Subj']==('S' + str(subject)))&(bdf['Condition']=='HF')&(bdf['Match']==1)]['Accu'])-hits
		fas = len(bdf[(bdf['Subj']==('S' + str(subject)))&(bdf['Condition']=='HF')&(bdf['Match']==0)]['Accu'])-crs
		dHF = dPrime(hits, misses, fas, crs)

		hits = sum(bdf[(bdf['Subj']==('S' + str(subject)))&(bdf['Condition']=='FH')&(bdf['Match']==1)]['Accu'])
		crs = sum(bdf[(bdf['Subj']==('S' + str(subject)))&(bdf['Condition']=='FH')&(bdf['Match']==0)]['Accu'])
		misses = len(bdf[(bdf['Subj']==('S' + str(subject)))&(bdf['Condition']=='FH')&(bdf['Match']==1)]['Accu'])-hits
		fas = len(bdf[(bdf['Subj']==('S' + str(subject)))&(bdf['Condition']=='FH')&(bdf['Match']==0)]['Accu'])-crs
		dFH = dPrime(hits, misses, fas, crs)
		
		behav_df.loc[i, 'Subject'] = subject
		behav_df.loc[i, 'D_FH'] = dFH['d']
		behav_df.loc[i, 'D_HF'] = dHF['d']
	
	return behav_df


def dPrime(hits, misses, fas, crs):
	'''calculate dprime, from'''

	# Floors an ceilings are replaced by half hits and half FA's
	halfHit = 0.5/(hits+misses)
	halfFa = 0.5/(fas+crs)
 
	# Calculate hitrate and avoid d' infinity
	hitRate = hits/(hits+misses)
	if hitRate == 1: hitRate = 1-halfHit
	if hitRate == 0: hitRate = halfHit	
	# Calculate false alarm rate and avoid d' infinity
	faRate = fas/(fas+crs)
	if faRate == 1: faRate = 1-halfFa
	if faRate == 0: faRate = halfFa	
	# Return d', beta, c and Ad'
	out = {}
	out['d'] = Z(hitRate) - Z(faRate)
	out['beta'] = exp((Z(faRate)**2 - Z(hitRate)**2)/2)
	out['c'] = -(Z(hitRate) + Z(faRate))/2
	out['Ad'] = norm.cdf(out['d']/sqrt(2))
	return out


if __name__ == "__main__":

	#TDSigEI
	Subjects = np.loadtxt('/home/despoB/kaihwang/bin/ThaMTD/Data/TDSigEI_subject', dtype=int)
	Conditions =['FH','HF', 'Fp', 'Hp']
	window = 15  #use 15! thats from Mac's simulation
	TD_thaMTD_df = loop_regressions(Subjects, Conditions, window, dset = 'TDSigEI')
	TD_thaMTD_t_df = run_ttests(TD_thaMTD_df)
	TD_MTD_df = test_MTD_task_modulation(Subjects, Conditions, window)
	TD_thacon_df = test_tha_task_connectivity(Subjects, Conditions, window)
	TD_auc_df = get_AUC(Subjects, Conditions)
	
	behav_df = cal_behav(Subjects)
	EI_df = consolidate_EI_df(Subjects, TD_auc_df, TD_thacon_df, TD_MTD_df, behav_df)

	#TRSE
	# Subjects = np.loadtxt(Data+'TRSE_subject', dtype=int)
	# Conditions =['FH','HF', 'CAT']
	# window = 23
	# TR_df = loop_regressions(Subjects, Conditions, window, dset = 'TRSE')
	# TRt_df = run_ttests(TR_df)
	# MTD_TR = test_MTD_task_modulation(Subjects, Conditions, window)
	# TR_Cdf = test_tha_task_connectivity(Subjects, Conditions, window)

 




