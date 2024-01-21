# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 20:17:38 2024

@author: asus
"""

import ToolBox as TB
import os
from Semi_beta_variation import *



data_dir = os.getcwd()

###########################################################################
## 1. Create basic dir to store the all kind of data
###########################################################################
# Create folders for sotring the intraday semi-beta variation estimating results
folders_dict = {str(year):{type_1:{type_:'' for type_ in ['15','20','25','30']} for type_1 in ['4000','300'] } for year in [5]}
TB.create_folders(data_dir + '\\data_sample\\Intraday_betas', folders_dict)

# Create folders for sotring the dispersion of intraday semi-beta variation estimating results
folders_dict = {measure:{str(year):{type_1:{type_:'' for type_ in ['15','20','25','30']} for type_1 in ['4000','300'] } for year in [5]} for measure in ['BQ100','AutoCorr','Square']}
TB.create_folders(data_dir + '\\data_sample\\Other_measure', folders_dict)

# Create folders to store stocks' continue and discontinue beta
folders_dict = {str(year):{type_1:'' for type_1 in ['300']} for year in [5]}
TB.create_folders(data_dir + '\\data_sample\\ConDisconBeta', folders_dict)

# Create folders to store stocks' semi-betas
folders_dict = {str(year):{type_1:'' for type_1 in ['300','4000']} for year in [5]}
TB.create_folders(data_dir + '\\data_sample\\SemiBeta\\Betas', folders_dict)

# Create folders to store stocks' relative sign jump that constructed by rv
folders_dict = {str(year):'' for year in [5]}
TB.create_folders(data_dir + '\\data_sample\\RSJ_RV', folders_dict)

# Create folders to store stocks' IVOL
folders_dict = {str(year):'' for year in ['CH3','CH4','FF3','FF5']}
TB.create_folders(data_dir + '\\data_sample\\IVOL', folders_dict)

# Create folders to store the error stock code while calculating the intraday semi-beta variation
folders_dict = {'':''}
TB.create_folders(data_dir + '\\data_sample\\error', folders_dict)

# Create folders to store the error stock code while calculating the intraday semi-beta variation
folders_dict = {'':''}
TB.create_folders(data_dir + '\\data_sample\\error', folders_dict)

# Create folders to store the single sort results
folders_dict = {weight:{index:{str(est):'' for est in [15,20,25,30]} for index in ['300','4000']} for weight in ['vw','ew']}
TB.create_folders(data_dir + '\\data_sample\\Sorted_res\\Ssort', folders_dict)

# Create folders to store the double sort results
folders_dict = {weight:{index:{str(est):{reverse:'' for reverse in ['True','False']} for est in [15,20,25,30]} for index in ['300','4000']} for weight in ['vw','ew']}
TB.create_folders(data_dir + '\\data_sample\\Sorted_res\\Dsort', folders_dict)

# Create folders to store the fama-macbeth regression results
folders_dict = {str(min_):{index:{str(est):'' for est in [15,20,30,35]} for index in ['300','4000']} for min_ in [5]}
TB.create_folders(data_dir + '\\data_sample\\FMreg_res', folders_dict)

# Create folders to store the betting beta results
folders_dict = {'':''}
TB.create_folders(data_dir + '\\data_sample\\Strategy', folders_dict)


###########################################################################
## 2. Create all kind of basic data
###########################################################################

# Create the continue and discontinue beta
Muti_Exec_ConDisconBeta(idxcd=300, min_=5,mult=False)
Mrg_ConDisconBeta(index=300, min_=5)

# Create the semi-betas
Muti_Exec_SemiBeta(mult=False)
Mult_Mrg_SemiBeta([300, 4000], [5], mult=False)   

# Create the RSJ
Muti_Exec_RSJ_RV(mult=False)
Mrg_RSJ_RV(5)

# Create IVOL that based on CH3 factor model
Mult_Cpt_IVOL(mult=False)
Mrg_IVOL()

# Calculate intraday semi-beta variation and its dispersion
Cpt_All_Stock_DS([300,4000], [15,20,25,30], n=48, min_=5, mult=False)

# Merge different stock's mean of intraday semi-beta varitions as well as its three different
Mult_Mrg_Intraday_Beta_res(mult=False)
Mult_Mrg_Beta_measure(['Square','BQ100','AutoCorr'], [300,4000],[5],[15,20,25,30],mult=False)


##########################################################################
# 3. Create empirical results of sotring, regression and streatgy
##########################################################################

# Mult generate single sort results
Muti_exec_Ssort(['vw','ew'],[300,4000], ['15','20','25','30'], ['Square','Beta_abs_intra','BQ100','AutoCorr'], min_=5, mult=False,groupsNum=3)

# Mult generate double sort results    
Muti_Exec_TD_Dsort(['ew','vw'],['Square','Beta_abs_intra','BQ100','AutoCorr'],
                    [300,4000],['BM','ME','REV','IVOL','ILLIQ','MAX','MIN','CSK','CKT','RSJ','beta','Beta_neg','disconBeta'],
                    [15,20,25,30], min_=5, freq='W', mult=False, reverse=False, groupsNum1=3, groupsNum2=3)

# Merge double sorts results
control_list = ['BM','ME','REV','IVOL','ILLIQ','MAX','MIN','CSK','CKT','RSJ','beta','Beta_neg','disconBeta']
for key_tag in ['Square','Beta_abs_intra','BQ100','AutoCorr']:
    for weight_type in ['ew','vw']:            
        Mrg_Dsort_res_('W', 5, control_list,[15,20,25,30], 300, key_tag, weight_type, reverse=False)
        Mrg_Dsort_res_('W', 5, control_list,[15,20,25,30], 4000, key_tag, weight_type, reverse=False)


# # Mult generate fama-macbeth regression results    
# Muti_exec_FMR(5, ['W'], [15,20,25,30], index_lst=[300,4000], mult=False)    

# Mult generate beting beta strategy results
Muti_Bet_on_BetaDispersion(5, ['W'], 300, [15,20,25,30],mult=False, groupsNum=3)    
