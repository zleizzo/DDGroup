import pandas as pd
import numpy as np
import os

import pyreadstat # loading .sav 

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

def generate_regr_data(data, covariates_continuous, covariates_categorical, objective):
    '''
    Generate data X, y and name_covariates from data
    covariates_continuous: Names of continuous feature column
    covariates_categorical: Names of categorical feature column
    objective: Name of the objective column
    All the rows containing NaN are discarded
    NOTE: Categorcal First.
    '''
    data = data[covariates_continuous + covariates_categorical + [objective]]
    data = data.dropna()

    if covariates_categorical != []:
        X_categorical = data[covariates_categorical].astype('str')
        X_categorical = pd.get_dummies(X_categorical, drop_first=True)
        names_categorical = np.array(X_categorical.columns)
        X_categorical = np.array(X_categorical)
    if covariates_continuous != []:
        X_continuous = np.array(data[covariates_continuous])
        # Normalize
        X_continuous = StandardScaler().fit_transform(X_continuous)
    # Concatenate categorical and continuous covariates  
    if covariates_categorical == []:
        X = X_continuous
        names_covariates = covariates_continuous
    elif covariates_continuous == []:
        X = X_categorical
        names_covariates = names_categorical
    else:
        X = np.concatenate([X_categorical, X_continuous], axis=1)
        names_covariates = np.concatenate([names_categorical, covariates_continuous])

    y = np.array(data[objective])

    return X, y, names_covariates


def load_regr_data(name_data, dir_data='../data/regr'):
    if 'Dutch_drinking' in name_data:
        # Adolescent Heavy Drinking Does Not Affect Maturation of Basic Executive 
        # Functioning: Longitudinal Findings from the TRAILS Study
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0139186#pone-0139186-t005
        # Table 5
        
        data, _ = pyreadstat.read_sav(os.path.join(dir_data, 'Dutch_drinking.sav'))
        
        category = name_data[15:]

        covariates_categorical = ['Imputation_']
        covariates_continuous = ['t1'+category, 'sex', 't1age', 't1ses', 't1mat_alcohol', 't1pat_alcohol',
                                 't1ysr_del', 't3year_cannabis', 't4year_cannabis', 
                                 't3daily_smoking', 't4month_smoking']
        objective = 'Zdelta_' + category

        X, y, names_covariates = generate_regr_data(data, covariates_continuous, 
                                                    covariates_categorical, objective)
        
    elif 'Brazil_health' in name_data:
        # Did the Family Health Strategy have an impact on indicators of hospitalizations 
        # for stroke and heart failure? Longitudinal study in Brazil: 1998-2013
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198428#sec011
        # Table 3 (Heart) Table 4 (Stroke)
        
        data = pd.read_excel(os.path.join(dir_data, 'Brazil_health.xls'))
        
        data['ESFProportion'] = [str(s).replace(',', '.').replace('..', '.') for s in list(data['ESFProportion'])]
        data['ESFProportion'] = data['ESFProportion'].astype('float')

        data = data.loc[data['GDP']!='.']
        
        covariates_categorical = []
        covariates_continuous = ['Year', 'ESFProportion', 'ACSProportion', 'Population', 'GDP', 'DHI Value']
        if 'heart' in name_data:
            objective = 'HeartFailure per 10000'
            data = data.loc[data[objective]!='.']
        elif 'stroke' in name_data:
            objective = 'Stroke per 10000'

        X, y, names_covariates = generate_regr_data(data, covariates_continuous, 
                                                    covariates_categorical, objective)
        
    elif 'Korea_grip' in name_data:
        # Association between grip strength and hand and knee radiographic osteoarthritis in 
        # Korean adults: Data from the Dong-gu study
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0185343#sec017
        # Table 2
        
        data = pd.read_excel(os.path.join(dir_data, 'Korea_grip.xlsx'))

        if 'women' in name_data:
            data = data.loc[data['sex']==2]
        else:
            data = data.loc[data['sex']==1]
            
        covariates_categorical = []
        covariates_continuous = ['total_s_hand', 'JSN_hand', 'OP_hand', 'total_s_knee', 
                                 'OP_knee', 'JSN_knee',
                                 'age', 'BMI', 'smoking_c', 'alcohol_c', 'edu_2']
        objective = 'grip_strength'

        X, y, names_covariates = generate_regr_data(data, covariates_continuous, 
                                                    covariates_categorical, objective)
        
    elif 'China_glucose' in name_data:
        # Fasting plasma glucose and serum uric acid levels in a general Chinese 
        # population with normal glucose tolerance: A U-shaped curve
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180111#sec016
        # Table 2
        
        data = pd.read_excel(os.path.join(dir_data, 'China_glucose.xlsx'))
        
        if 'women1' in name_data:
            data = data.loc[(data['Gender(M1/W2)']==2)&(data['FPG']<4.6)]
        elif 'women2' in name_data:
            data = data.loc[(data['Gender(M1/W2)']==2)&(data['FPG']>=4.6)]
        elif 'men1' in name_data:
            data = data.loc[(data['Gender(M1/W2)']==1)&(data['FPG']<4.7)]
        elif 'men2' in name_data:
            data = data.loc[(data['Gender(M1/W2)']==1)&(data['FPG']>=4.7)]
            
        covariates_categorical = []
        covariates_continuous = ['FPG', 'Age', 'BMI', 'SBP', 'DBP', 'TC', 'TG', 'Drinker(N0/Y1)',
                                 'Smoker(N0/Y1)', 'eGFR', 'INS']
        objective = 'SUA'

        X, y, names_covariates = generate_regr_data(data, covariates_continuous, 
                                                    covariates_categorical, objective)
    
    elif name_data == 'Spain_Hair':
        # Hair cortisol concentrations in a Spanish sample of healthy adults
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0204807
        # Table 3
        # Data: https://figshare.com/s/c27f4958b81b188dab4e
        data, _ = pyreadstat.read_sav(os.path.join(dir_data, 'Spain_Hair_Healthy.sav'))
        
        
        covariates_categorical = []
        covariates_continuous = ['Age', 'Education', 'EmploymentS', 'HairDye', 'PhysicalAct']
        objective = 'Logcortisol'

        X, y, names_covariates = generate_regr_data(data, covariates_continuous, covariates_categorical, objective)
        
    elif name_data == 'China_HIV':
        # Stigma against People Living with HIV/AIDS in China: Does the Route of Infection Matter?
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0151078#sec014
        # Table 2 model b
        
        data = pd.read_stata(os.path.join(dir_data, 'China_HIV.dta'))
        
        # self-esteem: se???
        # Model b
        covariates_categorical = ['route', 'sex', 'ethni', 'relig', 'residence', 'marital', 'income',
                                  'coinf', 'smk', 'alch', 'drug', 'depression']
        covariates_continuous = ['yschool', 'age', 'resi', 'cope', 'ssupp', 'anxitot', 'se']
        # Model c
        # covariates_categorical = ['route', 'sex', 'ethni', 'relig', 'residence', 'marital', 'income', 'coinf']
        # covariates_continuous = ['yschool', 'age']
        objective = 'stigma'

        X, y, names_covariates = generate_regr_data(data, covariates_continuous, 
                                                    covariates_categorical, objective)

    return X, y, names_covariates


def generate_clss_data(data, covariates_continuous, covariates_categorical, objective):
    '''
    Generate data X, y and name_covariates from data
    covariates_continuous: Names of continuous feature column
    covariates_categorical: Names of categorical feature column
    objective: Name of the objective column
    All the rows containing NaN are discarded
    NOTE: Categorcal First.
    '''
    data = data[covariates_continuous + covariates_categorical + [objective]]
    data = data.dropna()

    if covariates_categorical != []:
        X_categorical = data[covariates_categorical].astype('str')
        X_categorical = pd.get_dummies(X_categorical, drop_first=True)
        names_categorical = np.array(X_categorical.columns)
        X_categorical = np.array(X_categorical)
    if covariates_continuous != []:
        X_continuous = np.array(data[covariates_continuous])
        # Normalize
        X_continuous = StandardScaler().fit_transform(X_continuous)
    # Concatenate categorical and continuous covariates  
    if covariates_categorical == []:
        X = X_continuous
        names_covariates = covariates_continuous
    elif covariates_continuous == []:
        X = X_categorical
        names_covariates = names_categorical
    else:
        X = np.concatenate([X_categorical, X_continuous], axis=1)
        names_covariates = np.concatenate([names_categorical, covariates_continuous])

    y = np.array(data[objective])
    y_set = np.sort(list(set(list(y))))
    y_int = np.zeros(y.shape)
    for i, yi in enumerate(y_set):
        y_int[y==yi] = int(i)
    y = y_int
    # y = np.array(data[objective], dtype=np.int)

    return X, y, names_covariates


def load_clss_data(name_data, dir_data='../data/clss'):
    '''
    Load dataset
    return X, y, names_covariates
    '''
    if name_data == 'India_distress':
        # Disease-specific out-of-pocket and catastrophic health expenditure on hospitalization 
        # in India: Do Indian households face distress health financing?
        # Table 7
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196106#sec023
        data = pd.read_stata(os.path.join(dir_data, 'India_distress_finance.dta'))

        covariates_continuous = []
        covariates_categorical = ['disease_class500', 'age3', 'sex', 'edu', 'mpcet', 'sector', 'public']
        objective = 'distress_financ23_source1'

        X, y, names_covariates = generate_clss_data(data, covariates_continuous, 
                                               covariates_categorical, objective)
        
    elif name_data == 'USA_literacy':
        # Examining Associations between Self-Rated Health and Proficiency in Literacy and Numeracy 
        # among Immigrants and U.S.-Born Adults: Evidence from the Program for the International 
        # Assessment of Adult Competencies (PIAAC)
        # Table 2
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130257
        # Data: https://nces.ed.gov/pubsearch/pubsinfo.asp?pubid=2014045REV 
        ##################################################################
        ####### Categorical
        # 'J_Q04a': Immigrant Status - Background born in country: 1 US-Born 2 Foreign-Born 
        # 'AGEG10LFS': Age in 10 year bands (derived)
        # 'GENDER_R': Person resolved gender from BQ and QC check (derived)
        # 'RACETHN_5CAT': Background - race/ethnicity (derived, 5 categories)
        ## 'J_Q02a' (Need processing. 1 or else): Label: Background - Living with spouse or partner
        ## 'J_Q03d1_C' (Need processing. 4 or else): Label: Background - Age of the youngest child (categorised, 4 categories)
        # 'REGION_US': Geographical region - Respondent (US Census regions)
        ## 'I_Q08USX3' (Need Processing. else or 1):  About yourself - Health - Diagnosed learning disabled
        ## 'I_Q010bUSX1' (Need Processing. else or 1): About yourself - Health - Have medical insurance
        ## 'I_Q10bUSX3a' (Need processing. else or 1): About yourself - Health - Flu shot in past year
        ## 'B_Q01a_C' ('B_Q02bUS_C') (Need processing. only 1-6, discard letter D N V):  Education - current qualification - level (6 categories) 
        ## 'C_Q07' (Need processing. [1, 2], [3], [4, 5], [6], [7], [9, 10]:  Current status/work history - Subjective status
        # 'D_Q18a_T':  Annual net income before taxes and deductions 
        ## 'J_Q06bUS' (Need processing. 1, 2, 3 and exclude D N R): Background - Mother/female guardian - Highest level of education
        ## 'J_Q07bUS' (Need processing. 1, 2, 3 and exclude D N R):  Background - Father/male guardian - Highest level of education 
        ####### Continuous
        # 'PVLIT1': Literacy scale score - Plausible value 1
        # 'PVNUM1': Numeracy scale score - Plausible value 1J_Q04A
        # 'J_Q01_T1':  Number living in household (from 1 to 7) 
        # Missing: English proficiency level
        ####### Objective
        # I_Q08: self-reported Heath Status
        ##################################################################
        
        from sas7bdat import SAS7BDAT
        with SAS7BDAT(os.path.join(dir_data, 'USA_literacy_numeracy.sas7bdat'), skip_header=False) as reader:
            data = reader.to_data_frame()

        covariates_continuous = ['J_Q01_T1', 'PVLIT1', 'PVNUM1']
        covariates_categorical = ['J_Q04a', 'AGEG10LFS', 'GENDER_R', 'RACETHN_5CAT', 
                                  'J_Q02a', 'J_Q03d1_C', 'REGION_US', 'I_Q08USX3', 'I_Q010bUSX1',
                                  'I_Q10bUSX3a', 'B_Q01a_C', 'C_Q07', 'D_Q18a_T', 'J_Q06bUS', 'J_Q07bUS' ]
        objective = 'I_Q08'

        # NaN as a new category
        data.loc[data['D_Q18a_T']!=data['D_Q18a_T'], 'D_Q18a_T'] = 0
        # Processing classes
        data.loc[data['J_Q02a']!=1, 'J_Q02a'] = 0
        data.loc[data['J_Q03d1_C']!=4, 'J_Q03d1_C'] = 0
        data.loc[data['I_Q08USX3']!=1, 'I_Q08USX3'] = 0
        data.loc[data['I_Q010bUSX1']!=1, 'I_Q010bUSX1'] = 0
        data.loc[data['I_Q10bUSX3a']!=1, 'I_Q10bUSX3a'] = 0
            
        X, y, names_covariates = generate_clss_data(data, covariates_continuous, 
                                               covariates_categorical, objective)
        
    elif name_data == 'USA_kidney':
        # Hyponatremia and the risk of kidney stones: A matched case-control study in a large U.S. health system
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0203942#sec011
        # Table 2
        covariates_categorical = ['Hyponatremia_Prior', 'Hyponatremia_Recent', 'Hyponatremia_Recent_Chronic',
                                  'Calcium', 'Estrogen', 'VitaminD', 'VitaminB6', 'VitaminC', 'Thiazide',
                                  'Furosemide', 'Topiramate', 
                                  'htn', 'Obese', 'Dyslipid', 'Gout', 'Reg_Enteritis', 'Ulcer_Colitis', 
                                  'Celiac', 'Osteoporosis', 'Hyperparathyroid', 'Hypercalcemia', 'Acidosis', 
                                  'Bariatric', 'Sarcoid', 'Liver_Cirrhosis', 'Heart_Failure',
                                  'Tobacco', 'Alcohol']
        objective = 'Case'
        covariates_continuous = []

        data = pd.read_stata(os.path.join(dir_data, 'USA_kidney.dta'))
        X, y, names_covariates = generate_clss_data(data, covariates_continuous, 
                                               covariates_categorical, objective)
        
    elif name_data == 'India_HIV':
        # Exposure to Pornographic Videos and Its Effect on HIV-Related Sexual Risk 
        # Behaviours among Male Migrant Workers in Southern India
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0113599
        # Data: https://dataverse.harvard.edu/dataset.xhtml?persistentId=hdl:1902.1/18737
        
        ##################################################################
        ####### Categorical
        ## 'q101' (Need Processing. <25, 25-29, >=30): age
        ## 'q102' (Need Processing. 0, <=6, 7-9, >=10): Education
        ## 'q104' (Need Processing. [2], [3, 4], [1, 5, 6]): Marital status
        ## 'q304' (Need Processing. <=3000, 3001-4000, >=4001): Monthly income (in INR)
        ## 'q218' (Need Processing. <5, >=5): Number of individuals staying with
        ## 'q523a4' (Need Processing. <36, <=36<72, >=72): Duration of migration (in Years)
        ## 'q307' (Need Processing Only 1, 2): Able to save money
        ## 'q310' (Need Processing. [3], [1, 2]): Stay away from home overnight
        ## 'q402' (Need Processing. Only 1 2 3 4): Frequency of return of native place
        ####### Continuous
        ####### Objective
        ## 'q520d' (Need Processing. Only 1 2): Whether viewed pornographic videos in the one month prior to the survey
        ##################################################################

        def process_data(data, col, inds):
            for i, ind in enumerate(inds):
                data.loc[ind, col] = i
                if i == 0:
                    ind_all = ind
                else:
                    ind_all = ind_all | ind
            data = data.loc[ind_all].copy()
            return data


        data = pd.read_table(os.path.join(dir_data, 'India_HIV.tab'))

        cols = []
        inds = []

        col = 'q101'
        cols.append(col)
        inds.append([data[col]<25, (data[col]>=25)&(data[col]<30), data[col]>=30])
        col = 'q102'
        cols.append(col)
        inds.append([data[col]==0, (data[col]>0)&(data[col]<=6), (data[col]>=7)&(data[col]<=9), data[col]>=10])
        col = 'q104'
        cols.append(col)
        inds.append([data[col]==2, (data[col]>=3)&(data[col]<=4), (data[col]==1)|(data[col]==5)|(data[col]==6)])
        col = 'q304'
        cols.append(col)
        inds.append([data[col]<=3000, (data[col]>=3001)&(data[col]<=4000), data[col]>=4001])
        col = 'q218'
        cols.append(col)
        inds.append([data[col]<5, data[col]>=5])
        col = 'q523a4'
        cols.append(col)
        inds.append([data[col]<36, (data[col]>=36)&(data[col]<72), data[col]>=72])
        col = 'q307'
        cols.append(col)
        inds.append([data[col]==1, data[col]==2])
        col = 'q310'
        cols.append(col)
        inds.append([data[col]==3, (data[col]==1)|(data[col]==2)])
        col = 'q402'
        cols.append(col)
        inds.append([data[col]==1, data[col]==2, data[col]==3, data[col]==4])
        col = 'q520d'
        cols.append(col)
        inds.append([data[col]==1, data[col]==2])

        for col, ind in zip(cols, inds):
            data = process_data(data.copy(), col, ind)
            
        covariates_categorical = ['q101', 'q102', 'q104', 'q304', 'q218', 'q523a4', 'q307',
                                  'q310', 'q402']
        covariates_continuous = []
        objective = 'q520d'

        X, y, names_covariates = generate_clss_data(data, covariates_continuous, 
                                               covariates_categorical, objective)
        
        
    elif 'Zambia_perception' in name_data:
        # Women's Perceptions and Misperceptions of Male Circumcision: A Mixed Methods Study in Zambia
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0149517#sec021
        # Table 3. 
        # 'mc1' works. 'mc2' highly imbalanced: 31:567
        
        data = pd.read_stata(os.path.join(dir_data, 'Zambia_women_perception.dta'))
        
        ## 'religion1_grp' (Need processing (not yet). 'Catholic', All others ('Other Christian', 'Other/none'))
        covariates_categorical = ['age1_grp', 'province1_grp', 'religion1_grp', 'tribe1_grp',
                                  'highest_level1_grp', 'ever_married1', 'current_relationship1',
                                  'hiv_test1']
        covariates_continuous = ['num_assets1']

        if 'mc1' in name_data:
            objective = 'heard_mc1'
        elif 'mc2' in name_data:
            objective = 'heard_mc2'
            covariates_categorical.append('circumcision1')

        X, y, names_covariates = generate_clss_data(data, covariates_continuous, 
                                               covariates_categorical, objective)

        
    elif name_data == 'Angola_maternal':
        # Determinants of maternal health care and birth outcome in the Dande Health and 
        # Demographic Surveillance System area, Angola
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0221280
        # Table 2
        
        data, _ = pyreadstat.read_sav(os.path.join(dir_data, 'Angola_maternal_health.sav'))
        
        covariates_categorical = ['ANC', 'Urban_rural', 'Place_delivery', 'DistanceHF']
        covariates_continuous = ['School_years', 'Age_years']
        objective = 'Pregnancy_outcome' # 0: alive. 1: stilbirth. 2: abortion.

        X, y, names_covariates = generate_clss_data(data, covariates_continuous, 
                                               covariates_categorical, objective)
        
        
    elif name_data == 'Congo_fever':
        # Typhoid fever outbreak in the Democratic Republic of Congo: Case control and ecological study
        # https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0006795#sec016
        # Table 3
        
        ##################
        ###### Categorical
        # 'partass': Plate sharing. 1 never, 2 sometimes, 3 never
        ## 'occup' (Need Processing. 1, 2, 3, 4, [5, 6]) 1 casual labour; 2 labour with regular job; 3 own business; 4 farmer; 5 Other; 6 no reply
        # 'eautrait': Tap water is ever used - do they treat their water (1=y, 2=n)
        # 'lavdef': Wash hands after defecating. 1 never, 2 sometimes, 3 always
        # 'sortrea': Water source chosen because it is protected. Source protected = 1
        ## 'occup.1' (Need Processing. Only 1, 2) Visible urine/faeces at latrine. 1 yes, 2 no
        ##################     
        
        data = pd.read_excel(os.path.join(dir_data, 'Congo_fever.xlsx'))
        data = data.iloc[1:, :]
        data.loc[data['occup']==6, 'occup'] = 5
        data = data.loc[(data['occup.1']==1)|(data['occup.1']==2)].copy()

        covariates_categorical = ['partass', 'occup', 'eautrait', 'lavdef', 'sortrea', 'occup.1']
        covariates_continuous = []
        objective = 'status' # cas=1, cont=0

        X, y, names_covariates = generate_clss_data(data, covariates_continuous, 
                                               covariates_categorical, objective)
        
    
    elif 'Sjogren' in name_data:
        # Association between a history of mycobacterial infection and the risk of newly diagnosed 
        # Sjogren's syndrome: A nationwide, population-based case-control study
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0176549#sec022
        # Table 2
        
        data = pd.read_excel(os.path.join(dir_data, 'Sjogren.xlsx'))
        
        data['CCI_Group'] = 0
        data.loc[data['CCI']>=1, 'CCI_Group'] = 1

        if 'ModelA' in name_data:
            covariates_categorical = ['Myco_TBorNTM(year)', 'CCI_Group', 'Bronchiectasis']
        elif 'ModelB' in name_data:
            covariates_categorical = ['NTM', 'TB', 'CCI_Group', 'Bronchiectasis']
        covariates_continuous = []
        objective = 'SS'

        X, y, names_covariates = generate_clss_data(data, covariates_continuous, 
                                               covariates_categorical, objective)
        
        
    elif name_data == 'USA_obesity':
        # Assessment of dietary patterns, physical activity and obesity from a national 
        # survey: Rural-urban health disparities in older adult
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0208268
        # Table 4
        
        data, _ = pyreadstat.read_sav(os.path.join(dir_data, 'USA_obesity.sav'))
        
        # How to find features: [name for name in data.columns if np.sum([a.islower() for a in name])>0]
        covariates_categorical = ['No_PhysAct', 'VEG_Low', 'FRUIT_Low', 'SEX', 'White', 'College_Grad', 'FairPoorHealth']
        covariates_continuous = ['AGE', 'PCIncome']
        objective = 'Obese'

        X, y, names_covariates = generate_clss_data(data, covariates_continuous, 
                                               covariates_categorical, objective)
        
    elif name_data == 'SouthAmerica_tuberculosis':
        # Predictors of noncompliance to pulmonary tuberculosis treatment: An insight from South America
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0202593#sec013
        # Table 2
        
        data = pd.read_excel(os.path.join(dir_data, 'SouthAmerica_tuberculosis.xlsx'))
        
        data.loc[data['case/control']=='controls', 'case/control'] = 0
        data.loc[data['case/control']=='cases', 'case/control'] = 1

        # How to find features: [name for name in data.columns if np.sum([a.islower() for a in name])>0]
        covariates_categorical = ['sex', 'age in years (cat)', 'race (skin color)', 
                                  'education (years)', 'income', 'homeless', 'smoking status', 
                                  'drug use', 'HIV infection', 'diabetes', 'treatment category']
        covariates_continuous = []
        objective = 'case/control'

        X, y, names_covariates = generate_clss_data(data, covariates_continuous, 
                                               covariates_categorical, objective)
        
    elif name_data == 'Infection':
        # Antibiotic prophylaxis for surgical site infections as a risk factor for infection with Clostridium difficile
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0179117
        # Table 4
        
        data = pd.read_csv(os.path.join(dir_data, 'Infection.txt'), sep="\t")
        
        # Listed In Table 4: 'recomm', 'allcomorb', 'antibeforehosp', 'surgery'
        covariates_categorical = ['recomm', 'allcomorb', 'antibeforehosp', 'surgery',
                                  'sex', 'comorbbin', 'sevcomorbbin']
        covariates_continuous = ['age', 'staylength']
        objective = 'status'

        X, y, names_covariates = generate_clss_data(data, covariates_continuous, 
                                               covariates_categorical, objective)
        
    elif name_data == 'Maternal_deaths':
        # Using Observational Data to Estimate the Effect of Hand Washing and Clean Delivery Kit Use 
        # by Birth Attendants on Maternal Deaths after Home Deliveries in Rural Bangladesh, India and Nepal
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136152#sec021
        
        data = pd.read_excel(os.path.join(dir_data, 'Maternal_deaths.xls'))
        # Listed in Table 4: 'cdk', 'handwash'
        covariates_categorical = ['cdk', 'handwash', 
                                  'educ', 'parity', 'assetCAT', 'country']
        covariates_continuous = ['mum_age', 'anc_num']
        objective = 'mumdied'

        X, y, names_covariates = generate_clss_data(data, covariates_continuous, 
                                               covariates_categorical, objective)
        
    elif name_data == 'Qatar_antibodies':
        # Dengue and chikungunya seroprevalence among Qatari nationals and immigrants residing in Qatar
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0211574#sec013
        # Table 4
       
        data = pd.read_excel(os.path.join(dir_data, 'Qatar_antibodies.xlsx'))
        
        data['Age_group'] = list(data['Age']).copy()
        data.loc[data['Age']<=24, 'Age_group'] = 0
        data.loc[(data['Age']>=25)&(data['Age']<=29), 'Age_group'] = 1
        data.loc[(data['Age']>=30)&(data['Age']<=34), 'Age_group'] = 2
        data.loc[(data['Age']>=35)&(data['Age']<=39), 'Age_group'] = 3
        data.loc[(data['Age']>=40)&(data['Age']<=44), 'Age_group'] = 4
        data.loc[(data['Age']>=45)&(data['Age']<=49), 'Age_group'] = 5
        data.loc[data['Age']>=50, 'Age_group'] = 6

        data['Region'] = -1
        # 'Asia'
        data.loc[(data['Nationality']== 'IND')|(data['Nationality']== 'PAK')|(data['Nationality']== 'PAk')|(data['Nationality']== 'PHI')|(data['Nationality']== 'PHIL')|(data['Nationality']== 'PHL'), 'Region'] = 0
        # 'Middle East'
        data.loc[(data['Nationality']== 'IRAN')|(data['Nationality']== 'IRM')|(data['Nationality']== 'IRN')|(data['Nationality']== 'JOR')|(data['Nationality']== 'LEB')|(data['Nationality']== 'PAL')|(data['Nationality']== 'PaL')|(data['Nationality']== 'QAT')|(data['Nationality']== 'SYR')|(data['Nationality']== 'YAM')|(data['Nationality']== 'YEM'), 'Region'] = 1
        # 'North Africa'
        data.loc[(data['Nationality']== 'EGY')|(data['Nationality']== 'SUD'), 'Region'] = 2
        # 'Qatar'
        data.loc[(data['Nationality']== 'QAT'), 'Region'] = 3

        data = data.loc[(data['Dengue Result']==0)|(data['Dengue Result']==1)]
        data = data.loc[(data['Chikungunia Result']==0)|(data['Chikungunia Result']==1)] 
        
        covariates_categorical = ['Region', 'Age_group', 'Chikungunia Result']
        covariates_continuous = []
        objective = 'Dengue Result'

        X, y, names_covariates = generate_clss_data(data, covariates_continuous, 
                                               covariates_categorical, objective)

    else:
        print('Dataset name does not exists!')
        
    return X, y, names_covariates
