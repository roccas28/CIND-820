import os
import pandas as pd
import numpy as np

#read in the dataset
year = '2021'
brfss_2021_dataset = pd.read_csv(r"C:\Users\Roccas\Documents\TMU Data\CIND820 Big Data Analytics Project\LLCP2021XPT\LLCP2021.CSV")

brfss_2021_dataset.shape

#(438700, 303)

pd.set_option('display.max_columns', 500)
brfss_2021_dataset.head()

# DIABETE4 = Question: (Ever told) (you had) diabetes?
# _RFHYPE6 = Question: Adults who have been told they have high blood pressure by a doctor, nurse, or other health professional
# _RFCHOL3 = Adults who have had their cholesterol checked and have been told by a doctor, nurse, or other health professional that it was high
# _CHOLCH3 = Question: Cholesterol check within past five years
# _BMI5 = Body Mass Index (BMI)
# SMOKE100 = Question: Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]
# CVDSTRK3 = Question: (Ever told) (you had) a stroke.
# _MICHD = Question: Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI)
# _TOTINDA = Question: Adults who reported doing physical activity or exercise during the past 30 days other than their regular job
# _FRTLT1A = Question: Consume Fruit 1 or more times per day
# _VEGLT1A = Question: Consume Vegetables 1 or more times per day
# _RFDRHV7 = Question: Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)
# _HLTHPLN = Question: Adults who had some form of health insurance
# MEDCOST1 = Question: Was there a time in the past 12 months when you needed to see a doctor but could not because you could not afford it?
# GENHLTH = Question: Would you say that in general your health is:
# MENTHLTH = Question: Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good?
# PHYSHLTH = Question: Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?
# DIFFWALK = Question: Do you have serious difficulty walking or climbing stairs?
# _SEX = Question: Calculated sex variable
# _AGEG5YR = Question: Fourteen-level age category
# EDUCA =  Question: What is the highest grade or year of school you completed?
# INCOME3 = Question: Is your annual household income from all sources: (If respondent refuses at any income level, code ´Refused.´)
# _RACEPRV = Question: Computed race groups used for internet prevalence tables
# HTM4 = Question: Reported height in meters
# WTKG3 = Question: Reported weight in kilograms
# _BMI5CAT = Question: Four-categories of Body Mass Index (BMI)
# _RFSMOK3 = Question: Adults who are current smokers
# _CURECI1 = Question: Adults who are current e-cigarette users
# DRNKANY5 = Question: Adults who reported having had at least one drink of alcohol in the past 30 days.
# 'HTM4', 'WTKG3',                                          '_BMI5', 
# MSCODE = Metropolitan Status Code
# _FLSHOT7 = Adults aged 65+ who have had a flu shot within the past year
# EMPLOY1 = Are you currently…?
# MARITAL = Are you: (marital status)
# PRIMINSR = What is the current primary source of your health insurance?
# CHCKDNY2 = Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?
# ADDEPEV3 = (Ever told) (you had) a depressive disorder (including depression, major depression, dysthymia, or minor depression)?
# RENTHOM1 = Do you own or rent your home?
# BLIND = Are you blind or do you have serious difficulty seeing, even when wearing glasses?
# DECIDE = Because of a physical, mental, or emotional condition, do you have serious difficulty concentrating, remembering, or making decisions?

brfss_df_selected = brfss_2021_dataset[['DIABETE4',
                                         '_RFHYPE6',  
                                         '_RFCHOL3', '_CHOLCH3', 
                                         'SMOKE100', 
                                         'CVDSTRK3', '_MICHD', 
                                         '_TOTINDA', 
                                         '_FRTLT1A', '_VEGLT1A', 
                                         '_RFDRHV7', 
                                         '_HLTHPLN', 'MEDCOST1', 
                                         'GENHLTH', 'MENTHLTH', 'PHYSHLTH', 'DIFFWALK', 
                                         '_SEX', '_AGEG5YR', 'EDUCA', 'INCOME3', 
                                         '_RACEPRV', '_BMI5CAT', '_RFSMOK3',
                                         '_CURECI1', 'DRNKANY5', 
                                         'MSCODE', '_FLSHOT7', 'EMPLOY1', 'MARITAL', 'PRIMINSR', 'CHCKDNY2',
                                         'ADDEPEV3', 'RENTHOM1', 'BLIND', 'DECIDE']]

#brfss_df_selected.shape

#(438700, 29)

#Drop missing values
brfss_df_selected = brfss_df_selected.dropna()

#brfss_df_selected.shape

# (438700, 29)

# DIABETE4
# 1 = Yes
# 0 = No diabetes or only during pregnancy AND pre-diabetes or borderline diabetes 
# Remove 7 + 9 (Missing)
brfss_df_selected['DIABETE4'] = brfss_df_selected['DIABETE4'].replace({1:1, 2:0, 3:0, 4:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.DIABETE4 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.DIABETE4 != 9]

# _RFHYPE6
# 2 = 1 = Yes
# 1 = 0 = No
brfss_df_selected['_RFHYPE6'] = brfss_df_selected['_RFHYPE6'].replace({1:0, 2:1})
brfss_df_selected = brfss_df_selected[brfss_df_selected._RFHYPE6 != 9]

# _RFCHOL3
# 1 = Yes
# 2 = 0 = No
# Remove 9 (Missing)
brfss_df_selected['_RFCHOL3'] = brfss_df_selected['_RFCHOL3'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._RFCHOL3 != 9]

# _CHOLCH3
# 1 = Yes
# 3 + 2 = 0 = Not checked cholesterol in past 5 years
# Remove 9 (Missing)
brfss_df_selected['_CHOLCH3'] = brfss_df_selected['_CHOLCH3'].replace({3:0,2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._CHOLCH3 != 9]

# _BMI5
# No change, divided by 100
#brfss_df_selected['_BMI5'] = brfss_df_selected['_BMI5'].div(100).round(0)

# SMOKE100
# 1 = Yes
# 2 = 0 = No
# Remove 7 + 9 (Missing)
brfss_df_selected['SMOKE100'] = brfss_df_selected['SMOKE100'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.SMOKE100 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.SMOKE100 != 9]

# CVDSTRK3
# 1 = Yes
# 2 = 0 = No
# Remove 7 + 9 (Missing)
brfss_df_selected['CVDSTRK3'] = brfss_df_selected['CVDSTRK3'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.CVDSTRK3 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.CVDSTRK3 != 9]

# _MICHD
# 1 = Yes
# 2 = 0 = No
brfss_df_selected['_MICHD'] = brfss_df_selected['_MICHD'].replace({2: 0})

# _TOTINDA
# 1 = Yes
# 2 = 0 = No
# Remove 9 (Missing)
brfss_df_selected['_TOTINDA'] = brfss_df_selected['_TOTINDA'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._TOTINDA != 9]

# _FRTLT1A
# 1 = Yes
# 2 = 0 = No
# Remove 9 (Missing)
brfss_df_selected['_FRTLT1A'] = brfss_df_selected['_FRTLT1A'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._FRTLT1A != 9]

# _VEGLT1A
# 1 = Yes
# 2 = 0 = No
# Remove 9 (Missing)
brfss_df_selected['_VEGLT1A'] = brfss_df_selected['_VEGLT1A'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._VEGLT1A != 9]

# _RFDRHV7
# 1 = Yes
# 2 = 0 = No
# Remove 9 (Missing)
brfss_df_selected['_RFDRHV7'] = brfss_df_selected['_RFDRHV7'].replace({1:0, 2:1})
brfss_df_selected = brfss_df_selected[brfss_df_selected._RFDRHV7 != 9]

# _HLTHPLN
# 1 = Yes
# 2 = 0 = No
# Remove 9 (Missing)
brfss_df_selected['_HLTHPLN'] = brfss_df_selected['_HLTHPLN'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._HLTHPLN != 9]

# MEDCOST1
# 1 = Yes
# 2 = 0 = No
# Remove 7 + 9 (Missing)
brfss_df_selected['MEDCOST1'] = brfss_df_selected['MEDCOST1'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.MEDCOST1 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.MEDCOST1 != 9]

# GENHLTH
# 1 = Excellent -> 5 = Poor
# Remove 7 + 9 (Missing)
brfss_df_selected = brfss_df_selected[brfss_df_selected.GENHLTH != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.GENHLTH != 9]

# MENTHLTH
# In Days: 0-30
# 88 = 0 = No
# Remove 77 + 99 (Missing)
brfss_df_selected['MENTHLTH'] = brfss_df_selected['MENTHLTH'].replace({88:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.MENTHLTH != 77]
brfss_df_selected = brfss_df_selected[brfss_df_selected.MENTHLTH != 99]

# PHYSHLTH
# In Days: 0-30
# 88 = 0 = No
# Remove 77 + 99 (Missing)
brfss_df_selected['PHYSHLTH'] = brfss_df_selected['PHYSHLTH'].replace({88:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.PHYSHLTH != 77]
brfss_df_selected = brfss_df_selected[brfss_df_selected.PHYSHLTH != 99]

# DIFFWALK
# 1 = Yes
# 2 = 0 = No
# Remove 7 + 9 (Missing)
brfss_df_selected['DIFFWALK'] = brfss_df_selected['DIFFWALK'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.DIFFWALK != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.DIFFWALK != 9]

# _SEX
# 1 = Male
# 2 = 0 = Female
brfss_df_selected['_SEX'] = brfss_df_selected['_SEX'].replace({2:0})

# _AGEG5YR
# 1 = 30 - 39 -> 6 = 80 and older
# Combined pairs into 1 group to mimic study
# 9 year increments.
# Removed 1 + 2 since below 30
# Remove 14 (Missing)
brfss_df_selected['_AGEG5YR'] = brfss_df_selected['_AGEG5YR'].replace({3:1, 4:1, 5:2, 6:2, 7:3, 8:3, 9:4, 10:4, 11:5, 12:5, 13:6})
brfss_df_selected = brfss_df_selected[brfss_df_selected._AGEG5YR != 1]
brfss_df_selected = brfss_df_selected[brfss_df_selected._AGEG5YR != 2]
brfss_df_selected = brfss_df_selected[brfss_df_selected._AGEG5YR != 14]

# EDUCA
# 1 -> 6 
# 1 = Never attended school -> 6 = 4 years of College Minimum
# Remove 9 (Missing)
brfss_df_selected = brfss_df_selected[brfss_df_selected.EDUCA != 9]

# INCOME3
# 1 = x < $10,000 -> 8 = $75,000 > x
# Remove 77 and 99 (Missing)
brfss_df_selected = brfss_df_selected[brfss_df_selected.INCOME3 != 77]
brfss_df_selected = brfss_df_selected[brfss_df_selected.INCOME3 != 99]

# _RACEPRV
# 1 - 8 = All different races, no changes needed.

# HTM4
# Height reported in meters.
# Remove 7777 and 9999 (Missing)
#brfss_df_selected = brfss_df_selected[brfss_df_selected.HTM4 != 7777]
#brfss_df_selected = brfss_df_selected[brfss_df_selected.HTM4 != 9999]

# WTKG3
# Weight reported in Kilograms
# Remove 7777 and 9999 (Missing)
# No change, divided by 100
#brfss_df_selected['WTKG3'] = brfss_df_selected['WTKG3'].div(100).round(0)
#brfss_df_selected = brfss_df_selected[brfss_df_selected.WTKG3 != 7777]
#brfss_df_selected = brfss_df_selected[brfss_df_selected.WTKG3 != 9999]

# _BMI5CAT
# 4 Categorites of BMI
# Remove 9999 (Missing)
brfss_df_selected = brfss_df_selected[brfss_df_selected._BMI5CAT != 9999]

# _RFSMOK3
# 2 = 1 = Yes
# 1 = 0 = No
# Remove 9 (Missing)
brfss_df_selected['_RFSMOK3'] = brfss_df_selected['_RFSMOK3'].replace({1:0, 2:1})
brfss_df_selected = brfss_df_selected[brfss_df_selected._RFSMOK3 != 9]

# _CURECI1
# 2 = 1 = Yes
# 1 = 0 = No
# Remove 9 (Missing)
brfss_df_selected['_CURECI1'] = brfss_df_selected['_CURECI1'].replace({1:0, 2:1})
brfss_df_selected = brfss_df_selected[brfss_df_selected._CURECI1 != 9]

# DRNKANY5
# 1 = Yes
# 2 = 0 = No
# Remove 7 + 9 (Missing)
brfss_df_selected['DRNKANY5'] = brfss_df_selected['DRNKANY5'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.DRNKANY5 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.DRNKANY5 != 9]

# MSCODE
# 1-5 Metropolitian status, no changes needed

# _FLSHOT7
# 1 = Yes
# 2 = 0 = No
# Remove 9 (Missing)
brfss_df_selected['_FLSHOT7'] = brfss_df_selected['_FLSHOT7'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._FLSHOT7 != 7]

# EMPLOY1
# 1-8 various options
# Remove 9 (Missing)
brfss_df_selected = brfss_df_selected[brfss_df_selected.EMPLOY1 != 9]

# MARITAL
# 1-6 various options
# Remove 9 (Missing)
brfss_df_selected = brfss_df_selected[brfss_df_selected.MARITAL != 9]

# PRIMINSR
# 1-10 + 88, various options
# Remove 77 and 99 (Missing)
brfss_df_selected = brfss_df_selected[brfss_df_selected.PRIMINSR != 77]
brfss_df_selected = brfss_df_selected[brfss_df_selected.PRIMINSR != 99]

# CHCKDNY2
# 1 = Yes
# 2 = 0 = No
# Remove 7 + 9 (Missing)
brfss_df_selected['CHCKDNY2'] = brfss_df_selected['CHCKDNY2'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.CHCKDNY2 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.CHCKDNY2 != 9]

# ADDEPEV3
# 1 = Yes
# 2 = 0 = No
# Remove 7 + 9 (Missing)
brfss_df_selected['ADDEPEV3'] = brfss_df_selected['ADDEPEV3'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.ADDEPEV3 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.ADDEPEV3 != 9]

# RENTHOM1
# 1 = Own
# 2 = 0 = Rent
# 3 = Other
# Remove 7 + 9 (Missing)
brfss_df_selected['RENTHOM1'] = brfss_df_selected['RENTHOM1'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.RENTHOM1 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.RENTHOM1 != 9]

# BLIND
# 1 = Yes
# 2 = 0 = No
# Remove 7 + 9 (Missing)
brfss_df_selected['BLIND'] = brfss_df_selected['BLIND'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.BLIND != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.BLIND != 9]

# DECIDE
# 1 = Yes
# 2 = 0 = No
# Remove 7 + 9 (Missing)
brfss_df_selected['DECIDE'] = brfss_df_selected['DECIDE'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.DECIDE != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.DECIDE != 9]

#brfss_df_selected.shape

#brfss_df_selected.groupby(['DIABETE4']).size()

#Rename the columns to make them more readable
# 'HTM4' : 'Height', 'WTKG3' : 'Weight',                                         '_BMI5':'BMI', 
brfss = brfss_df_selected.rename(columns = {'DIABETE4':'Diabetes', 
                                         '_RFHYPE6':'HighBP',  
                                         '_RFCHOL3':'HighChol', '_CHOLCH3':'CholCheck', 
                                         'SMOKE100':'Smoker', 
                                         'CVDSTRK3':'Stroke', '_MICHD':'HeartDiseaseorAttack', 
                                         '_TOTINDA':'PhysActivity', 
                                         '_FRTLT1A':'Fruits', '_VEGLT1A':"Veggies", 
                                         '_RFDRHV7':'HvyAlcoholConsump', 
                                         '_HLTHPLN':'AnyHealthcare', 'MEDCOST1':'NoDocbcCost', 
                                         'GENHLTH':'GenHlth', 'MENTHLTH':'MentHlth', 'PHYSHLTH':'PhysHlth', 'DIFFWALK':'DiffWalk', 
                                         '_SEX':'Sex', '_AGEG5YR':'Age', 'EDUCA':'Education', 'INCOME3':'Income',
                                         '_RACEPRV' : 'Race', '_BMI5CAT' : 'BMICat', '_RFSMOK3': 'CurrSmoke',
                                         '_CURECI1': 'CurrESmoke', 'DRNKANY5': 'Alcohol30', 'MSCODE' : 'Residence', 
                                         '_FLSHOT7' : 'FLUSHOT', 'EMPLOY1': 'EMPLOYED', 'MARITAL' : 'MARITAL', 'PRIMINSR': 'HLTINSURE', 'CHCKDNY2': 'KIDNEYDIS',
                                         'ADDEPEV3': 'DEPRESSDIS', 'RENTHOM1': 'RENT', 'BLIND': 'BLIND', 'DECIDE': 'DECISION'})

#Drop missing values - again
brfss_df_selected = brfss_df_selected.dropna()

brfss.to_csv(r"C:\Users\Roccas\Documents\TMU Data\CIND820 Big Data Analytics Project\LLCP2021XPT\diabetes_health_indicators_BRFSS2021_v20.csv")