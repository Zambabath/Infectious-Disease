import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from bs4 import BeautifulSoup
from ftfy import fix_text
from datetime import datetime

df = pd.read_csv('/Users/prim/Desktop/MSc/SCC450/project/data/ictrp_data.csv')

#EDA
## check datatype and missing value
print(df.info())
## check duplicate column
print(df.duplicated().sum())

# Data cleaning
df_cleaned = df.copy()

## remove white space
for c_strip in df_cleaned.columns:
    df_cleaned[c_strip] = df_cleaned[c_strip].apply(lambda x: str(x).strip())


## fill missing value with UNKNOWN
def format_missing_value(data):
    data = data.upper()
    if data in ['N/A','NA','NAN','NONE','MISSING']:
        return 'UNKNOWN'
    else:
        return data

cleaning_list = ['phase','randomization','secondary_sponsor','primary_outcome','secondary_outcome',
                 'primary_purpose','centre','countries','intervention','intervention_model',
                 'masking','endpoint_classification']

for col in cleaning_list:
    df_cleaned[col] = df_cleaned[col].fillna("UNKNOWN")
    df_cleaned[col] = df_cleaned[col].apply(format_missing_value)


## change date format to YYYY-MM-DD
print("\n---------------\n")
print('Date before')
print(df_cleaned['date_enrollment'].head(5))
def add_date(data):
    if (len(str(data))==7):
        return (data + '-01')
    else:
        return data 

def check_date(data):
    data = str(data)
    if data[:4].isdigit():
        return datetime.strptime(data, "%Y-%m-%d").date()
    elif data[:2].isdigit():
        return datetime.strptime(data, "%d-%m-%Y").date()
    else :
        return pd.NaT

date_cols = ['date_registration', 'date_enrollment', 'results_date_completed','results_date_posted']

for column in date_cols:
    df_cleaned[column] = df_cleaned[column].astype(str).str.replace('/','-')
    df_cleaned[column] = df_cleaned[column].apply(add_date)
    df_cleaned[column] = df_cleaned[column].apply(check_date)
    df_cleaned[column] = pd.to_datetime(df_cleaned[column], dayfirst=True, errors='coerce').dt.date

print('Date after')
print(df_cleaned['date_enrollment'].head(5))


## separate primary condition to separated columns
disease_list = ['Chagas Disease','Schistosomiasis','Soil-Transmitted Helminthiases','Visceral Leishmaniasis']
df_cleaned['standardised_condition'] = df_cleaned['standardised_condition'].apply(lambda x: x.split('|'))

for col in disease_list:
    df_cleaned[col]=df_cleaned['standardised_condition'].apply(lambda x: 1 if col in x else 0)


##convert age ex. 12Y to 12
print("\n---------------\n")
print('Age before')
print(df_cleaned['inclusion_age_max'].head(5))

age_cols = ['inclusion_age_min','inclusion_age_max']

def convert_age(age):
    age = age.upper()
    if age in ['NO LIMIT','NA','NOT SPECIFIC','NONE','NAN','N/A','NOT SPECIFIED','NOT APPLICABLE']:
        return pd.NA
    elif age.endswith('Y') and (',' not in str(age)):
        age = age[:-1]
        if (age.startswith('<')) or (age.startswith('>')):
            return age[1:]
        else:
            return age
    elif age.endswith('M') and (','  not in str(age)):
        age = int(age[:-1])/12
        return age
    elif age.endswith('D') and (',' not in str(age)):
        age = int(age[:-1])/365
        return age
    elif ((age.startswith('<')) or (age.startswith('>'))) and (',' not in str(age)):
        return age[1:]
    else:
        return age
    
for col in age_cols:
    df_cleaned[col] = df_cleaned[col].fillna('UNKNOWN')
    df_cleaned[col] = df_cleaned[col].apply(convert_age)

def max_age(age):
    if pd.isna(age):
        return pd.NA
    elif ',' in str(age):
        age = str(age).split(',')
        int_age = []
        for i in range(len(age)):
            if age[i].endswith('Y'):
                age[i] = age[i][:-1]
            elif age[i] == 'NO LIMIT':
                return pd.NA
            int_age.append(int(age[i]))
        return max(int_age)
    else:
        return round(float(age),3)
    
def min_age(age):
    if pd.isna(age):
        return pd.NA
    elif ',' in str(age):
        age = str(age).split(',')
        int_age = []
        for i in range(len(age)):
            if age[i].endswith('Y'):
                age[i] = age[i][:-1]
            elif age[i] == 'NO LIMIT':
                return pd.NA
            int_age.append(int(age[i]))
        return min(int_age)
    else:
        return round(float(age),3)

df_cleaned['inclusion_age_max'] = df_cleaned['inclusion_age_max'].apply(max_age)
df_cleaned['inclusion_age_min'] = df_cleaned['inclusion_age_min'].apply(min_age)

## apply median on each disease for missing value
median_max_disease = {}
median_min_disease = {}

median_pair_max = {}
median_pair_min = {}
## for one disease
for col in disease_list:         
    disease_max = df_cleaned.groupby([col])['inclusion_age_max'].median()
    disease_min = df_cleaned.groupby([col])['inclusion_age_min'].median()
    median_max_disease[col] = disease_max[1]
    median_min_disease[col] = disease_min[1]
    print(median_max_disease)
## for two disease
for dis1 in disease_list:
    for dis2 in disease_list:
        if dis1 != dis2:
            median_pair_max[(dis1, dis2)] = (median_max_disease[dis1]+median_max_disease[dis2])/2
            median_pair_min[(dis1, dis2)] = (median_min_disease[dis1]+median_min_disease[dis2])/2
            

    
def apply_median_age(row,disease,name,one_dis,two_dis):
    remaining = [d for d in disease_list if d != disease]
    if row[disease]==1 and pd.isna(row[name]):
        for col in remaining:
            if row[col] == 1:
                return two_dis[(disease, col)]
            
        return one_dis[disease]
    elif row[disease]==0 and pd.isna(row[name]):
        return pd.NA
    elif not pd.isna(row[name]):
        return row[name]

for col in disease_list:
    df_cleaned['inclusion_age_max'] = df_cleaned.apply(apply_median_age, axis=1, 
                                                                   disease=col, 
                                                                   name='inclusion_age_max',
                                                                   one_dis=median_max_disease,
                                                                   two_dis=median_pair_max)
    df_cleaned['inclusion_age_min'] = df_cleaned.apply(apply_median_age, axis=1, 
                                                                   disease=col, 
                                                                   name='inclusion_age_min',
                                                                   one_dis=median_min_disease,
                                                                   two_dis=median_pair_min)

print('Age after')
print(df_cleaned['inclusion_age_max'].head(5))
## remove html tag
def remove_html(text):
    return BeautifulSoup(str(text), 'html.parser').get_text(separator=" ")

html_col = [f for f in df_cleaned.columns if f != ['results_url_link','web_address','source_register']]

for col in html_col:
    df_cleaned[col] = df_cleaned[col].apply(remove_html)

moji_col = ['study_title','contact_affiliation']

for col in moji_col:
    df_cleaned[col] = df_cleaned[col].apply(lambda x: fix_text(x))


unknown_list = ['countries','pregnant_participants']

for col in unknown_list:  
    df_cleaned[col] = (df_cleaned[col].astype(str).str.strip().replace({"NA": pd.NA}).fillna('Unknown'))

df_cleaned['country_codes'] = (df_cleaned['country_codes'].astype(str).str.strip().replace({
            "": pd.NA,
            "NA": pd.NA,
            "N/A": pd.NA,
            "NaN": pd.NA,
            "nan": pd.NA,}).fillna('UNK'))

drop_col = ['trial_id','study_title','original_condition','countries','intervention','source_register',
            'inclusion_criteria','exclusion_criteria','contact_affiliation','primary_outcome','secondary_outcome']
df_cleaned = df_cleaned.drop(columns=drop_col)

df_cleaned.to_csv('/Users/prim/Desktop/MSc/SCC450/project/data/ictrp_data_clean.csv', index=False,  encoding="utf-8")


## convert age=UNKNOWN to 120, and select min/max value for data which has 2 value ex. 12Y,20

df = pd.read_csv('/Users/prim/Desktop/MSc/SCC450/project/data/ictrp_data_clean.csv',  encoding="utf-8")
df_preprocessing = df.copy()


## convert publication
df_preprocessing['results_binary'] = (df_preprocessing['results_ind'] == "YES").astype(int)


## find durations
def durations(row):
    if row['results_date_completed'] is not pd.NaT and row['date_enrollment'] is not pd.NaT:
        return (row['results_date_completed'] - row['date_enrollment']).days
    else:
        return -1
## change str to date time again   
date_cols = ['date_registration', 'date_enrollment', 'results_date_completed','results_date_posted']
for col in date_cols:  
    df_preprocessing[col] = pd.to_datetime(df_preprocessing[col], dayfirst=True, errors='coerce').dt.date

df_preprocessing['durations_day'] = df_preprocessing.apply(durations, axis=1)


## sample size
plt.figure(figsize=(12,8))
plt.hist(df_preprocessing['target_sample_size'], bins=30)
plt.title("target_sample_size")
plt.xlabel("sample size")
plt.ylabel("counts")
plt.show()

## fill target_sample_size missing value with median since it's right skew
sample_size_median = df_preprocessing['target_sample_size'].median()
df_preprocessing['target_sample_size'] = df_preprocessing['target_sample_size'].fillna(sample_size_median)

## one-hot encoding
one_hot_list = {'inclusion_gender':"Gender",
                'study_type':'Study_type',
                'phase':'phase',
                'randomization':'randomization',
                'primary_purpose':'primary_purpose',
                'intervention_model':'intervention_model',
                'masking':'masking',
                'endpoint_classification':'endpoint_classification',
                }

df_preprocessing = pd.get_dummies(df_preprocessing, columns=list(one_hot_list.keys()),prefix=one_hot_list, dtype=int)

df_preprocessing['primary_secondary_sponsor'] = df_preprocessing['secondary_sponsor'].apply(lambda x: 0 if x=='UNKNOWN' else 1)
df_preprocessing = df_preprocessing.drop(columns=['primary_sponsor','secondary_sponsor'])

## fill pregnent missing value by using age range criteria
### age < 10 , age > 60, and male are impossible to get pregnent
print("\n---------------\n")
print('Pregnent before')
print(df_preprocessing.shape)
def pregnent(row):
    if row['pregnant_participants']=='INCLUDED':
        return 'YES'
    else:
        if (row['inclusion_age_max'] < 10) or (row['inclusion_age_max'] > 60) or (row['Gender_MALE'] == 1):
            return 'NO'
        else:
            return 'UNKNOWN'

df_preprocessing['pregnant_participants'] = df_preprocessing.apply(pregnent, axis=1)
one_hot_preg = {'pregnant_participants':"pregnant_participants"}
df_preprocessing = pd.get_dummies(df_preprocessing, columns=list(one_hot_preg.keys()),prefix=one_hot_preg, dtype=int)
print('Pregnent after')
print(df_preprocessing.shape)
print(df_preprocessing['pregnant_participants_NO'].head(10))


##convert single/multiples country
map_country = {'SINGLE COUNTRY':0,
               'MULTI-COUNTRY':1,
               'UNKNOWN':-1}

df_preprocessing['Multi-Country'] = df_preprocessing['centre'].apply(lambda x: map_country.get(x))

##replace placebe missing value as NO
df_preprocessing['placebo'] = df_preprocessing['placebo'].fillna(0)
df_preprocessing['placebo'] = df_preprocessing['placebo'].apply(lambda x: 1 if x=='YES' else 0)

df_preprocessing = df_preprocessing.drop(columns=['centre','web_address','standardised_condition','results_ind',
                                                  'results_url_link','results_date_posted','results_date_completed',
                                                  'date_registration','date_enrollment'])
## convert Resprospective from Yes/No -> 1/0
df_preprocessing['retrospective_flag'] = df_preprocessing['retrospective_flag'].apply(lambda x: 1 if x=='YES' else 0)
## check duplicate column
print(df.duplicated().sum())



## data normalisation
### create normalisation for target sample size, age max and mix
#### use log transformation for target sample size since it's right skew
df_preprocessing['target_sample_size_normalised'] = np.log1p(df_preprocessing['target_sample_size'])

print(df_preprocessing['target_sample_size_normalised'].head(10))
#### use min-max scalar for age value since we know min-max value
scaler = MinMaxScaler()
df_preprocessing[['age_max_normalised', 'age_min_normalised']] = scaler.fit_transform(df_preprocessing[['inclusion_age_max', 'inclusion_age_min']])

## save file
df_preprocessing.to_csv('/Users/prim/Desktop/MSc/SCC450/project/data/ictrp_data_preprocessed.csv', index=False,  encoding="utf-8")
        
        
        
        
        
        