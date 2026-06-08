import pandas as pd
import plotly.express as px
from itertools import combinations
from collections import Counter

##########################################################################################################

#Can a network analysis of countries where studies are occurring provide insights? 
#Are there frequent partnerships between a few countries? 

#map showing number of trials done per country - white means none in dataset
#import data into a dataframe
df = pd.read_csv("Project/ictrp_data - Copy.csv")
df2 = pd.read_csv("Project\ictrp_data_preprocessed_prim.csv")

#show the length of the dataset for comparison
print(f"Number of studies: {len(df2)}")
#split the data points with multiple countries into each separate country then create a new column
#This allows for all countries to be seen and counted
df2["split_country"] = df["country_codes"].str.split('|')
df_expand = df2.explode("split_country").reset_index(drop=True)
df_expand.head()

#create a new dataframe that counts the number of trials per country
map_data = (
    df_expand
    .groupby("split_country")
    .size()
    .reset_index(name="trial_count")
)

#plot the map graph
fig = px.choropleth(map_data,
                    locations="split_country",
                    locationmode="ISO-3",
                    color="trial_count",
                    color_continuous_scale=px.colors.sequential.Viridis,
                    scope="world",
                    labels={"results_binary": "Trial Publication Status"},
                    title="Trial Publication Status per Country"
                   )

#adjust the map layout for better view
fig.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0}
)

fig.show()

#map showing proportion of publications done per country - white means none in dataset

#Same as before except we do .mean() not .size()
#create a new dataframe that counts the number of trials per country
map_data = (
    df_expand
    .groupby("split_country")["results_binary"]
    .mean()
    .reset_index(name="trial_count")
)

#plot the map graph
fig = px.choropleth(map_data,
                    locations="split_country",
                    locationmode="ISO-3",
                    color="trial_count",
                    color_continuous_scale=px.colors.sequential.Viridis,
                    scope="world",
                    labels={"results_binary": "Trial Publication Status"},
                    title="Proportion of Publications per Country"
                   )

#adjust the map layout for better view
fig.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0}
)

fig.show()

##########################################

#Filter the DataFrame for "Multi-Country" centres - use .copy() to remove some misc output in the console
#doesn't use cleaned multi-country from cleaned data since this was done before that
multi_country_df = df[df["centre"].astype(str).str.strip() == "Multi-Country"].copy()
print(f"Number of studies done by a single country: {len(multi_country_df)}")

single_country_df = df[df["centre"].astype(str).str.strip() == "Single Country"]
print(f"Number of studies done by a single country: {len(single_country_df)}")

def process_country_codes(code_string):   
    #Same as before a choropleth maps just using itertools instead so that the dataset can be expanded
    cleaned_code_set = set(c.strip() for c in code_string.upper().split('|') if c.strip())
    return sorted(list(cleaned_code_set))


#Get the country codes and apply the function - have to use .apply() since other way didn't work on the column
multi_country_df["cleaned_codes"] = multi_country_df["country_codes"].apply(process_country_codes)
#Get the number of countries in the multiple country row and puts it in a dataframe
multi_country_df["num_countries"] = multi_country_df["cleaned_codes"].apply(len)

max_countries = multi_country_df["num_countries"].max()
print("Max group size = " + str(max_countries))

#Set the iteration range from 2 up to the determined maximum
#added 10 which is one more than current max just to make sure the code works in the end 
#10 can be changed at will for different lengths of datasets but failsafe already in place if people cba with changing it 
#max_n = max(max_countries, 10)
collaboration_range = range(2, max_countries + 1)


results = {}
for N in collaboration_range:
    collaboration_counter = Counter()
    
    #Code to find the combinations for each N from 2 to 10
    for code_list in multi_country_df["cleaned_codes"]:
        #Only consider projects that involve at least N codes
        if len(code_list) >= N:
            #Generate all unique combinations of size N
            for group in combinations(code_list, N):
                collaboration_counter[group] += 1
    
    #Store top 10 results for each N, more can be done at request
    results[N] = collaboration_counter.most_common(100)

#My understanding is that it iterates through the results data so that the N value and the combination country codes are joined together, 
#the count of how many times they were mentioned together, and it prints the top 5 of each N size from 2 to 10
for N, top_collaborations in results.items():
    
    if not top_collaborations:
        print(f"\nNo collaborations involving {N} country codes found.")
        continue
    
    print(f"\nTop 5 Collaborations Involving N={N} Codes:")
    
    for rank, (codes, count) in enumerate(top_collaborations[:10], 1): 
        codes_str = ", ".join(codes)
        print(f"  {rank}. ({codes_str}): {count} collaborations")


############################################################################

#Are certain populations, such as pregnant women or children, being included in studies? 
#Typically, they are excluded due to the extra vulnerability in the population.

#get the number of pregnant women and divide it over the number of rows
proportion_pregnant = df2["pregnant_participants_YES"].sum() / len(df2)


print(f"Total Number of Studies with Pregnant Participants: {df2['pregnant_participants_YES'].sum()}")
print(f"Proportion: {proportion_pregnant:.4f}")
#severely underrepresented for pregnant women

##################################
map_data = (
    df_expand
    .groupby("split_country")["pregnant_participants_YES"]
    .mean()
    .reset_index(name="trial_count")
)

#plot the map graph
fig = px.choropleth(map_data,
                    locations="split_country",
                    locationmode="ISO-3",
                    color="trial_count",
                    color_continuous_scale=px.colors.sequential.Viridis,
                    scope="world",
                    labels={"results_binary": "Trial Publication Status"},
                    title="Proportion of Trials with Pregnant Participants per Country"
                   )

fig.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0}
)

fig.show()
#################################
age = [18,16,14,12,10]
prop_pediatric = {}

for i in age:
    df2['min_age'] = (
        df2['inclusion_age_min'] < i
    ).astype(int)
    prop_pediatric[i] = df2['min_age'].sum() / len(df2)
    print(f"Proportion of Trials with Min Age < {i}: {prop_pediatric[i]:.4f}")

##########################################################################################################

numeric_ages = pd.to_numeric(df2['inclusion_age_min'], errors='coerce')

age_df = df2.dropna(subset=['inclusion_age_min']).copy()

fig = px.histogram(
        age_df,
        x="inclusion_age_min",
        nbins=50,
        title="Distribution of Minimum Age",
        labels={"inclusion_age_min": "Minimum Age (Years)",
                "count": "Number of Trials"},
        color_discrete_sequence=px.colors.sequential.Viridis
    )
fig.show()