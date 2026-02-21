import numpy as np

def fetch_medal_tally(df, year, country):
    medal_df = df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'])

    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df
    elif year == 'Overall':
        temp_df = medal_df[medal_df['region'] == country]
    elif country == 'Overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]
    else:
        temp_df = medal_df[(medal_df['Year'] == int(year)) & (medal_df['region'] == country)]

    if country != 'Overall' and year == 'Overall':
        x = temp_df.groupby('Year')[['Gold','Silver','Bronze']].sum().reset_index()
    else:
        x = temp_df.groupby('region')[['Gold','Silver','Bronze']].sum().reset_index()

    x['total'] = x[['Gold','Silver','Bronze']].sum(axis=1)
    return x

def country_year_list(df):
    years = sorted(df['Year'].unique().tolist())
    years.insert(0, 'Overall')

    countries = sorted(df['region'].dropna().unique().tolist())
    countries.insert(0, 'Overall')

    return years, countries

def data_over_time(df, col):
    temp = df.drop_duplicates(['Year', col])
    result = temp.groupby('Year')[col].count().reset_index()

    result.rename(columns={'Year': 'Edition', col: col}, inplace=True)

    return result.sort_values('Edition')

def most_successful(df, sport):
    temp = df.dropna(subset=['Medal'])
    if sport != 'Overall':
        temp = temp[temp['Sport'] == sport]

    return temp['Name'].value_counts().reset_index().head(15)

def yearwise_medal_tally(df, country):
    temp = df.dropna(subset=['Medal']).copy()
    temp = temp[temp['region'] == country]
    return temp.groupby('Year')['Medal'].count().reset_index()

def country_event_heatmap(df, country):
    temp = df.dropna(subset=['Medal']).copy()
    temp = temp[temp['region'] == country]
    return temp.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count').fillna(0)

def most_successful_countrywise(df, country):
    temp = df.dropna(subset=['Medal'])
    temp = temp[temp['region'] == country]
    return temp['Name'].value_counts().reset_index().head(10)

def weight_v_height(df, sport):
    temp = df.drop_duplicates(subset=['Name','region']).copy()
    temp['Medal'] = temp['Medal'].fillna('No Medal')

    if sport != 'Overall':
        temp = temp[temp['Sport'] == sport]

    return temp

def men_vs_women(df):
    temp = df.drop_duplicates(subset=['Name','region'])

    men = temp[temp['Sex']=='M'].groupby('Year')['Name'].count().reset_index()
    women = temp[temp['Sex']=='F'].groupby('Year')['Name'].count().reset_index()

    final = men.merge(women,on='Year',how='left')
    final.rename(columns={'Name_x':'Male','Name_y':'Female'}, inplace=True)
    final.fillna(0, inplace=True)

    return final