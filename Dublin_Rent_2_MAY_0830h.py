#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

import statistics as stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn import metrics


# ### Introduction
# 
# We are looking at this dataset for rent prices as they were in Dublin for a range of years between 2008 and 2021, these are prices of rented properties; our goal is to be able to make some conclusions on the prices of rent in recent years and to be able to predict the rent prices.
# 
# We have imported our data from the Central Statistics Office (CSO) which collates information from Residential Tenancies Board (RTB) on Average Month Rents in Dublin by way of completed Houehold Surveys.  The CSO is the official, impartial collector and distributor of statistics about Ireland

# In[2]:


#Importing 2 sets of data and concatenating them.  

df1 = pd.read_csv('drp1.csv')
df2 = pd.read_csv('drp2.csv')

df = pd.concat([df1, df2], axis=0)
df.to_csv('df.csv', index=False)


# In[3]:


# Show the first five records
df.head()


# In[4]:


#explain enc_df = pd.DataFrame()
#enc_df = df.iloc[:,:]


# In[5]:


# Identify the number of columns, labels, data types, memory usage, range index, and non-null values
df.info()


# In[6]:


# missing value % calculation

df_null = round(100*(df.isnull().sum())/len(df), 2)
df_null


# In[7]:


#  Contains information for float/interger in each column: count, mean, standard derviation, minimum, maximum, quartile 1, mode and quartile 3
df.describe()


# We can see this gives us a general understanding of the distribution of rent prices in Dublin such as the average rent price over the entire period from 2008 to 2021 is €1,349.78.  The standard deviation of the rental prices is €461.64, which indicates that the rent prices vary significantly from the mean. 

# In[8]:


# Describe the object data
df.describe(include=object)


# This allows us to see that there are. 107548 unique records int he dataset.  This includes STATISTIC Label,Number of Bedrooms, Property Type, Location, UNIT.  We can also see that there are 167 unique values for the Location.  As we are concentrating on the Dublin Region we know we will have to cleanse this data further.  

# In[9]:


# We will drop Statistic Label and unit columns, as these features do not provide any information or pattern we can use in our anaylsis
drop_col = ['STATISTIC Label', 'UNIT']
df.drop(columns=drop_col,axis=1, inplace=True)


# In[10]:


# Use describe() function to see if above features were dropped
df.describe(include=object)


# In[11]:


df.duplicated().sum()


# In[12]:


print("before:",df.shape)
df.drop_duplicates(inplace=True)
print("after",df.shape)


# In[13]:


# identify the various types of beds with this feature
df['Number of Bedrooms'].unique()


# In[14]:


df = df.loc[~df['Number of Bedrooms'].str.contains('All bedrooms')]
#df = df.loc[~df['Number of Bedrooms'].str.contains('1 to 2 bed')]
#df = df.loc[~df['Number of Bedrooms'].str.contains('1 to 3 bed')]


# In[15]:


# check to see if bedrooms have been removed
df['Number of Bedrooms'].unique()


# In[16]:


rdm_frst = df.copy()


# # Testing Encoding Methods

# In[17]:


enc_df = pd.DataFrame()
enc_df = df.iloc[:,:]


# In[18]:


hot = pd.get_dummies(df["Number of Bedrooms"])
hot


# In[19]:


enc_df = pd.concat([df,hot], axis=1)
enc_df.drop("Number of Bedrooms", inplace=True, axis=1)


# In[20]:


enc_df


# In[21]:


#sns.pairplot(df)


# In[22]:


# Use describe() function to see if object features were transformed into numerical
enc_df.describe()


# In[23]:


# identify the various Property TYpes within this feature
enc_df['Property Type'].unique()


# In[24]:


# transform the object features to numerial
# df['Property Type'].replace(['All property types', 'Detached house', 'Semi detached house','Terrace house', 'Apartment', 'Other flats'], [1, 2, 3, 4, 5, 6,], inplace=True)


# In[25]:


enc_df.Location.nunique()
#number of unique values


# In[26]:


#Replacing the full strings with just the dublin code
enc_df.Location = enc_df.Location.str.replace(r".*Dublin$", "Dublin 0")
enc_df.Location = enc_df.Location.str.replace(r"^.*Dublin", "Dublin")
#df.Location = df.Location.str.replace("Dublin", "")
enc_df.Location.value_counts()


# In[27]:


enc_df.head()


# In[52]:


corr_df = df.dropna()
corr_df.Location = corr_df.Location.str.replace(r".*Dublin$", "Dublin 0")


# In[53]:


from numpy.random import seed, randn
from scipy.stats import pearsonr

seed(1)
corr, _ = pearsonr(corr_df["VALUE"], corr_df.Location.str.replace(r"[^0-9]", "").astype(int))
print(corr)


# In[ ]:





# In[ ]:





# In[ ]:


#Creating a data frame dublin_map to test out geospatial choropleth maps 
dublin_map = enc_df


# In[ ]:


one_hot_location = pd.get_dummies(enc_df['Location'])
enc_df = enc_df.join(one_hot_location)
enc_df.drop("Location", axis=1, inplace=True)
enc_df


# In[ ]:


one_hot = pd.get_dummies(enc_df['Property Type'])
#one_hot
#enc_df.columns

# Join the one-hot encoded column(s) back to the original dataframe
enc_df = enc_df.join(one_hot)
enc_df.drop("Property Type", axis=1, inplace=True)


# In[ ]:


# Use describe() function to confirm one hot encoding
enc_df.describe()


# In[ ]:


# identify the various Property Types within this feature
enc_df.info()

df['Location'].map({'Dublin 24':'Dublin 24', 'Ballycullen, Dublin 24':'Dublin 24',
       'Citywest, Dublin 24':'Dublin 24', 'Firhouse, Dublin 24':'Dublin 24',
       'Tallaght, Dublin 24':'Dublin 24','Clondalkin, Dublin 22': 'Dublin 22', 'Dublin 20':'Dublin 20',
       'Chapelizod, Dublin 20':'Dublin 20', 'Palmerstown, Dublin 20':'Dublin 20', 'Dublin 18':'Dublin 18', 'Cabinteely, Dublin 18':'Dublin 18', 'Carrickmines, Dublin 18':'Dublin 18',
       'Foxrock, Dublin 18':'Dublin 18', 'Leopardstown, Dublin 18':'Dublin 18',
       'Sandyford, Dublin 18':'Dublin 18', 'Stepaside, Dublin 18':'Dublin 18', 'Dublin 17':'Dublin 17',
       'Malahide Road, Dublin 17':'Dublin 17', 'Northern Cross, Dublin 17':'Dublin 17', 'Dublin 16':'Dublin 16', 'Ballinteer, Dublin 16':'Dublin 16',
       'Dundrum, Dublin 16':'Dublin 16', 'Knocklyon, Dublin 16':'Dublin 16',
       'Rathfarnham, Dublin 16':'Dublin 16', 'Sandyford, Dublin 16':'Dublin 16', 'Dublin 15':'Dublin 15',
       'Ashtown, Dublin 15':'Dublin 15', 'Blanchardstown, Dublin 15':'Dublin 15',
       'Carpenterstown, Dublin 15':'Dublin 15', 'Castleknock, Dublin 15':'Dublin 15',
       'Clonee, Dublin 15':'Dublin 15', 'Clonsilla, Dublin 15':'Dublin 15', 'Coolmine, Dublin 15':'Dublin 15',
       'Mulhuddart, Dublin 15':'Dublin 15', 'Ongar, Dublin 15':'Dublin 15',
       'Porterstown, Dublin 15':'Dublin 15', 'Royal Canal Park, Dublin 15':'Dublin 15',
       'Tyrrelstown, Dublin 15':'Dublin 15', 'Dublin 14':'Dublin 14', 'Churchtown, Dublin 14':'Dublin 14',
       'Clonskeagh, Dublin 14':'Dublin 14', 'Dundrum, Dublin 14':'Dublin 14',
       'Goatstown, Dublin 14':'Dublin 14', 'Rathfarnham, Dublin 14':'Dublin 14', 'Dublin 13':'Dublin 13',
       'Baldoyle, Dublin 13':'Dublin 13', 'Balgriffin, Dublin 13':'Dublin 13',
       'Clongriffin, Dublin 13':'Dublin 13', 'Donaghmede, Dublin 13':'Dublin 13',
       'Sutton, Dublin 13':'Dublin 13', 'Dublin 12':'Dublin 12', 'Crumlin, Dublin 12':'Dublin 12', 'Drimnagh, Dublin 12':'Dublin 12',
       'Kimmage, Dublin 12':'Dublin 12', 'Walkinstown, Dublin 12':'Dublin 12', 'Dublin 11':'Dublin 11', 'Ballymun, Dublin 11':'Dublin 11',
       'Finglas, Dublin 11':'Dublin 11', 'Glasnevin, Dublin 11':'Dublin 11',
       'Meakstown, Dublin 11':'Dublin 11', 'St Margarets Road, Dublin 11':'Dublin 11', 'Dublin 10':'Dublin 10', 'Ballyfermot, Dublin 10':'Dublin 10',
       'Cherry Orchard, Dublin 10':'Dublin 10', 'Dublin 9':'Dublin 9', 'Ballymun, Dublin 9':'Dublin 9', 'Beaumont, Dublin 9':'Dublin 9',
       'Drumcondra, Dublin 9':'Dublin 9', 'Glasnevin, Dublin 9':'Dublin 9', 'Santry, Dublin 9':'Dublin 9',
       'Whitehall, Dublin 9':'Dublin 9', 'Dublin 8':'Dublin 8',
       'Christchurch, Dublin 8':'Dublin 8', 'Cork Street, Dublin 8':'Dublin 8',
       'Dolphins Barn, Dublin 8':'Dublin 8', 'Inchicore, Dublin 8':'Dublin 8',
       'Islandbridge, Dublin 8':'Dublin 8', 'Kilmainham , Dublin 8':'Dublin 8',
       'Portobello, Dublin 8':'Dublin 8', 'Rialto, Dublin 8':'Dublin 8',
       'South Circular Road, Dublin 8':'Dublin 8', 'The Coombe, Dublin 8':'Dublin 8', 'Dublin 7':'Dublin 7', 'Arbour Hill, Dublin 7':'Dublin 7',
       'Cabra, Dublin 7':'Dublin 7', 'Navan Road, Dublin 7':'Dublin 7',
       'North Circular Road, Dublin 7':'Dublin 7', 'Phibsboro, Dublin 7':'Dublin 7',
       'Smithfield, Dublin 7':'Dublin 7', 'Stoneybatter, Dublin 7':'Dublin 7', 'Dublin 6W':'Dublin 6w',
       'Harolds Cross, Dublin 6W':'Dublin 6w', 'Templeogue, Dublin 6W':'Dublin 6w',
       'Terenure, Dublin 6W':'Dublin 6w', 'Dublin 6':'Dublin 6', 'Dartry, Dublin 6':'Dublin 6', 'Harolds Cross, Dublin 6':'Dublin 6',
       'Milltown, Dublin 6':'Dublin 6', 'Ranelagh, Dublin 6':'Dublin 6', 'Rathgar, Dublin 6':'Dublin 6',
       'Rathmines, Dublin 6':'Dublin 6', 'Terenure, Dublin 6':'Dublin 6', 'Dublin 5':'Dublin 5',
       'Artane, Dublin 5':'Dublin 5', 'Killester, Dublin 5':'Dublin 5', 'Raheny, Dublin 5':'Dublin 5', 'Dublin 4':'Dublin 4',
       'Ballsbridge, Dublin 4':'Dublin 4', 'Donnybrook, Dublin 4':'Dublin 4',
       'Irishtown, Dublin 4':'Dublin 4', 'Merrion, Dublin 4':'Dublin 4', 'Pembroke, Dublin 4':'Dublin 4',
       'Ringsend, Dublin 4':'Dublin 4', 'Sandymount, Dublin 4':'Dublin 4', 'Dublin 3':'Dublin 3', 'Clonliffe, Dublin 3':'Dublin 3',
       'Clontarf, Dublin 3':'Dublin 3', 'Drumcondra, Dublin 3':'Dublin 3',
       'East Wall, Dublin 3':'Dublin 3', 'Eastwall, Dublin 3':'Dublin 3', 'Fairview, Dublin 3':'Dublin 3',
       'Marino, Dublin 3':'Dublin 3', 'North Strand, Dublin 3':'Dublin 3', 'Dublin 2':'Dublin 2', 'Aungier Street, Dublin 2':'Dublin 2',
       'Charlemont Street, Dublin 2':'Dublin 2', 'Grand Canal Dock, Dublin 2':'Dublin 2',
       'Grand Canal Square, Dublin 2':'Dublin 2', 'Hanover Quay, Dublin 2':'Dublin 2',
       'Lower Mount Street, Dublin 2':'Dublin 2', 'Pearse Street, Dublin 2':'Dublin 2',
       'Tara Street, Dublin 2':'Dublin 2', 'Temple Bar, Dublin 2':'Dublin 2',
       'Townsend Street, Dublin 2':'Dublin 2','Dublin 1':'Dublin 1', 'I.F.S.C., Dublin 1':'Dublin 1',
       'Parnell Street, Dublin 1':'Dublin 1', 'Spencer Dock, Dublin 1':'Dublin 1',
       'Summerhill, Dublin 1':'Dublin 1', 'Dublin':'Dublin xxx', 'Balbriggan, Dublin':'Dublin xxx', 'Blackrock, Dublin':'Dublin xxx',
       'Booterstown, Dublin':'Dublin xxx', 'Cabinteely, Dublin':'Dublin xxx', 'Citywest, Dublin':'Dublin xxx',
       'Dalkey, Dublin':'Dublin xxx', 'Donabate, Dublin':'Dublin xxx', 'Dun Laoghaire, Dublin':'Dublin xxx',
       'Glenageary, Dublin':'Dublin xxx', 'Howth, Dublin':'Dublin xxx', 'Killiney, Dublin':'Dublin xxx',
       'Kinsealy, Dublin':'Dublin xxx', 'Lucan, Dublin':'Dublin xxx', 'Lusk, Dublin':'Dublin xxx',
       'Malahide, Dublin':'Dublin xxx', 'Monkstown, Dublin':'Dublin xxx', 'Mount Merrion, Dublin':'Dublin xxx',
       'Portmarnock, Dublin':'Dublin xxx', 'Rathcoole, Dublin':'Dublin xxx', 'Rush, Dublin':'Dublin xxx',
       'Saggart, Dublin':'Dublin xxx', 'Sandycove, Dublin':'Dublin xxx', 'Shankill, Dublin':'Dublin xxx',
       'Skerries, Dublin':'Dublin xxx', 'Stepaside, Dublin':'Dublin xxx', 'Stillorgan, Dublin':'Dublin xxx',
       'Swords, Dublin':'Dublin xxx'})
# In[ ]:


#  Replace the transform the object features to numerial


# In[ ]:


# Check for dulipcates
#enc_df.duplicated().sum()


# In[ ]:


# identify nulls. Can we do anything with these?
enc_df.isnull().sum()


# In[ ]:


# drop all null values
#enc_df.drop_duplicates(inplace=True)
enc_df.dropna(axis=0, inplace=True)


# In[ ]:


# Confirm nulls have been dropped
# Describe the object data
enc_df.shape


# ## Looking at the '1 to 2' and '1 to 3' bed columns

# In[ ]:


fig, ax = plt.subplots(figsize=(18, 6))
ax.set_title('Comparing the data of "1 to 2 bed" to that of 1, 2 and 3 bed data')
ax.set_xlabel('Average Monthly Rent')
ax.set_yticks([1])

ax.scatter(enc_df[enc_df['One bed'] == 1]["VALUE"], enc_df[enc_df["One bed"]==1]["One bed"], 
           marker="o", label="1 bed", facecolors='blue', edgecolors='blue', s=300, alpha=1);

ax.scatter(enc_df[enc_df['Two bed'] == 1]["VALUE"], enc_df[enc_df["Two bed"]==1]["Two bed"], 
           marker="o", label="2 bed", facecolors='red', edgecolors='red', s=140, alpha=1);

ax.scatter(enc_df[enc_df['Three bed'] == 1]["VALUE"], enc_df[enc_df['Three bed'] == 1]["Three bed"], 
           marker="o", label="3 bed", facecolors='green', edgecolors='green', s=70, alpha=1);

ax.scatter(enc_df[enc_df['1 to 2 bed'] == 1]["VALUE"], enc_df[enc_df['1 to 2 bed'] == 1]["1 to 2 bed"], 
           marker=".", label="1 to 2 bed", facecolors='black', edgecolors='black');
ax.legend(loc="upper right");


# In[ ]:


# specify the font and style
font = fm.FontProperties(family='Arial', style='normal', weight='bold', size=20)

# set the ggplot style
plt.style.use('ggplot')

# create a figure and axis object
fig, ax = plt.subplots(figsize=(18, 6))

# plot overlapping histograms for each group
sns.histplot(data=enc_df[enc_df['One bed'] == 1], x='VALUE', color='blue', label='1 bedroom', alpha=0.5, ax=ax, kde=False)
sns.histplot(data=enc_df[enc_df['Two bed'] == 1], x='VALUE', color='red', label='2 bedroom', alpha=0.5, ax=ax, kde=False)
sns.histplot(data=enc_df[enc_df['Three bed'] == 1], x='VALUE', color='green', label='3 bedroom', alpha=0.5, ax=ax, kde=False)
sns.histplot(data=enc_df[enc_df['1 to 2 bed'] == 1], x='VALUE', color='black', label='1 to 2 bedroom', alpha=0.5, ax=ax, kde=False)

# set the title and axis labels with the specified font
ax.set_title('Comparing the data of "1 to 2 bed" to that of 1, 2 and 3 bed data', fontproperties=font)
ax.set_xlabel('Average Monthly Rent', fontproperties=font)
ax.set_ylabel('Number of Properties', fontproperties=font)

# set the legend font and size
ax.legend(prop=font)

# show the plot
plt.show()


# We can see that the data for 0 is equal for all, but the data for 1 to 2 bed rent prices matchs with the data for 2 bed
# therefore we will group these two values togther, though the minimum value is slightly lower than 2 bedrooms

# In[ ]:


fig, ax = plt.subplots(figsize=(18, 5))
ax.set_title('Comparing the data of "1 to 2 bed" to that of 1, 2 and 3 bed data')
ax.set_xlabel('Average Monthly Rent')
ax.set_yticks([1])

ax.scatter(enc_df[enc_df['One bed'] == 1]["VALUE"], enc_df[enc_df["One bed"]==1]["One bed"], 
           marker="o", label="1 bed", facecolors='blue', edgecolors='blue', s=300, alpha=1);

ax.scatter(enc_df[enc_df['Two bed'] == 1]["VALUE"], enc_df[enc_df["Two bed"]==1]["Two bed"], 
           marker="o", label="2 bed", facecolors='red', edgecolors='red', s=140, alpha=1);

ax.scatter(enc_df[enc_df['Three bed'] == 1]["VALUE"], enc_df[enc_df['Three bed'] == 1]["Three bed"], 
           marker="o", label="3 bed", facecolors='green', edgecolors='green', s=70, alpha=1);

ax.scatter(enc_df[enc_df['1 to 3 bed'] == 1]["VALUE"], enc_df[enc_df['1 to 3 bed'] == 1]["1 to 3 bed"], 
           marker=".", label="1 to 3 bed", facecolors='black', edgecolors='black');
ax.legend(loc="upper right");


# With 1 to 3 bed data, its unclear, we can see from the graph that there are values which dont match with the 2 bed prices as they are higher than any seen there, and yet the 3 bed prices dont match either, as the 1 to 3 bed data has a lower minimum value, and in fact a low minimum value. Therefore we should either treat 1 to 3 bed as its own seperate value or drop the rows.

# # # Label encoding the number of bedrooms
# As we have a number of bedrooms variables that is categorical, we have the option of chosing the encoding methods, the one-hot encoding mthod may be used but as we have a variable with a clear ordinal pattern and we can see from the graphs that the number of bedrooms is correlated with the price, it makes sense to label encode this variable.
# As we said, for the 1 to 2 bed variable we can group it with the 2 bed variable, for the 1 to 3 bed variable we have some options: <br>
# <ul>
#    <li> eliminate the variable and its data
#    <li> eliminate the variation of this variable and put the data points which conform with (2 bed) together
#    <li> test the variable with different values to see if it increases accuracy eg. 1.5, 2.5

# In[ ]:


# replace the transform the object features to numerial


#df['Number of Bedrooms'].replace([1 to 3 bed'], [2.5], inplace=True)


# In[ ]:


df["Number of Bedrooms"].value_counts()


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


print("dropping '1 t 3 bed' values would mean losing: ",((df["Number of Bedrooms"] == "1 to 3 bed").sum()/df.shape[0]*100), "%")


# In[ ]:


ord_df = pd.DataFrame()
ord_df = df
ord_df['Number of Bedrooms'].replace(['One bed', 'Two bed', "1 to 2 bed", 'Three bed',  'Four plus bed'], [1, 2, 2, 3, 4], inplace=True)
ord_df.drop(columns=["Location", "Property Type"], axis=1, inplace=True)
ord_df = ord_df.join(one_hot)
ord_df


# In[ ]:


ord_df.dropna(inplace=True)
ord_df.shape


# # Testing accuracy of encoding

# In[ ]:


ord_df["Number of Bedrooms"].unique()


# In[ ]:


# test 1 - eliminating the 1 to 3 bed variable
df_eliminate = pd.DataFrame()
df_eliminate = ord_df.loc[ord_df['Number of Bedrooms']!="1 to 3 bed"]
df_eliminate.shape


# In[ ]:


## Test accuracy
from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor(max_depth = 4, random_state = 4)


# In[ ]:


y = df_eliminate["VALUE"]
X = df_eliminate.drop(columns="VALUE", axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[ ]:


metrics.r2_score(y_test, y_pred)


# ## Testing with grouping '1 to 3 bed' with '2 bed'

# In[ ]:


df_group = pd.DataFrame()
df_group = ord_df
df_group['Number of Bedrooms'].replace(['1 to 3 bed'], [2], inplace=True)
df_group.shape


# In[ ]:


clf = DecisionTreeRegressor(max_depth = 4, random_state = 4)
y = df_group["VALUE"]
X = df_group.drop(columns="VALUE", axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
metrics.r2_score(y_test, y_pred)


# The accuracy has dropped with this course of action so we should not continue, this makes sense as this way means that we are blurring the pattern of the data for '2bed' by adding more outliers without adding more information

# In[ ]:


## Testing with a new value eg. 2.5


# In[ ]:


for v in range(1,10):
    val = v/2
    df_test = pd.DataFrame()
    df_test = ord_df
    df_test['Number of Bedrooms'].replace(['1 to 3 bed'], [v], inplace=True)
    clf = DecisionTreeRegressor(max_depth = 4, random_state = 4)
    y = df_group["VALUE"]
    X = df_group.drop(columns="VALUE", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("for val", val, "accuracy is: ",metrics.r2_score(y_test, y_pred))


# No improvement: from this analysis we should drop the 1 to 3 bed rows from our dataset

# In[ ]:


ord_df = ord_df.loc[ord_df['Number of Bedrooms']!="1 to 3 bed"]
ord_df["Number of Bedrooms"].value_counts()


# ## Creating maps of Dublin 

# In[ ]:


#Checking the newly created dataframe
dublin_map.head()


# In[ ]:


#Dropping the property type and no. of bedroom columns and na values
dublin_map = dublin_map.drop(['Property Type', '1 to 2 bed', '1 to 3 bed', 'Four plus bed','One bed','Three bed','Two bed'], axis=1)
dublin_map.dropna(axis=0, inplace=True)


# In[ ]:


dublin_map.head()


# Looks good, as we will only need the year location and values

# In[ ]:


#Making sure the years are 2008-2021
dublin_map['Year'].unique()


# In[ ]:


#Calculating the mean values for each year, this in turn provides us with the figures of the mean rental price for that year
dublin_map.groupby('Year')['VALUE'].mean()


# In[ ]:


# Calculate mean for every year and naming it 'mean_values'
mean_values = dublin_map.groupby('Year')['VALUE'].mean()

# With the mean values we now create a plot with an attribute to increase the figure size
plt.figure(figsize=(18, 8))
plt.plot(mean_values.index, mean_values.values)

# Setting the chart title and x and y labels, finally plotting the values
plt.title('Mean Values by Year')
plt.xlabel('Year')
plt.ylabel('Mean Value')
plt.show()


# This is an interesting graph, as we can see the average rents go from around 1300 in 2008 down to around 1000 in 2011 and then they start climbing up again, this naturally shows us the effects of the 2008 global financial crisis. Moving on we can also see a change in the 2020 - 2021 years with a considerably slower growth. This could be explained by the covid-19 pandemic and the shakeup of the rental market. 

# In[ ]:


#Next I group the values by location and find the mean for each location for every year. 
mean_values = dublin_map.groupby(['Year','Location'])['VALUE'].mean()
mean_values


# In[ ]:


#Implementing a catplot here and checking the values for each of Dublin's postal codes
sns.catplot(x="Location", y="VALUE", kind="box", data=dublin_map, height=5, aspect=3)
plt.xticks(rotation=45);


# We can see a considerable amount of outliers in Dublin 1, and quite a high mean for Dublin 4. Given that these figures have all the years (2008 to 2021) its hard to see any major differences and also a lot of the information could be lost due to the compounding year effects.

# In[ ]:


#Creating a new data frame
mean_dublin_map = pd.DataFrame()

#Grouping by year and location and calculating the mean value
mean_dublin_map = dublin_map.groupby(['Year', 'Location'])['VALUE'].mean().reset_index()
mean_dublin_map.head()


# In[ ]:


#Specifiying the size of the graph
plt.figure(figsize=(14, 8))

#plotting a line plot with the legend on the right side outside the graph
sns.lineplot(x='Year', y='VALUE', hue='Location', data=mean_dublin_map);
plt.legend(bbox_to_anchor=(1.05, 1));


# Similar to the compounded linegraph we see very similiar trajectories in each of Dublin's postal regions with the notable exception of Dublin 1 displaying quite a random pattern and Dublin 14 continious to grow past 2020 and 2021.

# In[ ]:


#Rounding the values to the nearest € 
mean_dublin_map = mean_dublin_map.round()


# In[ ]:


#importing the necessary libraries 
import json


# In[ ]:


#loading new json file for coordinates of dublin postal regions
map_json = json.load(open("dublin_map.json", 'r'))


# In[ ]:


#checking the properties of the features
map_json['features'][0].keys()


# In[ ]:


#Creating an id map that can map the id's from the json file to the data frame
dublin_id_map = {}
for feature in map_json['features']:
    feature['id'] = feature['properties']['Name']
    dublin_id_map[feature['properties']['description']] = feature['id']


# In[ ]:


#checking the Id map
dublin_id_map


# In[ ]:


#Checking the properties
map_json['features'][1]['properties']


# In[ ]:


#Linking the json file and the dataframe and creating a new id column in the dataframe
mean_dublin_map['id'] = mean_dublin_map['Location'].apply(lambda x: dublin_id_map[x])


# In[ ]:


mean_dublin_map.head()


# In[ ]:


#importing the libraries to create a choropleth map of Dublin Postal regions
import plotly.express as px
import plotly.io as pio


# In[ ]:


#Here's a choropleth map created using the dataframe mean_dublin_map and json location file 
fig = px.choropleth(mean_dublin_map,
                    locations='id', 
                    geojson=map_json, 
                    color='VALUE',
                    scope='europe',
                    hover_name='Location',
                    animation_frame="Year",
                    range_color = [900,2200],
                    title = 'Average rent prices in Dublin 2008-2021')
fig.update_geos(fitbounds='locations', visible=False)
fig.show()


# # Modelling

# Gaussian Naive Bayes classifier or other clustering algorythms are not suitable for this problem as we have a continuous output y value and these algorythms need an output values that consists of groups or clusters to label.

# In[ ]:


#As dicussed we will drop column '1 to  bed' and merge '1 to 2 bed' into 'two bed'

enc_df.insert(8, 'new_col', (enc_df['1 to 2 bed'] | enc_df['Two bed']))

enc_df.drop(columns=["1 to 2 bed", "Two bed"], inplace=True)
enc_df.rename(columns={"new_col":"Two bed"}, inplace=True)
enc_df.drop(columns=["1 to 3 bed"], inplace=True)


# Since we have 35 columns in this dataset we may want to run PCA to do dimensionality reduction

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


X_r = enc_df.iloc[:,2:]
X_r["Year"] = enc_df["Year"]
y = enc_df.iloc[:,1]


# In[ ]:


pca = PCA()
pca.fit(X=X_r, y=y)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axis()
plt.ticklabel_format(useOffset=False)
plt.yticks([0.88, 0.9,0.92,0.94,0.95,0.96,0.98,1])
plt.xticks([0,  5, 7, 8, 10, 15, 20, 25, 30, 35])
plt.grid(True, which='both')
plt.title("Effectiveness of PCA");


# We see that the explained variance reaches 95% at about 7.5, so we will set the number of n-components at 8

# In[ ]:


pca = PCA(n_components=8)
X = pca.fit_transform(X=X_r)
pd.DataFrame(X)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)


# In[ ]:


lgr = LinearRegression()
lgr.fit(X_train, y_train)


# In[ ]:


mse = mean_squared_error(y_test, lgr.predict(X_test))
r2 = r2_score(y_test, lgr.predict(X_test))
print("The Mean squared error for linear regression is: {:.0f}".format(mse))
print("The r2 for linear regression is: {:.2f}".format(r2))


# The Mean squared error is very high and an r2 of one is a bit suspicious, lets look at the data it's predicting

# In[ ]:


new = pd.DataFrame(y_test)
new["pred"] = lgr.predict(X_test)


# In[ ]:


new.sample(10)


# When we look at the data we see that it doesnt look too bad, the values are similar, though still a rough prediction.
# 

# In[ ]:


plt.scatter(new["VALUE"], new["pred"], c="firebrick", facecolor="white")
plt.ylabel("Predicted Prices")
plt.xlabel("Actual Prices")
plt.title("Linear regression results");


# 
# We can see here that the values are skewed, especially by the outliers, the model doesnt predict these well and all results are > 2500. Ideally this graph should be a straight line if it predicted perfectly

# In[ ]:


lr = LinearRegression()
cross_val_score(lr, X_r, y, cv=10)[9]


# In[ ]:


from sklearn.model_selection import cross_val_score
yscores_pred = []

def get_scores(model):
    scores = cross_val_score(model,
                             X_train,
                             y_train,
                             cv = 10)
    
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))


# In[ ]:


lr = LinearRegression().fit(X_train, y_train)
get_scores(lr)


# # Random Forest 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rdm_frst.head()


# In[ ]:


#rdm_frst.drop(columns=["STATISTIC"],axis=1, inplace=True)


# In[ ]:


rdm_frst.dropna(inplace=True)


# In[ ]:


rdm_frst.Location = rdm_frst.Location.str.replace(r".*Dublin$", "Dublin County")
rdm_frst.Location = rdm_frst.Location.str.replace(r"^.*Dublin", "Dublin")


# In[ ]:


rdm_frst.groupby('Number of Bedrooms')['VALUE'].mean().sort_values(ascending=False)


# In[ ]:


rdm_frst['Number of Bedrooms'].value_counts()


# In[ ]:


rdm_frst['Number of Bedrooms'].replace({'One bed': '1', '1 to 2 bed':'1.5', 'Two bed':'2', '1 to 3 bed':'2', 'Three bed':'3', 'Four plus bed':'4'}, inplace=True)


# In[ ]:


rdm_frst['Number of Bedrooms'].info()


# In[ ]:


rdm_frst['Number of Bedrooms'] = rdm_frst['Number of Bedrooms'].astype(float)


# In[ ]:


rdm_frst['Property Type'].unique()


# In[ ]:


rdm_frst.groupby('Property Type')['VALUE'].mean().sort_values(ascending=False)


# In[ ]:


#!pip install category_encoders
import category_encoders as ce


# In[ ]:


binary_encoder = ce.BinaryEncoder(cols=['Location','Property Type'])
rdm_frst_encoded = binary_encoder.fit_transform(rdm_frst)
rdm_frst_encoded.head()


# In[ ]:


xVars = rdm_frst_encoded.drop('VALUE', axis=1)
yVars = rdm_frst_encoded[['VALUE']]
xTrain, xTest, yTrain, yTest = train_test_split(xVars, yVars, test_size=0.25, random_state=10)


# In[ ]:


regressor = RandomForestRegressor(n_estimators=100, random_state=10)


# In[ ]:


regressor.fit(xTrain,yTrain)


# In[ ]:


y_pred=regressor.predict(xTest)


# In[ ]:


y_pred


# In[ ]:


yTest


# In[ ]:


plt.figure(figsize=(10, 10))
plt.scatter(yTest, y_pred, color = 'blue', label='Comparison of Prediction between Actual & Prediction data')
plt.plot([0, 6000], [0, 6000], '--', color='red')
plt.title("Random Forest Regression")
plt.xlabel('Prediction data')
plt.ylabel('Actual data')
plt.xlim(0, 6000) 
plt.ylim(0, 6000)
plt.show();


# In[ ]:


metrics.r2_score(yTest,y_pred)


# ## Random Forest Hyperparameter tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = [
{'n_estimators': [10, 50], 'max_features': [5, 10], 
 'max_depth': [5, 30, None], 'bootstrap': [True, False]}
]

grid_rndm_frst = GridSearchCV(regressor, param_grid, cv=10, scoring='neg_mean_squared_error')
grid_rndm_frst.fit(xTrain, yTrain)
grid_rndm_frst.best_estimator_


# In[ ]:




