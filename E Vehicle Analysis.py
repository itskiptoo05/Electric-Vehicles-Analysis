#!/usr/bin/env python
# coding: utf-8

# # Electric Vehicle Analysis

# ## Data Import, Preview Cleaning

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


ev_pop = pd.read_csv("Electric_Vehicle_Population_Data.csv")
ev_pop.head()


# VIN (1-10) column has the vehicle identification number which is not of much benefit on our analysis so we will get rid of it. Among the columns we will drop include
# * Postal Code
# * Base MSRP
# * Legislative District
# * DOL Vehicle ID
# * 2020 Census Tract

# In[4]:


drop_columns = ["VIN (1-10)", "Postal Code", "Base MSRP", "Legislative District", "DOL Vehicle ID", "2020 Census Tract"]

ev_pop = ev_pop.drop(drop_columns, axis =1)


# In[5]:


ev_pop.head()


# In[6]:


ev_pop.shape


# After getting rid of the 5 columns, our dataset now is composed of 150482 rows and 12 columns. Let's also check if there are rows with null values

# In[7]:


na_col = ev_pop.isna().sum(axis=0)
na_col


# Only a few columns columns have null values hence will not affect our analysis

# In[8]:


ev_pop["State"].value_counts().head(10)


# In the results of the above code, we can see that 99 percent of the data are from Washington (WA) state, so let's filter out other states

# In[9]:


ev_pop_wa = ev_pop[ev_pop["State"] == "WA"]
ev_pop_wa.head()


# ## Analysis

# ### Electric Vehicle Types

# We will start our analysis by taking a look at the electric vehicle types

# In[10]:


ev_pop_wa["Electric Vehicle Type"].value_counts()


# In[11]:


type_counts = ev_pop_wa['Electric Vehicle Type'].value_counts()

plt.figure(figsize=(6, 6))
type_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Electric Vehicle Types', fontweight='bold', fontsize=14)
plt.ylabel('')

plt.text(.5, 0, "Data From Data.gov (https://data.gov/)", ha="center", va="center", transform=plt.gcf().transFigure)

plt.show()


# There are about 116585 battery electric vehicles (BEVs) and 33556 plug-in hybrid electric vehicles (PHEVs), this can be axpressed as roughly 77% and 22% respectively. BEVs are fully electric vehicles and are powered solely by electricity stored in large batteries. PHEVs on the other hand  have both an internal combustion engine (usually gasoline) and an electric motor. They can operate on electric power alone, gasoline power alone, or a combination of both.

# ### Top 20 EV Makes

# In[12]:


ev_pop_wa["Make"].value_counts()


# This dataset has over 40 car makes, we will just have a look at the top 20. Tesla comes first in the list folowed by Nissan, Chevrolet and so on 

# In[13]:


make_counts = ev_pop_wa['Make'].value_counts().nlargest(20)

plt.figure(figsize=(13, 7))
sns.barplot(x=make_counts.values, y=make_counts.index, palette="mako")
plt.title('Top 20 Makes of Electric Vehicles', fontweight='bold', fontsize=14)
plt.xlabel('Counts', fontweight='bold', fontsize=13)
plt.ylabel('Make', fontweight='bold', fontsize=13)

plt.text(.5, 0, "Data From Data.gov (https://data.gov/)", ha="center", va="center", transform=plt.gcf().transFigure)

plt.show()


# For a further analysis, we shall filter our data to contain only the top 10 vehicle makes i.e Tesla,  Nissan, Chevrolet, Ford, BMW, KIA, Toyota, Volkswagen, Volvo, and Jeep

# ### Counts Of The Top 10 Makes According To Their EV Types With 

# In[14]:


top_makes = ["TESLA", "NISSAN", "CHEVROLET", "FORD", "BMW", "KIA", "TOYOTA", "VOLKSWAGEN", "VOLVO", "JEEP"]

make_evtype = ev_pop_wa[ev_pop_wa["Make"].isin(top_makes)]

top_make_evtype = make_evtype.groupby(["Electric Vehicle Type", "Make"]).size().reset_index(name="Count")

top_make_evtype


# In[15]:


plt.figure(figsize = (13,7))
sns.barplot(data=top_make_evtype, x="Make", y="Count", hue="Electric Vehicle Type", palette="rocket")
plt.xlabel("Car Make", fontsize=13, fontweight="bold")
plt.ylabel("Count", fontsize=13, fontweight="bold")
plt.title("Counts of Electric Vehicle Types by Make", fontsize=14, fontweight="bold")
plt.legend(title='Electric Vehicle Type', loc="upper right")

plt.text(.5, 0, "Data From Data.gov (https://data.gov/)", ha="center", va="center", transform=plt.gcf().transFigure)

plt.show()


# From the above visualization, you can conclude the folowing about the electric cars in Washington state:
# * There are only battery electric vehicles by Tesla, Nissan and Volkswagen.
# * Jeeps on the other hand are only plug-in hybrid, whereas in the case of Toyota, there are more hybrids that only electric one.
# * For the remaining 5 makes, i.e Chevrolet, Ford, BMW, KIA and Volvo, there are almost same number of each vehicle type.

# ### Top 20 Models

# Let us even go deeper and do a make a head to head comparison of the car models instead of makes

# In[18]:


ev_pop_wa["Make & Model"] = ev_pop_wa.loc[:,"Make"] + " " + ev_pop_wa.loc[:,"Model"]

model_counts = ev_pop_wa["Make & Model"].value_counts()[:20]


model_counts


# In[19]:


plt.figure(figsize=(12, 6))
sns.barplot(x=model_counts.values, y=model_counts.index, palette="rocket")
plt.title("Top 20 Models of Electric Vehicles", fontweight='bold', fontsize =14)
plt.xlabel("Counts", fontsize=13, fontweight="bold")
plt.ylabel("Vehicle Models", fontsize=13, fontweight="bold")

plt.text(.5, 0, "Data From Data.gov (https://data.gov/)", ha="center", va="center", transform=plt.gcf().transFigure)

plt.show()


# In the above visualization, we can conclude that there are 4 Tesla models in the top 20 category, Tesla's  Model Y and Model 3 being at the top of the list. 

# ### Clean Alternative Fuel Vehicle Eligibility

# In[20]:


cavf = ev_pop_wa["Clean Alternative Fuel Vehicle (CAFV) Eligibility"].value_counts()
cavf


# In[21]:


plt.figure(figsize=(13, 7))

sns.barplot(x=cavf.index, y=cavf.values, palette="rocket")
plt.xlabel('CAFV Eligibility', fontsize=13, fontweight="bold")
plt.ylabel('Count', fontsize=14, fontweight="bold")
plt.title('Distribution of CAFV Eligibility', fontsize=14, fontweight='bold')
plt.xticks(fontsize=9)

for i, v in enumerate(cavf.values):
    plt.text(i, v, str(v), ha='center', va='bottom')
    
plt.text(.5, 0, "Data From Data.gov (https://data.gov/)", ha="center", va="center", transform=plt.gcf().transFigure)

plt.show()


# When it comes to clean alternative fuel vehicle (CAFV) eligibility category: 
# * About 46% of cars in Washington have their eligibility unkown as battery range has not been researched
# * About 41 percent of total cars are CAFV eligible
# * Close to 12 percent of total cars are not eligible due to low battery range

# ### Counties With The Most EV Vehicles 
# In our next analysis, we shall take a glance at only the top 20 counties

# In[22]:


top_20_counties = ev_pop_wa["County"].value_counts().nlargest(20).reset_index()
top_20_counties.columns = ["County", "Counts"]
top_20_counties


# In[24]:


plt.figure(figsize=(13,7))
sns.barplot(x=top_20_counties.Counts, y=top_20_counties.County, palette="rocket")
plt.xlabel("Counts", fontweight="bold", fontsize=13)
plt.ylabel("County", fontweight="bold", fontsize=13)
plt.title("Washington's Top 20 Counties With Electric Vehicles", fontweight="bold", fontsize=14)

plt.text(.5, 0, "Data From Data.gov (https://data.gov/)", ha="center", va="center", transform=plt.gcf().transFigure)

plt.show()


# King County has close to 8000 electric vehicles, this can be directrly attributed to various policies and incentives implemented to promote electric vehicle adoption. These include tax incentives, rebates, and access to carpool lanes for EVs, making it more appealing for residents to switch to electric transportation.

# #### Top 5 Counties Vs The Top 5 EV Makes

# Earlier on, we had noted that Tesla, Nissan, Chevrolet, Ford and BMW are at the top 5 electric car models overally. We have also seen above that King, Snohomish, Pierce, Clark, and Thurston are the top 5 counties in Washington which has high number of electric vehicles. With this, lets take a look at how many individual car makes are there in each of the said cities. 

# In[25]:


counties = ["King", "Snohomish", "Pierce", "Clark", "Thurston"]
makes1 = ["TESLA", "NISSAN", "CHEVROLET", "FORD", "BMW"]

county_makes = ev_pop_wa[(ev_pop_wa["County"].isin(counties) & ev_pop_wa["Make"].isin(makes1))]
top5_county_makes = county_makes.groupby(["County", "Make"]).size().reset_index(name="Count")

top5_county_makes


# In[26]:


plt.figure(figsize=(12, 6))
sns.barplot(data=top5_county_makes, x="Count", y="Make", hue="County", palette="rocket")
plt.xlabel("Count", fontweight="bold", fontsize=13)
plt.ylabel("Car Make", fontweight="bold", fontsize=13)
plt.title("Top 5 Car Makes in Each County", fontweight="bold", fontsize=14)
plt.legend(title='County')

plt.text(.5, 0, "Data From Data.gov (https://data.gov/)", ha="center", va="center", transform=plt.gcf().transFigure)

plt.show()


# In the above visualization, we can conclude that King County has the highest number of the top five electric vehicles. There are also many Teslas in this county than any other car make. Tesla vehicles are expensinve and since King County is the most wealthy county in Washington, this explains the reason behind the large number of this specific car make

# ### Electric Vehicles Versus Their Years Of Model: 2012 To 2022
# Now let's check which vehicle model year are common. Since the year 2023 is not yet over, we will filter it out although in the data below, we can see that this is the year with the highest electric car models. From the 1997 all teh way to 2011, there were fewer car modelled in this period and we can conclude that from 2012 is the electric vehicles era.

# In[27]:


ev_pop_wa["Model Year"].value_counts()


# In[28]:


yr_12_22 = ev_pop_wa[(ev_pop_wa["Model Year"] >=2012) & (ev_pop_wa["Model Year"] <= 2022)]

yr_12_22["Model Year"].describe()


# In[29]:


yr_counts = yr_12_22["Model Year"].value_counts()
yr_counts


# In[30]:


custom_palette= ["blue"]

plt.figure(figsize=(13,6))
sns.barplot(x=yr_counts.index, y=yr_counts.values, palette=custom_palette)
plt.xlabel("Year", fontweight="bold", fontsize=13)
plt.ylabel("Counts", fontweight="bold", fontsize=13)
plt.title("Growth Of Electric Vehicles Over The Years: 2012 Till 2022", fontweight="bold", fontsize=14)

plt.text(.5, 0, "Data From Data.gov (https://data.gov/)", ha="center", va="center", transform=plt.gcf().transFigure)
    
plt.show()


# From the above visualization, we can clearly see that electric vehicles industry has been on an uptrend. This could be due to technological innovations or the climate change campaigns advocating for less carbon emmisions. In the years 2019 and 2020, this growth stagnated a bit before shooting up in the preceeding years. 

# ### Electric Vehicle Types And Thier Years Of Model

# Since it seems the second and the third decades of the 21st century are the electric vehicles era, we can make a head to head comparison between the fully electric and hybrid cars 

# In[31]:


year_ev_type = yr_12_22.groupby(["Electric Vehicle Type", "Model Year"]).size().reset_index(name="Count")

year_ev_type


# In[32]:


plt.figure(figsize=(13,6))
sns.barplot(data=year_ev_type, x="Model Year", y="Count", hue="Electric Vehicle Type", palette="rocket")
plt.xlabel("Year", fontweight="bold")
plt.ylabel("Counts",fontweight="bold")
plt.title("Evolution of BEVs and PHEVs Over The Years: 2012 Till 2022", fontsize=14, fontweight="bold")

plt.text(.5, 0, "Data From Data.gov (https://data.gov/)", ha="center", va="center", transform=plt.gcf().transFigure)

plt.show()


# In the graph above, we can observe that in the years 2012 till 2022, hybrid, better known as  Plug-in Hybrid Electric Vehicles (PHEV), have been on somewhat like a stagnant growth. On the other hand, the fully elecrtic ones, Battery Electric Vehicles (BEV), has been doing really well with the highest growth witnessed in the models of the year 2022. Due to climate change implemantation plans and campaigns for electric vehicles, this growth is expected to even skyrocket in the remaining part of this third decade.

# ### Top 3 EVs Vs Their Year Of Model
# Ealier on we had witnessed that Tesla, Nissan and Volkswagen are the top electric car makes, we can see do a brief analysis of how each car make, and electric vehicle type have been perfoming in the said period, 2012 to 2022.

# In[33]:


bev_makes = ["TESLA", "NISSAN", "VOLKSWAGEN"]

yr_bev_makes = yr_12_22[(yr_12_22["Make"].isin(bev_makes)) & (yr_12_22["Electric Vehicle Type"] =="Battery Electric Vehicle (BEV)")]


# In[34]:


yr_bev_makes_1 = yr_bev_makes.groupby(["Make", "Model Year"]).size().reset_index(name="Counts")
yr_bev_makes_1


# In[35]:


palette_1= ["blue", "grey", "black"]

plt.figure(figsize=(13,7))
sns.barplot(data=yr_bev_makes_1, x="Model Year", y="Counts", hue="Make", palette=palette_1)
plt.ylabel("Counts", fontweight="bold")
plt.xlabel("Car Model Year", fontweight="bold")
plt.title("Evolution Of Top 3 Battery Electric Vehicles: Nissan, Volkswagen And Tesla", fontsize=14, fontweight="bold")
plt.legend(title="Car Make", loc="upper left")

plt.text(.5, 0, "Data From Data.gov (https://data.gov/)", ha="center", va="center", transform=plt.gcf().transFigure)

plt.show()


# In the visualization above, we can see in Washington, for every model year from 2012, the number of Teslas has are rising. 
# Although Volkswagen had their first electric car in 2013, it became first popular in Washington on 2015.

# ### Tesla Models Versus Year Of Mode

# In[37]:


yr_tesla = yr_12_22[(yr_12_22["Make"] == "TESLA")]
yr_tesla.head()


# In[38]:


yr_tesla_count = yr_tesla.groupby(["Model", "Model Year"]).size().reset_index(name="Counts")
yr_tesla_count


# In[39]:


plt.figure(figsize=(13, 7))
sns.barplot(data= yr_tesla_count, x="Model Year", y="Counts", hue="Model", palette="rocket")
plt.xlabel("Year Of Model", fontweight="bold", fontsize="14")
plt.ylabel("Counts", fontweight="bold", fontsize="14")
plt.title("Comparison Of Different Tesla Models Against Their Year Of Model", fontweight="bold", fontsize="14")
plt.legend(title="Tesla Models", loc="upper left")

plt.text(.5, 0, "Data From Data.gov (https://data.gov/)", ha="center", va="center", transform=plt.gcf().transFigure)

plt.show()


# In[ ]:




