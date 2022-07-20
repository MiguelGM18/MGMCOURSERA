### MIGUEL GONZALEZ

## here the titles to make your search easier
### *** Web Scrapping
### *** Lab 1: Collecting the data
### *** Lab 2: Data wrangling
### *** Assignment: Exploring and Preparing Data
### *** Build a Dashboard Application with Plotly Dash
### *** Assignment: Machine Learning Prediction
### Launch Sites Locations Analysis with Folium
### Assignment: SQL Notebook for Peer Assignment


### Web Scrapping
## Import requests

```
# !pip3 install beautifulsoup4
# !pip3 install requests

import sys

import requests
from bs4 import BeautifulSoup
import re
import unicodedata
import pandas as pd
```

## HTML TABLE

```

def date_time(table_cells):
    """ This function returns the data and time from the HTML  table cell
    Input: the  element of a table data cell extracts extra row """
    return [data_time.strip() for data_time in list(table_cells.strings)][0:2]

def booster_version(table_cells):
    """ This function returns the booster version from the HTML  table cell 
    Input: the  element of a table data cell extracts extra row """
    out=''.join([booster_version for i,booster_version in enumerate( table_cells.strings) if i%2==0][0:-1])
    return out

def landing_status(table_cells):
    """ This function returns the landing status from the HTML table cell 
    Input: the  element of a table data cell extracts extra row """
    out=[i for i in table_cells.strings][0]
    return out


def get_mass(table_cells):
    mass=unicodedata.normalize("NFKD", table_cells.text).strip()
    if mass:
        mass.find("kg")
        new_mass=mass[0:mass.find("kg")+2]
    else:
        new_mass=0
    return new_mass


def extract_column_from_header(row):
    """ This function returns the landing status from the HTML table cell 
    Input: the  element of a table data cell extracts extra row """
    if (row.br):
        row.br.extract()
    if row.a:
        row.a.extract()
    if row.sup:
        row.sup.extract()
        
    colunm_name = ' '.join(row.contents)
    
    # Filter the digit and empty names
    if not(colunm_name.strip().isdigit()):
        colunm_name = colunm_name.strip()
        return colunm_name 

```

```
static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"
```

```
# use requests.get() method with the provided static_url
# assign the response to a object

html_text = requests.get(static_url).text
# Use BeautifulSoup() to create a BeautifulSoup object from a response text content
soup = BeautifulSoup(html_text, 'lxml')
# Use soup.title attribute
soup.title

# Use the find_all function in the BeautifulSoup object, with element type `table`
# Assign the result to a list called `html_tables`
html_tables = soup.find_all('table')


# Let's print the third table and check its content
first_launch_table = html_tables[2]
print(first_launch_table)

column_names = []

# Apply find_all() function with `th` element on first_launch_table
# Iterate each th element and apply the provided extract_column_from_header() to get a column name
# Append the Non-empty column name (`if name is not None and len(name) > 0`) into a list called column_names

heads = soup.find_all('th')
for head in heads:
#     print(head.text)
    try:
        name = extract_column_from_header(head)
    #     print(name, '-----------------------------------------------------------')
        if name is not None and len(name) > 0:
            column_names.append(name)
    except TypeError:
        pass
```
 
```
print(column_names)
```

```
launch_dict= dict.fromkeys(column_names)

# Remove an irrelvant column
del launch_dict['Date and time ( )']

# Let's initial the launch_dict with each value to be an empty list
launch_dict['Flight No.'] = []
launch_dict['Launch Site'] = []
launch_dict['PayLoad'] = []
launch_dict['Payload mass'] = []
launch_dict['Orbit'] = []
launch_dict['Customer'] = []
launch_dict['Launch outcome'] = []
# Added some new columns
launch_dict['Version Booster']=[]
launch_dict['Booster landing']=[]
launch_dict['Date']=[]
launch_dict['Time']=[]
```

```
#To simplify the parsing process, we have provided an incomplete code snippet below to help you to fill up the launch_dict. Please complete the following code snippet with TODOs or you can choose to write your own logic to parse all launch tables:

extracted_row = 0
#Extract each table 
for table_number,table in enumerate(soup.find_all('table',"wikitable plainrowheaders collapsible")):
   # get table row 
    for rows in table.find_all("tr"):
        #check to see if first table heading is as number corresponding to launch a number 
        if rows.th:
            if rows.th.string:
                flight_number=rows.th.string.strip()
                flag=flight_number.isdigit()
        else:
            flag=False
        #get table element 
        row=rows.find_all('td')
        #if it is number save cells in a dictonary 
        if flag:
            extracted_row += 1
            # Flight Number value
            flight_number = extracted_row
            # TODO: Append the flight_number into launch_dict with key `Flight No.`
            launch_dict['Flight No.'].append(flight_number)
#             print(flight_number)
            datatimelist=date_time(row[0])
            
            # Date value
            # TODO: Append the date into launch_dict with key `Date`
            date = datatimelist[0].strip(',')
            launch_dict['Date'].append(date)
#             print(date)
            
            # Time value
            # TODO: Append the time into launch_dict with key `Time`
            time = datatimelist[1]
            launch_dict['Time'].append(time)
#             print(time)
              
            # Booster version
            # TODO: Append the bv into launch_dict with key `Version Booster`
            bv=booster_version(row[1])
            if not(bv):
                bv=row[1].a.string
#             print(bv)
            launch_dict['Version Booster'].append(bv)
            
            # Launch Site
            # TODO: Append the bv into launch_dict with key `Launch Site`
            launch_site = row[2].a.string
#             print(launch_site)
            launch_dict['Launch Site'].append(launch_site)
#             print(launch_dict['Launch Site'])
            
            # Payload
            # TODO: Append the payload into launch_dict with key `Payload`
            payload = row[3].a.string
            launch_dict['PayLoad'].append(payload)
#             print(payload)
            
            # Payload Mass
            # TODO: Append the payload_mass into launch_dict with key `Payload mass`
            payload_mass = get_mass(row[4])
            launch_dict['Payload mass'].append(payload_mass)
#             print(payload)
            
            # Orbit
            # TODO: Append the orbit into launch_dict with key `Orbit`
            orbit = row[5].a.string
            launch_dict['Orbit'].append(orbit)
#             print(orbit)
            
            # Customer
            # TODO: Append the customer into launch_dict with key `Customer`
            try:
                customer = row[6].a.string
#             print(customer)
            except AttributeError:
                customer = None
            
            launch_dict['Customer'].append(customer)
            
            # Launch outcome
            # TODO: Append the launch_outcome into launch_dict with key `Launch outcome`
            launch_outcome = list(row[7].strings)[0]
            launch_dict['Launch outcome'].append(launch_outcome)
#             print(launch_outcome)
            
            # Booster landing
            # TODO: Append the launch_outcome into launch_dict with key `Booster landing`
            booster_landing = landing_status(row[8])
            launch_dict['Booster landing'].append(booster_landing)
#             print(booster_landing)
```

```
df=pd.DataFrame(launch_dict)

df.dropna(axis=1, how='all', inplace=True)

df.to_csv('spacex_web_scraped_part.csv', index=False)
```



### Lab 1: Collecting the data

```
# Requests allows us to make HTTP requests which we will use to get data from an API
import requests
# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Datetime is a library that allows us to represent dates
import datetime

# Setting this option will print all collumns of a dataframe
pd.set_option('display.max_columns', None)
# Setting this option will print all of the data in a feature
pd.set_option('display.max_colwidth', None)
```

```
# Takes the dataset and uses the rocket column to call the API and append the data to the list
def getBoosterVersion(data):
    for x in data['rocket']:
        response = requests.get("https://api.spacexdata.com/v4/rockets/"+str(x)).json()
        BoosterVersion.append(response['name'])
```

```
# Takes the dataset and uses the launchpad column to call the API and append the data to the list
def getLaunchSite(data):
    for x in data['launchpad']:
        response = requests.get("https://api.spacexdata.com/v4/launchpads/"+str(x)).json()
        Longitude.append(response['longitude'])
        Latitude.append(response['latitude'])
        LaunchSite.append(response['name'])
```

```
# Takes the dataset and uses the payloads column to call the API and append the data to the lists
def getPayloadData(data):
    for load in data['payloads']:
        response = requests.get("https://api.spacexdata.com/v4/payloads/"+load).json()
        PayloadMass.append(response['mass_kg'])
        Orbit.append(response['orbit'])
```

```
# Takes the dataset and uses the cores column to call the API and append the data to the lists
def getCoreData(data):
    for core in data['cores']:
            if core['core'] != None:
                response = requests.get("https://api.spacexdata.com/v4/cores/"+core['core']).json()
                Block.append(response['block'])
                ReusedCount.append(response['reuse_count'])
                Serial.append(response['serial'])
            else:
                Block.append(None)
                ReusedCount.append(None)
                Serial.append(None)
            Outcome.append(str(core['landing_success'])+' '+str(core['landing_type']))
            Flights.append(core['flight'])
            GridFins.append(core['gridfins'])
            Reused.append(core['reused'])
            Legs.append(core['legs'])
            LandingPad.append(core['landpad'])
```

```
spacex_url="https://api.spacexdata.com/v4/launches/past"
response = requests.get(spacex_url)
print(response.content)
```

```
static_json_url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json'
```

```
response.status_code
```
Now we decode the response content as a Json using .json() and turn it into a Pandas dataframe using .json_normalize()
```
# Use json_normalize meethod to convert the json result into a dataframe
file = response.json()
df = pd.json_normalize(file)
df.head()
```

```
# Get the head of the dataframe
data = df.copy()
# Lets take a subset of our dataframe keeping only the features we want and the flight number, and date_utc.
data = data[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]

# We will remove rows with multiple cores because those are falcon rockets with 2 extra rocket boosters and rows that have multiple payloads in a single rocket.
data = data[data['cores'].map(len)==1]
data = data[data['payloads'].map(len)==1]

# Since payloads and cores are lists of size 1 we will also extract the single value in the list and replace the feature.
data['cores'] = data['cores'].map(lambda x : x[0])
data['payloads'] = data['payloads'].map(lambda x : x[0])

# We also want to convert the date_utc to a datetime datatype and then extracting the date leaving the time
data['date'] = pd.to_datetime(data['date_utc']).dt.date

# Using the date we will restrict the dates of the launches
data = data[data['date'] <= datetime.date(2020, 11, 13)]
```

```
#Global variables 
BoosterVersion = []
PayloadMass = []
Orbit = []
LaunchSite = []
Outcome = []
Flights = []
GridFins = []
Reused = []
Legs = []
LandingPad = []
Block = []
ReusedCount = []
Serial = []
Longitude = []
Latitude = []
```

```
# Call getBoosterVersion
getBoosterVersion(data)
```

```
BoosterVersion[0:5]
```

```
# Call getLaunchSite
getLaunchSite(data)
```

```
# Call getPayloadData
getPayloadData(data)
```

```
# Call getCoreData
getCoreData(data)
```

```
launch_dict = {'FlightNumber': list(data['flight_number']),
'Date': list(data['date']),
'BoosterVersion':BoosterVersion,
'PayloadMass':PayloadMass,
'Orbit':Orbit,
'LaunchSite':LaunchSite,
'Outcome':Outcome,
'Flights':Flights,
'GridFins':GridFins,
'Reused':Reused,
'Legs':Legs,
'LandingPad':LandingPad,
'Block':Block,
'ReusedCount':ReusedCount,
'Serial':Serial,
'Longitude': Longitude,
'Latitude': Latitude}
```

```
# Create a data from launch_dict
new_df = pd.DataFrame(launch_dict)
```

```
# Show the head of the dataframe
new_df.head()
```

```
# Hint data['BoosterVersion']!='Falcon 1'
mask = new_df['BoosterVersion']!='Falcon 1'
data_falcon9 = new_df[mask].copy()
```

```
data_falcon9.loc[:,'FlightNumber'] = list(range(1, data_falcon9.shape[0]+1))
data_falcon9
```

```
data_falcon9.isnull().sum()
```

```
# Calculate the mean value of PayloadMass column
mean = data_falcon9['PayloadMass'].mean()

# Replace the np.nan values with its mean value
data_falcon9['PayloadMass'].fillna(mean, inplace=True)
```

```
data_falcon9.to_csv('dataset_part_1.csv', index=False)
```



### Lab 2: Data wrangling

``` 
df=pd.read_csv("space_x_web_scraped_part_1")
df.head(10) 
df.isnull().sum()/df.count()*100
df.dtypes
#Use the method value_counts() on the column LaunchSite to determine the number of launches on each site:
df['LaunchSite'].value_counts()
``` 

##Each launch aims to an dedicated orbit, and here are some common orbit types:

* LEO: Low Earth orbit (LEO)is an Earth-centred orbit with an altitude of 2,000 km (1,200 mi) or less (approximately one-third of the radius of Earth),[1] or with at least 11.25 periods per day (an orbital period of 128 minutes or less) and an eccentricity less than 0.25.[2] Most of the manmade objects in outer space are in LEO [1].

* VLEO: Very Low Earth Orbits (VLEO) can be defined as the orbits with a mean altitude below 450 km. Operating in these orbits can provide a number of benefits to Earth observation spacecraft as the spacecraft operates closer to the observation[2].

* GTO A geosynchronous orbit is a high Earth orbit that allows satellites to match Earth's rotation. Located at 22,236 miles (35,786 kilometers) above Earth's equator, this position is a valuable spot for monitoring weather, communications and surveillance. Because the satellite orbits at the same speed that the Earth is turning, the satellite seems to stay in place over a single longitude, though it may drift north to south,” NASA wrote on its Earth Observatory website [3] .

* SSO (or SO): It is a Sun-synchronous orbit also called a heliosynchronous orbit is a nearly polar orbit around a planet, in which the satellite passes over any given point of the planet's surface at the same local mean solar time [4] .

* ES-L1 :At the Lagrange points the gravitational forces of the two large bodies cancel out in such a way that a small object placed in orbit there is in equilibrium relative to the center of mass of the large bodies. L1 is one such point between the sun and the earth [5] .

* HEO A highly elliptical orbit, is an elliptic orbit with high eccentricity, usually referring to one around Earth [6].

* ISS A modular space station (habitable artificial satellite) in low Earth orbit. It is a multinational collaborative project between five participating space agencies: NASA (United States), Roscosmos (Russia), JAXA (Japan), ESA (Europe), and CSA (Canada) [7]

* MEO Geocentric orbits ranging in altitude from 2,000 km (1,200 mi) to just below geosynchronous orbit at 35,786 kilometers (22,236 mi). Also known as an intermediate circular orbit. These are "most commonly at 20,200 kilometers (12,600 mi), or 20,650 kilometers (12,830 mi), with an orbital period of 12 hours [8]

* HEO Geocentric orbits above the altitude of geosynchronous orbit (35,786 km or 22,236 mi) [9]

* GEO It is a circular geosynchronous orbit 35,786 kilometres (22,236 miles) above Earth's equator and following the direction of Earth's rotation [10]

* PO It is one type of satellites in which a satellite passes above or nearly above both poles of the body being orbited (usually a planet such as the Earth [11]

some are shown in the following plot:

[![Orbits.png](https://i.postimg.cc/k5Vj93bg/Orbits.png)](https://postimg.cc/CZgCH273)

True Ocean means the mission outcome was successfully landed to a specific region of the ocean while False Ocean means the mission outcome was unsuccessfully landed to a specific region of the ocean. True RTLS means the mission outcome was successfully landed to a ground pad False RTLS means the mission outcome was unsuccessfully landed to a ground pad.True ASDS means the mission outcome was successfully landed to a drone ship False ASDS means the mission outcome was unsuccessfully landed to a drone ship. None ASDS and None None these represent a failure to land.

```
for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)
    
#We create a set of outcomes where the second stage did not land successfully:
    bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
    
#Create a landing outcome label from Outcome column
landing_class=[0 if outcome in bad_outcomes else 1 for outcome in df['Outcome']]
df['Class']=landing_class
df[['Class']].head(8)
df.head(5)

#We can use the following line of code to determine the success rate:
df["Class"].mean()
df.to_csv("dataset_part_2.csv", index=False)
```


### Assignment: Exploring and Preparing Data

#First, let's try to see how the FlightNumber (indicating the continuous launch attempts.) and Payload variables would affect the launch outcome.

#We can plot out the FlightNumber vs. PayloadMassand overlay the outcome of the launch. We see that as the flight number increases, the first stage is more likely to land successfully. The payload mass is also important; it seems the more massive the payload, the less likely the first stage will return.**

```
sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
plt.show()
```

#We see that different launch sites have different success rates. CCAFS LC-40, has a success rate of 60 %, while KSC LC-39A and VAFB SLC 4E has a success rate of 77%.

```
#Plot a scatter point chart with x axis to be Flight Number and y axis to be the launch site, and hue to be the class value
sns.catplot(y="LaunchSite", x="FlightNumber", hue="Class", data=df, aspect = 5)

##Visualize the relationship between Flight Number and Launch Site
#Plot a scatter point chart with x axis to be Pay Load Mass (kg) and y axis to be the launch site, and hue to be the class value
sns.catplot(y="LaunchSite", x="PayloadMass", hue="Class", data=df, aspect = 5)

##Next, we want to visually check if there are any relationship between success rate and orbit type.
#HINT use groupby method on Orbit column and get the mean of Class column
df.groupby("Orbit")["Class"].mean().plot(kind= 'bar', legend= 'reverse')

##For each orbit, we want to see if there is any relationship between FlightNumber and Orbit type.
#Plot a scatter point chart with x axis to be FlightNumber and y axis to be the Orbit, and hue to be the class value
sns.catplot(y="FlightNumber", x="Orbit", hue="Class", data=df, aspect = 5)

##Visualize the relationship between Payload and Orbit type
#Plot a scatter point chart with x axis to be Payload and y axis to be the Orbit, and hue to be the class value
sns.catplot(y="PayloadMass", x="Orbit", hue="Class", data=df, aspect = 5)

##You can plot a line chart with x axis to be Year and y axis to be average success rate, to get the average launch success trend.
# Plot a line chart with x axis to be the extracted year and y axis to be the success rate
df.groupby(("Date"))["Class"].mean().plot(kind= 'bar', legend= 'reverse')

##By now, you should obtain some preliminary insights about how each important variable would affect the success rate, we will select the features that will be used in success prediction in the future module.
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()

##Create dummy variables to categorical columns
#HINT: Use get_dummies() function on the categorical columns
features_one_hot = pd.get_dummies(features["Orbits","LaunchSite","LandingPad","Serial"], axis = ["Orbits","LaunchSite","LandingPad","Serial"])

##Cast all numeric columns to float64
#HINT: use astype function
features_one_hot = float(features_one_hot.astype)

features_one_hot.to_csv('dataset_part_3.csv', index=False)
```


### Build a Dashboard Application with Plotly Dash

```
# Import required libraries
import pandas as pd
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
import wget
```

```
# Uncomment the following cell to get download skeleton app and dataset:

# spacex_dataset = wget.download("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv")
# spacex_dash_app = wget.download("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_3/spacex
```

```
# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)
server = app.server

# Create an app layout

app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                html.P("Created by Molo Munyansanga", style={'textAlign': 'center'}),
                                
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                # dcc.Dropdown(id='site-dropdown',...)
                                dcc.Dropdown(id='site-dropdown',
                                                options=[
                                                    {'label': 'All Sites', 'value': 'ALL'},
                                                    {'label': 'CCAFS LC-40', 'value': 'CCAFS LC-40'},
                                                    {'label': 'VAFB SLC-4E', 'value': 'VAFB SLC-4E'},
                                                    {'label': 'KSC LC-39A', 'value': 'KSC LC-39A'},
                                                    {'label': 'CCAFS SLC-40', 'value': 'CCAFS SLC-40'},
                                                ],
                                                value='ALL',
                                                placeholder="Select a Launch Site here",
                                                searchable=True
                                                ),

                                html.Br(),

                                # Task 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected,
                                # show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                # Add payload mass slider text
                                html.P(id="slider-text"),
                                
                                # TASK 3: Add a slider to select payload range
                                # dcc.RangeSlider(id='payload-slider',...)
                                dcc.RangeSlider(id='payload-slider',
                                                min=0, max=10000, step=1000,
                                                marks={0: '0', 2500: '2500', 5000: '5000', 7500: '7500',
                                                       10000: '10000'},
                                                value=[min_payload, max_payload]),

                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart'))
                                ])


# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
@app.callback(Output(component_id='success-pie-chart', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'))
def get_pie_chart(entered_site):
    filtered_df = spacex_df
    if entered_site == 'ALL':
        fig = px.pie(filtered_df, values='class', names='Launch Site', title='Total Successful Launches By Site')
        return fig
    else:
        # return the outcomes piechart for a selected site
        site_chosen = entered_site
        mask = filtered_df['Launch Site'] == site_chosen
        fig = px.pie(filtered_df[mask], names='class',
                     title=f'Total Successful Launches For Site {site_chosen}')
        return fig


# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(Output(component_id='success-payload-scatter-chart', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'),
              [Input(component_id='payload-slider', component_property='value')])
def get_scatter_chart(entered_site, mass):

    # filter masses from payload slider
    mass_1 = spacex_df['Payload Mass (kg)'] >= float(mass[0])
    mass_2 = spacex_df['Payload Mass (kg)'] <= float(mass[1])
    
    filtered_df = spacex_df[mass_1][mass_2]
    
    if entered_site == 'ALL':

        fig = px.scatter(filtered_df, x='Payload Mass (kg)', y='class', color="Booster Version Category",
                         title=f'Correlation between Payload Mass and Launch Success for All Sites for Payload Mass(kg) Between {mass[0]} and {mass[1]}')
        return fig
    else:
        
        # return the outcomes scatter chart for a selected site
        site_chosen = entered_site
        mask = filtered_df['Launch Site'] == site_chosen
        fig = px.scatter(filtered_df[mask], x='Payload Mass (kg)', y='class', color="Booster Version Category",
                         title=f'Correlation between Payload Mass and Launch Success for Site {site_chosen}')
        return fig
    
    
#function to return payload_mass success_rate
@app.callback(Output('slider-text', 'children'),
              [Input(component_id='payload-slider', component_property='value')])
def get_success_rate(mass):

    # filter masses from payload slider
    mass_1 = spacex_df['Payload Mass (kg)'] >= float(mass[0])
    mass_2 = spacex_df['Payload Mass (kg)'] <= float(mass[1])
    
    filtered_df = spacex_df[mass_1][mass_2]
    
    # find success rate
    rate = (filtered_df['class'].value_counts().loc[1])/filtered_df['class'].value_counts().sum() * 100
    rate = 'Payload range (Kg): ' + str(round(rate, 2)) + '% Success Rate'
    
    return rate
    
    
# Run the app
if __name__ == '__main__':
    app.run_server()
```



### Assignment: Machine Learning Prediction

```
# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier
```

```
def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed'])
```

```
# data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")

# If you were unable to complete the previous lab correctly you can uncomment and load this csv

data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/dataset_part_2.csv')

data.head()
data
```

```
# X = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv')

# If you were unable to complete the previous lab correctly you can uncomment and load this csv

X = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/dataset_part_3.csv')

X.head(100)
```

```
y = data['Class'].to_numpy()
```

```
# students get this 
transform = preprocessing.StandardScaler()
x = transform.fit(X).transform(X)
```

```
# split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( x, y, test_size=0.2, random_state=2)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)
```

```
Y_test.shape
```

```
parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}             
```

```
parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge

lr=LogisticRegression() # Logistic regression object

logreg_cv = GridSearchCV(estimator=lr, cv=10, param_grid=parameters).fit(X_train, Y_train) # GridSearchCV object that is fitted
```

```
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)
```

```
logreg_score = logreg_cv.score(X_test, Y_test)
print("score :",logreg_score)
```

```
yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)
```

```
parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}

svm = SVC() # Support vector machine object

svm_cv = GridSearchCV(estimator=svm, cv=10, param_grid=parameters).fit(X_train, Y_train) # GridSearchCV object that is fitted
```


```
print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)
```

```
svm_cv_score = svm_cv.score(X_test, Y_test)

print("score :",svm_cv_score)
```

```
yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)
```

```
parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier() # decision tree classifier object

tree_cv = GridSearchCV(estimator=tree, cv=10, param_grid=parameters).fit(X_train, Y_train) # GridSearchCV object that is fitted
```

```
print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)
```

```
tree_cv_score = tree_cv.score(X_test, Y_test)

print("score :",tree_cv_score)
```

```
yhat = svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)
```

```
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier() #  k nearest neighbors object

knn_cv = GridSearchCV(estimator=KNN, cv=10, param_grid=parameters).fit(X_train, Y_train) # GridSearchCV object that is fitted
```

```
print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
print("accuracy :",knn_cv.best_score_)
```

```
knn_cv_score = knn_cv.score(X_test, Y_test)
print("score :",knn_cv_score)
```

```
yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)
```

```
# Created a dataframe showing accuracy scores:
accuracy = [svm_cv_score, logreg_score, knn_cv_score, tree_cv_score]
accuracy = [i * 100 for i in accuracy]

method = ['Support Vector Machine', 'Logistic Regression', 'K Nearest Neighbour', 'Decision Tree']
models = {'ML Method':method, 'Accuracy Score (%)':accuracy}

ML_df = pd.DataFrame(models)
ML_df
```

```
# Plot bar chart
ML_df.plot(kind='bar', x='ML Method', y='Accuracy Score (%)', ylabel='Accuracy (%)', figsize=(10,10), 
           legend=None, rot= 1, title='Machine Learning Method Accuracy');
```

```
# Using Logistic regression and best parameters we received earlier:
# 'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs'
LR = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs').fit(X_train, Y_train)
```

```
# prediction
yhat = LR.predict(X_test)
```

```
plt.figure(figsize=(8,7))
plot_confusion_matrix(Y_test,yhat)
```



### Launch Sites Locations Analysis with Folium

```
import folium
import wget
import pandas as pd
import numpy as np
```

```
# Import folium MarkerCluster plugin
from folium.plugins import MarkerCluster
# Import folium MousePosition plugin
from folium.plugins import MousePosition
# Import folium DivIcon plugin
from folium.features import DivIcon
```

```
# Download and read the `spacex_launch_geo.csv`
spacex_csv_file = wget.download('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv')
spacex_df=pd.read_csv(spacex_csv_file)
```

```
# Select relevant sub-columns: `Launch Site`, `Lat(Latitude)`, `Long(Longitude)`, `class`
spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]
launch_sites_df
```

```
# Start location is NASA Johnson Space Center
nasa_coordinate = [29.559684888503615, -100]
site_map = folium.Map(location=nasa_coordinate, zoom_start=4.5)
```

```
# Create a blue circle at NASA Johnson Space Center's coordinate with a popup label showing its name
circle = folium.Circle(nasa_coordinate, radius=10, color='#0000FF', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))
# Create a blue circle at NASA Johnson Space Center's coordinate with a icon showing its name
marker = folium.map.Marker(
    nasa_coordinate,
    # Create an icon as a text label
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#000000;"><b>%s</b></div>' % 'NASA JSC',
        )
    )
site_map.add_child(circle)
site_map.add_child(marker)
```

```
col = launch_sites_df.columns

# save coordinates in lists
sites = [v for k,v in launch_sites_df[col[0]].items()]
lat = [v for k,v in launch_sites_df[col[1]].items()]
long = [v for k,v in launch_sites_df[col[2]].items()]
circles = []

for i in range(len(sites)):
    coordinate = [lat[i], long[i]]
    # circle objects
    circle = folium.Circle(coordinate, radius=100, color='#000000', fill=True).add_child(folium.Popup(f'{sites[i]}'))
    
    # marker objects
    marker = folium.map.Marker(coordinate, icon=DivIcon(icon_size=(20,20),icon_anchor=(0,0), 
                                               html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % f'{sites[i]}', ))
    site_map.add_child(circle)
    site_map.add_child(marker)
    
site_map
```

```
spacex_df.tail(10)
marker_cluster = MarkerCluster()
```

```
# Function to assign color to launch outcome
def assign_marker_color(launch_outcome):
    if launch_outcome == 1:
        return 'green'
    else:
        return 'red'
    
spacex_df['marker_color'] = spacex_df['class'].apply(assign_marker_color)
spacex_df.tail(10)
```

```
# Add marker_cluster to current site_map
site_map.add_child(marker_cluster)

# for each row in spacex_df data frame
# create a Marker object with its coordinate
# and customize the Marker's icon property to indicate if this launch was successed or failed, 
# e.g., icon=folium.Icon(color='white', icon_color=row['marker_color']
for index, record in spacex_df.iterrows():
    
    label = record[0]
    lat = record[1]
    long = record[2]
    row_color = record[4]
    
    # TODO: Create and add a Marker cluster to the site map
    marker = folium.Marker(
        location = [lat, long],
#         popup = label,
        icon=folium.Icon(color='white', icon_color=row_color)
    )
    marker_cluster.add_child(marker)

site_map.fit_bounds([[28.60577, -80.68102], [28.52813, -80.52168]]) # Insert resting coordinates for map
site_map
```

```
# Add Mouse Position to get the coordinate (Lat, Long) for a mouse over on the map
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN',
    lng_first=False,
    num_digits=20,
    prefix='Lat:',
    lat_formatter=formatter,
    lng_formatter=formatter,
)

site_map.add_child(mouse_position)
site_map.fit_bounds([[42.81152, -130.3418], [13.32548, -64.33594]])
site_map
```

```
from math import sin, cos, sqrt, atan2, radians

def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance
```

```
# find coordinate of the closet coastline
# e.g.,: Lat: 28.56367  Lon: -80.57163
# distance_coastline = calculate_distance(launch_site_lat, launch_site_lon, coastline_lat, coastline_lon)

coastline_distances = []

lat = [v for k,v in launch_sites_df[col[1]].items()]
long = [v for k,v in launch_sites_df[col[2]].items()]

coast_lat = [28.56225, 28.56334, 28.57339, 34.63284]
coast_long = [-80.5678, -80.56797, -80.60693, -120.62634]


for i in range(len(coast_long)):
    coastline_distances.append(calculate_distance(lat[i], long[i], coast_lat[i], coast_long[i]))
```

```
# Create and add a folium.Marker on your selected closest coastline point on the map
# Display the distance between coastline point and launch site using the icon property 

for i in range(len(coast_long)):
    
    # nearest coastline coordinates
    coordinate = [coast_lat[i], coast_long[i]]
    
    # marker              
    distance_marker = folium.Marker(
       coordinate,
       icon=DivIcon(
           icon_size=(20,20),
           icon_anchor=(0,0),
           html='<div style="font-size: 12; color:#000080;"><b>%s</b></div>' % "{:10.2f} KM".format(coastline_distances[i]),
           )
       )
    
    site_map.add_child(distance_marker)
```

```
# Create a `folium.PolyLine` object using the coastline coordinates and launch site coordinate

for i in range(len(lat)):

    # coordinate
    coordinates = [[lat[i], long[i]], [coast_lat[i], coast_long[i]]]

    lines=folium.PolyLine(coordinates, weight=0.5)
    site_map.add_child(lines)

site_map.fit_bounds([[28.56798, -80.58229], [28.55791, -80.56224]])
site_map
```

```
# enter coordinates
locations = []
locs_distances = []

railway_lat = [28.57211, 28.57211, 28.57307, 34.6338]
railway_long = [-80.58527, -80.58527, -80.65403, -120.62478]

highway_lat = [28.56278, 28.56278, 28.57359, 34.70472]
highway_long = [-80.57075, -80.57075, -80.6553, -120.56937]

city_lat = [28.39804, 28.39804, 28.60984, 34.63716]
city_long = [-80.60257, -80.60257, -80.80376, -120.45616]

for i in range(len(city_lat)):
    
    # save coordinates in a list object
    locs = [railway_lat[i], railway_long[i], '#BA55D3'], [highway_lat[i], highway_long[i], '#2E8B57'], [city_lat[i], city_long[i], '#DC143C']
    locations.append([locs])
    
    # save distances(km) in another list
    dist = [(calculate_distance(lat[i], long[i], railway_lat[i], railway_long[i])), 
    (calculate_distance(lat[i], long[i], highway_lat[i], highway_long[i])),
    (calculate_distance(lat[i], long[i], city_lat[i], city_long[i]))]
    
    locs_distances.append(dist)
```

```
# Create a marker with distance to a closest city, railway, highway, etc.

# loop through the location items for each launch site
for i in range(len(locations)):
    
    # site_items 
    site = locations[i]
    
    # site coordinates
    site_coordinates = [lat[i], long[i]]
    
    # loop through location item
    for i in range(len(site)):
    
        # nearest coordinates
        coordinates = site[i]
        
        # distance
        distances = locs_distances[i]
        
        # loop through each distance
        for i in range(len(distances)):
            
            # individual coordinate
            coordinate = coordinates[i][:2]
            
            # individual distance
            distance = distances[i]

#             print(coordinate)
#             print(distance)
#             break
            
            # marker              
            distance_marker = folium.Marker(
               coordinate,
               icon=DivIcon(
                   icon_size=(20,20),
                   icon_anchor=(0,0),
                   html=f'<div style="font-size: 5; color:{coordinates[i][2]};"><b>%s</b></div>' % "{:10.2f} KM".format(distance),
                   )
               )

            
            # Draw a line between the marker to the launch site
            
            # coordinates from launch site to marker
            new_coordinates = [site_coordinates, coordinate]

            line = folium.PolyLine(new_coordinates, weight=0.5)

            site_map.add_child(line)
            site_map.add_child(distance_marker)
        

site_map.fit_bounds([[34.64242, -120.63443], [34.6229, -120.59301]])
site_map
```


### Assignment: SQL Notebook for Peer Assignment


```
# !pip install sqlalchemy==1.3.9
# !pip install ibm_db_sa
# !pip install ipython-sql
```

```
import sqlite3
import sql
import pandas as pd
```

```
%load_ext sql
```

```
%sql sqlite:///SpaceEx.sqlite
```

```
# %sql ibm_db_sa://
```

```
%%sql

SELECT Launch_Site, COUNT(DISTINCT(Launch_Site))
from Spacex
GROUP BY Launch_Site
```

```
%%sql

SELECT *
from Spacex
where Launch_Site like 'CCA%'
Limit 5
```

```
%%sql

select Customer, SUM(PAYLOAD_MASS__KG_) as Total_Payload_Mass
FROM Spacex
GROUP BY Customer
HAVING Customer LIKE 'NASA%'
ORDER BY Total_Payload_Mass DESC
```

```
%%sql

select AVG(PAYLOAD_MASS__KG_) as Average_Payload_Mass, Booster_Version
FROM Spacex
where Booster_Version == 'F9 v1.1'
```

```
%%sql

select Date, Landing_Outcome
FROM Spacex
where Landing_Outcome = 'Success (ground pad)'
limit 1
```

```
%%sql

select Booster_Version, PAYLOAD_MASS__KG_, Landing_Outcome
FROM Spacex
where Landing_Outcome = 'Success (drone ship)'
    and PAYLOAD_MASS__KG_ > 4000
    and PAYLOAD_MASS__KG_ < 6000
```

```
%%sql


select Mission_Outcome, COUNT(Mission_Outcome) as Outcomes
from Spacex
GROUP BY(Mission_Outcome)
```

```
%%sql

select Booster_Version, PAYLOAD_MASS__KG_
FROM Spacex
where PAYLOAD_MASS__KG_ = (
            select MAX(PAYLOAD_MASS__KG_)
            from Spacex
);
```

```
%%sql

SELECT Date, Launch_Site, Booster_Version, Landing_Outcome
from Spacex
where Date LIKE '%2015'
    and Landing_Outcome LIKE 'Failure%'
```

```
%%sql


SELECT substr(Date,7,4) || '-' || substr(Date,4,2) || '-' || substr(Date,1,2) || substr(Date,11) as _date_, 
Landing_Outcome, COUNT(Landing_Outcome) as Outcomes
from Spacex
GROUP BY(Landing_Outcome)
Having (Date(_date_) > '2010-06-04' and Date(_date_) < '2017-03-20')
ORDER BY DATE(_date_) DESC
```
