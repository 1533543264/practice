#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import into storage
import pandas as pd #Use pd's dataframe to process CSV files
from datetime import datetime #Use the time format datetime to process dates in the data
import numpy as np #Use numpy to calculate the price of the item


# In[ ]:


#read dirty_data csv
dirty_data = pd.read_csv("32891059_dirty_data.csv")


# In[ ]:


#Look at the content of the data to determine errors
dirty_data.info()


# In[ ]:


#latest_customer_review was found to have a NAN value but the data is correct according to the job requirements
dirty_data.loc[dirty_data['latest_customer_review'].isnull()]


# # Look for errors in date

# In[ ]:


#Look for the wrong date
wrong_date = []
for _,row in dirty_data.iterrows():
    try:
        datetime.strptime(row['date'],'%Y-%m-%d')
        continue
    except ValueError:
        wrong_date.append(row['date'])


# In[ ]:


#The wrong date was found
wrong_date


# In[ ]:


#Discover their styles based on the wrong date
date_formats = ['%Y-%m-%d','%Y-%d-%m','%Y-%b-%d']


# In[ ]:


#Convert the date format using the datetime.strptime method
def changedateformat(x):
    for dateformat in date_formats:
        try:
            return datetime.strptime(x,dateformat).strftime('%Y-%m-%d')
        except ValueError:
            pass
    return None


# In[ ]:


#Loop the date in dirty_data and replace the contents of the error
dirty_data['date'] = dirty_data['date'].apply(lambda x: changedateformat(x))


# In[ ]:


#Look for the wrong date
wrong_date = []
for _,row in dirty_data.iterrows():
    try:
        datetime.strptime(row['date'],'%Y-%m-%d')
        continue
    except ValueError:
        wrong_date.append(row['date'])


# In[ ]:


#There are no wrong dates,the modification is successful
wrong_date


# # Correct the wrong latitude and longitude

# In[ ]:


#Find the wrong latitude and longitude
dirty_data.loc[(dirty_data['customer_lat']>0) | (dirty_data['customer_long']<0)]


# In[ ]:


#Generates an array of error line numbers
wrong_index = dirty_data.loc[(dirty_data['customer_lat']>0) | (dirty_data['customer_long']<0)].index.tolist()


# In[ ]:


#Gets the line number of the error line
wrong_index


# In[ ]:


#The data obtained should be interchangeable with latitude and longitude
#The to_numpy method is used to interchange the values of the customer_lat and customer_long columns
dirty_data.loc[wrong_index, ['customer_lat','customer_long']] = dirty_data.loc[wrong_index, ['customer_long','customer_lat']].to_numpy()


# In[ ]:


#Check whether the data is modified successfully
dirty_data.loc[(dirty_data['customer_lat']>0) | (dirty_data['customer_long']<0)]


# # Calculate the distance and correct the wrong distance

# In[ ]:


#Read the latitude and longitude files for each warehouse
warehouses = pd.read_csv('warehouses.csv')
warehouses


# In[ ]:


#Import the math library
from math import radians, sin,cos, asin, sqrt
#The function calculates the distance between two latitudes and longitudes
#Enter the latitude and longitude of the two locations
#Output the calculated distance
def calculate_distance(lat1,lon1,lat2,lon2):
    # Convert the Angle to radians
    lon1,lat1,lon2,lat2 = map(radians, [float(lon1),float(lat1),float(lon2),float(lat2)])
    # Difference between latitude and longitude
    difference_lon = lon2 -lon1
    difference_lat = lat2 -lat1
    # Haversine formula
    spherical_distance_square = sin(difference_lat/2)**2+cos(lat1)*cos(lat2)*sin(difference_lon/2)**2
    actual_distance = 2*asin(sqrt(spherical_distance_square))
    # Mean radius of the Earth 
    radius = 6378
    # Return distance
    return actual_distance *radius


# In[ ]:


#This function is used to compute the nearest warehouse
#Enter customer_location consisting of customer_lat and customer_long
#output closest_distance, closest_warehouse
def closest_warehouse(customer_location):
    #Get the customer's latitude and longitude coordinates
    lat, lon = customer_location
    #Calculate the name and distances of warehouse nearest to customer based on latitude and longitude coordinates of warehouses and store the result in warehouse_distances list
    warehouse_distances = [(calculate_distance(lat, lon, row['lat'], row['lon']), row['names'])
                           for _, row in warehouses.iterrows()]
    # Find the name and distance of warehouse closest to warehouse_distances and assign the result to closest_distance and closest_warehouse respectively
    closest_distance, closest_warehouse = min(warehouse_distances, key=lambda x: x[0])
    # Returns the name and distance of the nearest warehouse
    return closest_distance, closest_warehouse


# In[ ]:


#Do the calculation with closest_warehouse and store the returned result in the calculated_distance and calculated_nearest_warehouse columns
dirty_data[['calculated_distance','calculated_nearest_warehouse']] = dirty_data.apply(lambda row: pd.Series(closest_warehouse((row['customer_lat'],row['customer_long']))),axis =1)
#Keep the calculated distance to 5 decimal places
dirty_data['calculated_distance'] = dirty_data['calculated_distance'].round(5)


# In[ ]:


#Check data to see whether the calculation and addition are successful
dirty_data.head()


# In[ ]:


#See how nearest_warehouse differs from calculated data
dirty_data.loc[dirty_data['nearest_warehouse'] != dirty_data['calculated_nearest_warehouse']]


# In[ ]:


#See how distance_to_nearest_warehouse differs from the calculated data
dirty_data.loc[dirty_data['distance_to_nearest_warehouse'] != dirty_data['calculated_distance']]


# In[ ]:


#Modify and replace data
dirty_data['distance_to_nearest_warehouse'] = dirty_data['calculated_distance']
dirty_data['nearest_warehouse'] = dirty_data['calculated_nearest_warehouse']


# In[ ]:


#Check whether the modification is successful
dirty_data.loc[dirty_data['distance_to_nearest_warehouse'] != dirty_data['calculated_distance']]


# In[ ]:


#Check whether the modification is successful
dirty_data.loc[dirty_data['nearest_warehouse'] != dirty_data['calculated_nearest_warehouse']]


# In[ ]:


#Delete redundant data columns
dirty_data.drop(labels = ['calculated_distance','calculated_nearest_warehouse'],axis=1,inplace = True)


# In[ ]:


#Check whether the current data is complete
dirty_data.head()


# # Calculate the item and price and modify the shopping_cart order_price

# In[ ]:


#Read the csv file for miss_data
miss_data = pd.read_csv("32891059_missing_data.csv")


# In[ ]:


#View data
miss_data.info()


# In[ ]:


#Calculating commodity prices
#Input miss
#The shop_mapping is displayed
def calculate_prices(miss):
    #Convert the shopping cart into a list with each element as a list containing items and quantities
    shopping_cart = miss['shopping_cart'].apply(eval).tolist()
    #Create an all-zero matrix with rows equal to the number of items
    shop_list = np.zeros((len(shopping_cart),10))
    #Record the index of the item in the matrix
    item_indices = {}
    count = 0
    for i,cart in enumerate(shopping_cart):
        for item,quantity in cart:
            if item not in item_indices:
                # If the item is not numbered, a new number is assigned
                item_indices[item] = count
                count += 1
            # Record the number of item
            item_id = item_indices[item]
            shop_list[i,item_id] = quantity
    # Get order_price values
    order_price = miss['order_price'].values
    #Use linear regression to solve commodity prices
    item_price = np.linalg.lstsq(shop_list,order_price,rcond=None)
    # Build a mapping between commodity name and price
    shop_mapping ={}
    for item,index in item_indices.items():
        shop_mapping[item] = item_price[0][index]
    return shop_mapping


# In[ ]:


#Get variety and price
shop_mapping = calculate_prices(miss_data)


# In[ ]:


#View shop_mapping
shop_mapping


# In[ ]:


# Calculate the total price
# Enter row and shop_mapping
# # Output cart_total
def calculate_cart_prices(row,shop_mapping):
    cart_items = eval(row['shopping_cart'])
    cart_total = 0
    for item,quantity in cart_items:
        cart_total += shop_mapping[item] * quantity
    return round(cart_total)


# In[ ]:


# Calculate and store the result to cart_total
dirty_data['cart_total'] = dirty_data.apply(calculate_cart_prices,axis = 1,args =(shop_mapping,))


# In[ ]:


# View data
dirty_data.head()


# In[ ]:


## Store the rows different from cart_total and order_price as not_equal_rows and display not_equal_rows
not_equal_rows = dirty_data.loc[dirty_data['order_price']!=dirty_data['cart_total']]
not_equal_rows


# In[ ]:


def find_commodity(products, total_price,quantity_list):
    """
        In the given list of items, find a combination of four items so that the total price equals the given price.
        input products, total_price,quantity_list
        Returns the found combination of items or none
    """
    # Store the product and price correspondence in the dictionary
    price_dict = {str(k): int(round(v)) for k, v in products.items()}
    # Go through all the possible combinations
    for p1, price1 in price_dict.items():
        for p2, price2 in price_dict.items():
            for p3, price3 in price_dict.items():
                for p4, price4 in price_dict.items():
                    if quantity_list[0]*price1 + quantity_list[1]*price2 + quantity_list[2]*price3 + quantity_list[3]*price4 == total_price:
                        if quantity_list[1] == 0:
                            return [(p1,quantity_list[0])]
                        elif quantity_list[2] == 0:
                            return [(p1,quantity_list[0]),(p2,quantity_list[1])]
                        elif quantity_list[3] == 0:
                            return [(p1,quantity_list[0]),(p2,quantity_list[1]),(p3,quantity_list[2])]
                        else:
                            return [(p1,quantity_list[0]),(p2,quantity_list[1]),(p3,quantity_list[2]),(p4,quantity_list[3])]

    # If no suitable combination is found, None is returned
    return None


# In[ ]:


# Convert shopping_cart and order_price into lists
shopping_cart_list = not_equal_rows['shopping_cart'].tolist()
order_price_list = not_equal_rows['order_price'].tolist()
result_list = []
# Extract the number after each comma
for i in range(len(shopping_cart_list)):
    product_list = eval(shopping_cart_list[i])
    quantity_list = []
    for product in product_list:
        quantity_list.append(product[1])
    # Complete the list of four items with zeros
    quantity_list += [0] * (4 - len(quantity_list))
    # Store the returned value into the result_list
    result_list.append(find_commodity(shop_mapping, order_price_list[i],quantity_list))


# In[ ]:


# not_equal_rows traverses the loop
for i, (index, row) in enumerate(not_equal_rows.iterrows()):
    #Determine whether the result_list[i] is null
    if result_list[i] is not None:
        #If there is a suitable number of items corresponding to the value of order_price modify shopping_cart according to the index index
        dirty_data.at[index, 'shopping_cart'] = str(result_list[i])
    else:
        #If there is no appropriate number of items corresponding to the value of order_price modify shopping_cart according to the index index
        dirty_data.at[index, 'order_price'] = dirty_data.at[index, 'cart_total']


# In[ ]:


# Calculate the cart_total of the changed shipping_cart
dirty_data['cart_total'] = dirty_data.apply(calculate_cart_prices,axis = 1,args =(shop_mapping,))
# Check whether there are different values
not_equal_rows = dirty_data.loc[dirty_data['order_price']!=dirty_data['cart_total']]
not_equal_rows


# # Fixed season error

# In[ ]:


# The date column is converted to the date-time format
dirty_data['date'] = pd.to_datetime(dirty_data['date'])


# In[ ]:


# input date
# output season or none
# Calculate the season corresponding to the date
def determine_season(date):
    season_dict = {'Spring': [9, 10, 11], 'Summer': [12, 1, 2], 'Autumn': [3, 4, 5], 'Winter': [6, 7, 8]}
    # Obtain the seasons according to the traversal loop
    for season, months in season_dict.items():
        if date.month in months:
            return season
    return None


# In[ ]:


# Store the returned seasons in the cal_season column
dirty_data['cal_season'] = dirty_data['date'].apply(determine_season)


# In[ ]:


# Check errors
not_equal_rows = dirty_data.loc[dirty_data['season']!=dirty_data['cal_season']]
not_equal_rows


# In[ ]:


# Correction error
dirty_data['season'] = dirty_data['cal_season']


# In[ ]:


# Check whether the modification is successful and whether errors exist
not_equal_rows = dirty_data.loc[dirty_data['season']!=dirty_data['cal_season']]
not_equal_rows


# In[ ]:


# Delete the cal_season column
dirty_data.drop(labels = ['cal_season'],axis=1,inplace = True)


# # Calculate and modify is_happy_customer

# In[ ]:


# Import the SentimentIntensityAnalyzer class, scores used to calculate the text sentiment analysis
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# In[ ]:


# Create a SentimentIntensityAnalyzer class variables
sia = SentimentIntensityAnalyzer()


# In[ ]:


#input review
#output True or False(sia.polarity_scores(review)['compound'] >= 0.05)
#Calculate if the customer is happy
def is_happy_customer(review):
    # Determines if review is a string, and returns True if it is not
    if not isinstance(review, str):
        return True
    return sia.polarity_scores(review)['compound'] >= 0.05


# In[ ]:


# Store the return value in the cal_is_happy_customer column
dirty_data['cal_is_happy_customer'] = dirty_data['latest_customer_review'].apply(is_happy_customer)


# In[ ]:


# Check whether there are any error values
dirty_data.loc[dirty_data['cal_is_happy_customer'] != dirty_data['is_happy_customer']]


# In[ ]:


# Modify and check whether the modification is successful
dirty_data['is_happy_customer'] = dirty_data['cal_is_happy_customer']
dirty_data.loc[dirty_data['cal_is_happy_customer'] != dirty_data['is_happy_customer']]


# In[ ]:


#Delete the cal_is_happy_customer column
dirty_data.drop(labels = ['cal_is_happy_customer'],axis=1,inplace = True)


# # Error modifying missing_data.csv file

# In[ ]:


# read 32891059_missing_data.csv
miss_data = pd.read_csv("32891059_missing_data.csv")


# In[ ]:


# Calculate the distance and name of the warehouse closest to the customer's address and substitute
miss_data = miss_data.assign(
    distance_to_nearest_warehouse = miss_data.apply(
        lambda row: round(closest_warehouse((row['customer_lat'],row['customer_long']))[0], 5),
        axis=1
    ),
    nearest_warehouse = miss_data.apply(
        lambda row: closest_warehouse((row['customer_lat'],row['customer_long']))[1],
        axis=1
    )
)


# In[ ]:


# Calculation of true freight
def calculate_true_freight(row):
    return round(row['delivery_charges'] / (1 - row['delivery_discount'] / 100), 3)


# In[ ]:


#Store the return value into cal_price
miss_data['cal_price']=miss_data.apply(calculate_true_freight,axis=1)


# In[ ]:


# Convert bool to int
# input bool_value
#output int 1 0
def replace_bool_int(bool_value):
    if bool_value == True:
        return 1
    elif bool_value == False:
        return 0
    else:
        return bool_value


# In[ ]:


#Converts Boolean values in miss_data to int values 0 and 1
miss_data = miss_data.apply(lambda x:x.apply(replace_bool_int))


# In[ ]:


#Copy miss_data with df_missing
df_missing = miss_data 


# In[ ]:


#Delete rows with missing values in the delivery_charges column
miss_data = miss_data.dropna(subset=['delivery_charges'],axis=0)


# In[ ]:


#view miss_data
miss_data 


# In[ ]:


#Import linear regression model 
from sklearn.linear_model import LinearRegression


# In[ ]:


#input miss
#ouput r_scores, season_models
#This function uses a linear regression model to calculate the R^2 fraction
def fill_vacancies_season(miss):
    r_scores = {}
    season_models ={}
    for season in miss.season.unique():
        # The miss data was screened by season
        season_missing = miss[miss.season==season].copy() 
        # Linear regression models were used to fit the data for the season
        season_for_impute = LinearRegression().fit(season_missing[['distance_to_nearest_warehouse', 'is_expedited_delivery', 'is_happy_customer']], season_missing['cal_price'])
        # Calculate the R^2 fraction of the model
        r_scores[season] = season_for_impute.score(season_missing[['distance_to_nearest_warehouse', 'is_expedited_delivery', 'is_happy_customer']], season_missing['cal_price'])
        # Store the model and R^2 scores in a dictionary
        season_models[season] = season_for_impute
    return r_scores, season_models


# In[ ]:


# Return values are assigned to season_r_scores,season_models 
season_r_scores,season_models = fill_vacancies_season(miss_data)


# In[ ]:


# View season_r_scores
season_r_scores


# In[ ]:


#View season_models
season_models


# In[ ]:


#Four seasons variables were set to store the corresponding linear regression model
spring_model = season_models['Spring']
summer_model = season_models['Summer']
autumn_model = season_models['Autumn']
winter_model = season_models['Winter']


# In[ ]:


#A function of predicting prices by season
#input row,season_models,features
#output price or  row['cal_price']
def predict_price_by_season(row,season_models,features):
    #If the actual price is the missing value, make the forecast
    if np.isnan(row['cal_price']):
        # Get the linear regression model for the corresponding season
        model = get_season_model(row['season'],season_models)
        #Extract the eigenvalues required for prediction
        features_values = pd.DataFrame([row.loc[features]],columns = features)
        # Make price forecasts on the data
        price = model.predict(features_values)[0]
        return price
    else:
        #Otherwise, return the actual price
        return row['cal_price']
#Obtain the linear regression model for corresponding seasons
#input season,season_models
#output season_models.get(season)
def get_season_model(season,season_models):
    return season_models.get(season)
#List of features needed to predict price
features = ['distance_to_nearest_warehouse','is_expedited_delivery','is_happy_customer']
#A dictionary containing linear regression models for each season
season_models = {'Spring': spring_model,'Summer': summer_model,'Autumn': autumn_model,'Winter': winter_model}
#Make a price forecast for the missing price data and fill in the missing value
df_missing['imputed_price'] = df_missing.apply(lambda row: predict_price_by_season(row,season_models,features),axis=1)


# In[ ]:


df_missing.info()


# In[ ]:


# The linear regression model is used to fill in the value of cal_price and subtract discounts to get the actual charges. The values are filled into the delivery_charges column
df_missing['delivery_charges'] = df_missing.apply(
    lambda row: round(row['imputed_price'] * (1 - (row['delivery_discount'] / 100)), 3) 
                 if pd.isnull(row['delivery_charges']) else row['delivery_charges'],
    axis=1
)


# In[ ]:


#View df_missing data
df_missing.info()


# In[ ]:


#Delete the cart_total column
dirty_data.drop(labels=['cart_total'],axis=1,inplace=True)


# In[ ]:


#Boolean type is converted to Int 0 and 1
#input bool_value
#output 0 or 1
def replace_bool_int(bool_value):
    if bool_value == True:
        return 1
    elif bool_value == False:
        return 0
    else:
        return bool_value
dirty_data = dirty_data.apply(lambda x:x.apply(replace_bool_int))
#Calculate actual price
#input row
#output round(discounted_delivery_charges,3)
def calculate_freight_discount(row):
    delivery_charges = row['delivery_charges']
    delivery_discount = row['delivery_discount']
    discounted_delivery_charges = delivery_charges / (1-(delivery_discount/100))
    return round(discounted_delivery_charges,3)
dirty_data['cal_price'] = dirty_data.apply(calculate_freight_discount,axis=1)
    


# In[ ]:


#A function of predicting prices by season
#input row,season_models,features
#ouput price
def predict_price_by_season(row,season_models,features):
    model = get_season_model(row['season'],season_models)
    features_values = pd.DataFrame([row.loc[features]],columns = features)
    price = model.predict(features_values)[0]
    return price
#Obtain the linear regression model for corresponding seasons
#input season,season_models
#output season_models.get(season)
def get_season_model(season,season_models):
    return season_models.get(season)
#List of features needed to predict price
features = ['distance_to_nearest_warehouse','is_expedited_delivery','is_happy_customer']
#A dictionary containing linear regression models for each season
season_models = {'Spring': spring_model,'Summer': summer_model,'Autumn': autumn_model,'Winter': winter_model}
#Make a price forecast for the missing price data and fill in the missing value
dirty_data['imputed_price'] = dirty_data.apply(lambda row: predict_price_by_season(row,season_models,features),axis=1)


# In[ ]:


#view dirty_data
dirty_data.head()


# In[ ]:


# Determine whether is_expedited_delivery is incorrect
dirty_data[(abs(dirty_data.cal_price - dirty_data.imputed_price)>0.05*abs(dirty_data.cal_price))&(dirty_data['is_expedited_delivery']==0)]


# In[ ]:


#Fix the is_expedited_delivery error
dirty_data.loc[(abs(dirty_data.cal_price - dirty_data.imputed_price) > 0.05 * abs(dirty_data.cal_price)) & (dirty_data['is_expedited_delivery'] == 0), 'is_expedited_delivery'] = 1


# In[ ]:


# Check whether the modification is successful
dirty_data[(abs(dirty_data.cal_price - dirty_data.imputed_price)>0.05*abs(dirty_data.cal_price))&(dirty_data['is_expedited_delivery']==0)]


# In[ ]:


# Determine whether is_expedited_delivery is incorrect
dirty_data[(abs(dirty_data.cal_price - dirty_data.imputed_price)>0.05*abs(dirty_data.cal_price))&(dirty_data['is_expedited_delivery']==1)]


# In[ ]:


#Fix the is_expedited_delivery error
dirty_data.loc[(abs(dirty_data.cal_price - dirty_data.imputed_price) > 0.05 * abs(dirty_data.cal_price)) & (dirty_data['is_expedited_delivery'] == 1), 'is_expedited_delivery'] = 0


# In[ ]:


# Check whether the modification is successful
dirty_data[(abs(dirty_data.cal_price - dirty_data.imputed_price)>0.05*abs(dirty_data.cal_price))&(dirty_data['is_expedited_delivery']==1)]


# # Error Modifying the outlier_data.csv file

# In[ ]:


#Read file
outlier_data = pd.read_csv('32891059_outlier_data.csv')


# In[ ]:


#Boolean type is converted to Int 0 and 1
#input bool_value
#output 0 or 1
def replace_bool_int(bool_value):
    if bool_value == True:
        return 1
    elif bool_value == False:
        return 0
    else:
        return bool_value
outlier_data = outlier_data.apply(lambda x:x.apply(replace_bool_int))


# In[ ]:


# view outlier_data
outlier_data.info()


# In[ ]:


#Calculate actual price
#input row
#output round(discounted_delivery_charges,3)
def calculate_freight_discount(row):
    delivery_charges = row['delivery_charges']
    delivery_discount = row['delivery_discount']
    discounted_delivery_charges = delivery_charges / (1-(delivery_discount/100))
    return round(discounted_delivery_charges,3)
outlier_data['cal_price'] = outlier_data.apply(calculate_freight_discount,axis=1)


# In[ ]:


#A function of predicting prices by season
#input row,season_models,features
#ouput price
def predict_price_by_season(row,season_models,features):
    model = get_season_model(row['season'],season_models)
    features_values = pd.DataFrame([row.loc[features]],columns = features)
    price = model.predict(features_values)[0]
    return price
outlier_data['imputed_price'] = outlier_data.apply(lambda row: predict_price_by_season(row,season_models,features),axis=1)


# In[ ]:


# view outlier_data
outlier_data.head()


# In[ ]:


#Store the data cal_price minus imputed_price in the difference_price column
outlier_data['difference_price'] = (outlier_data.cal_price - outlier_data.imputed_price)


# In[ ]:


#Calculate the IQR upper lower
IQR = (outlier_data.difference_price.quantile(0.75)-outlier_data.difference_price.quantile(0.25))
upper = outlier_data.difference_price.quantile(0.75)+ (IQR*1.5)
lower = outlier_data.difference_price.quantile(0.25)- (IQR*1.5)
print(str(upper))
print(str(lower))


# In[ ]:


#Count outliers and non-outliers
((outlier_data.difference_price>upper)|(outlier_data.difference_price<lower)).value_counts()


# In[ ]:


#Delete the line for outliers
outlier_data.drop(outlier_data[(outlier_data['difference_price']<lower)|(outlier_data['difference_price']>upper)].index,inplace=True)


# In[ ]:


# view outlier_data
outlier_data.info()


# In[ ]:


# Convert the is_expedited_delivery and is_happy_customer of several data to Boolean types
dirty_data['is_expedited_delivery'] = dirty_data['is_expedited_delivery'].replace({0: False, 1: True})
dirty_data['is_happy_customer'] = dirty_data['is_happy_customer'].replace({0: False, 1: True})
df_missing['is_expedited_delivery'] = df_missing['is_expedited_delivery'].replace({0: False, 1: True})
df_missing['is_happy_customer'] = df_missing['is_happy_customer'].replace({0: False, 1: True})
outlier_data['is_expedited_delivery'] = outlier_data['is_expedited_delivery'].replace({0: False, 1: True})
outlier_data['is_happy_customer'] = outlier_data['is_happy_customer'].replace({0: False, 1: True})


# In[ ]:


# view outlier_data
outlier_data.info()


# In[ ]:


# view df_missing
df_missing.info()


# In[ ]:


# view dirty_data
dirty_data.info()


# ## It's all been repaired according to observations

# # EDA drawing

# In[ ]:


#Import matplotlib and seaborn for painting
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Print the first few lines of data
dirty_data.head()


# In[ ]:


#Draw the histogram
plt.hist(dirty_data['delivery_charges'], bins=20)
# Set the graphic title and axis label
plt.xlabel('delivery_charges')
plt.ylabel('Count')
# Display graphics
plt.show()


# According to the histogram, we can find that the amount of customers' consumption is between 30-80, and there are two peaks between 30-40 and 40-50 in the midterm, which proves that the maximum consumption power of customers is between 30-50

# In[ ]:


#Draw a box diagram
sns.boxplot(y=dirty_data['distance_to_nearest_warehouse'])
# Set the graphic title and axis label
plt.title("Diagram of warehouse box type nearest customer")
plt.ylabel("km")
# Display graphics
plt.show()


# As can be seen from the image, most of the distance is within 2.5KM, and the maximum is 0.5-1.5km in the first period. A very few discrete values are greater than 2.5KM, indicating that the distance between the customer and the nearest warehouse is basically less than 2.5km

# In[ ]:


clear_data = dirty_data
# Sort the seasons column by spring, summer, autumn and winter
clear_data['season'] = pd.Categorical(clear_data['season'], categories=['Spring', 'Summer', 'Autumn', 'Winter'], ordered=True)
# Draw a line graph
sns.lineplot(x='season', y='delivery_charges', data=clear_data)

# Set the graphic title and axis label
plt.title("The relationship between season and total price sold")
plt.xlabel("season")
plt.ylabel("value")

# Display graphics
plt.show()


# According to the image, we can find that the total sales volume in Spring is the highest, while the sales volume in winter is the least. With the change of seasons, the sales volume in spring, summer, autumn and winter decreases successively. According to the data analysis, the sales volume in 2022 has been declining, indicating that the current business condition is not good, and the current business model or rules need to be modified

# In[ ]:


# Draw bar charts
sns.countplot(x='nearest_warehouse', data=clear_data)

# Set the graphic title and axis label
plt.title('Bar chart of number of customers in different warehouse')
plt.xlabel('Warehouse name')
plt.ylabel('Count')

# Display graphics
plt.show()


# According to the picture, we can find that Thompson has the largest number of customers and Bakers the least. According to the conclusion of the distance box diagram, we can know that the largest number of customers live near Thompson and the least number of customers near Bakers, so we should focus on the development of Thompson warehouse

# In[ ]:


#Turn the shopping_cart column into a list
shopping_cart_list_clear = clear_data['shopping_cart'].tolist()


# In[ ]:


data = {}
#Calculate the number of units sold per item
for lst in shopping_cart_list_clear:
    lst = eval(lst)
    for eng, num in lst:
        if eng in data:
            data[eng] += num
        else:
            data[eng] = num

# Create data frame
sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)

# Get X - and Y-axis data
x = [item[0] for item in sorted_data]
y = [item[1] for item in sorted_data]
# Draw a bar chart
plt.bar(x, y)

# Add chart title and axis label
plt.title('Sales Data')
plt.xlabel('Product')
plt.ylabel('Sales')
plt.xticks(fontsize=6)
plt.xticks(rotation=45)
# Display chart
plt.show()


# According to the picture, we can find that among the products sold, Lucent 3305 has the largest sales volume, while iStream has the lowest sales volume. However, there is not much sales gap among various products. We can develop lucent3305 and candle infemo and similar products of these products. Comparatively reduce iStream's commodity development. This is conducive to reducing the cost of some goods, easier capital flow, gain greater benefits.

# Through data collation and analysis, it is found that the company's business condition is deteriorating, and the following business plan should be adjusted:
# 1. Provide more warehouse so that users can go shopping closer
# 2. The goods sold should be between 30 and 80, in line with the shopper's funds
# 3. Focus on developing more customers warehouse: such as Thompson
# 4. Purchase of goods sold shall be adjusted according to the sales volume
# The above is my data analysis. Although there are limitations and limited data, the results of linear equation generally conform to the above rules.

# # Output three CSV files

# In[ ]:


#Delete the imputed_price  cal_price difference_price column
dirty_data.drop(labels = ['imputed_price'],axis=1,inplace = True)
df_missing.drop(labels = ['imputed_price'],axis=1,inplace = True)
outlier_data.drop(labels = ['imputed_price'],axis=1,inplace = True)
dirty_data.drop(labels = ['cal_price'],axis=1,inplace = True)
df_missing.drop(labels = ['cal_price'],axis=1,inplace = True)
outlier_data.drop(labels = ['cal_price'],axis=1,inplace = True)
outlier_data.drop(labels = ['difference_price'],axis=1,inplace = True)


# In[ ]:


#Check data before output
dirty_data.info()


# In[ ]:


#Check data before output
df_missing.info()


# In[ ]:


#Check data before output
outlier_data.info()


# In[ ]:


# output dirty_data_solution.csv
dirty_data.to_csv('32891059_dirty_data_solution.csv', index=False)


# In[ ]:


# output missing_data_solution.csv
df_missing.to_csv('32891059_missing_data_solution.csv', index=False)


# In[ ]:


# output outlier_data_solution.csv
outlier_data.to_csv('32891059_outlier_data_solution.csv', index=False)

