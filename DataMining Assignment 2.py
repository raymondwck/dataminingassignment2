#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc


# In[2]:


quality = pd.read_csv("C:/Users/Ck/Desktop/Study/2024 Jan Sesmester/Data Mining/Assignment 2 group/apple_quality.csv")


# In[3]:


quality.head()


# # EDA

# In[4]:


quality.info()


# In[5]:


quality.describe()


# In[6]:


quality.count()


# # Data Preprocessing

# ### Checking for NULL values

# In[7]:


# Missing data
for i in quality.columns:
    null_rate = quality[i].isna().sum() / len(quality) * 100 
    if null_rate > 0 :
        print("{} null rate: {}%".format(i,round(null_rate,2)))


# In[8]:


quality.isnull().sum()


# In[9]:


#drop the null values 
quality=quality.dropna()
quality.head()


# In[10]:


quality.isnull().sum()


# ### Checking for duplicated values

# In[11]:


quality.duplicated().sum()


# In[12]:


# For viz: Ratio of Movies & TV shows
x=quality.groupby(['Quality'])['Quality'].count()
y=len(quality)
r=((x/y)).round(2)

mf_ratio = pd.DataFrame(r).T


# In[13]:


fig, ax = plt.subplots(1,1,figsize=(6.5, 2.5))

ax.barh(mf_ratio.index, mf_ratio['good'], 
        color='#b20710', alpha=0.9, label='Male')
ax.barh(mf_ratio.index, mf_ratio['bad'], left=mf_ratio['good'], 
        color='green', alpha=0.9, label='Female')

ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_yticks([])

for i in mf_ratio.index:
    ax.annotate(f"{int(mf_ratio['good'][i]*100)}%", 
                   xy=(mf_ratio['good'][i]/2, i),
                   va = 'center', ha='center',fontsize=40, fontweight='light', fontfamily='serif',
                   color='white')

    ax.annotate("Good", 
                   xy=(mf_ratio['good'][i]/2, -0.25),
                   va = 'center', ha='center',fontsize=15, fontweight='light', fontfamily='serif',
                   color='white')
    
    
for i in mf_ratio.index:
    ax.annotate(f"{int(mf_ratio['bad'][i]*100)}%", 
                   xy=(mf_ratio['bad'][i]+mf_ratio['bad'][i]/2, i),
                   va = 'center', ha='center',fontsize=40, fontweight='light', fontfamily='serif',
                   color='white')
    ax.annotate("Bad", 
                   xy=(mf_ratio['good'][i]+mf_ratio['bad'][i]/2, -0.25),
                   va = 'center', ha='center',fontsize=15, fontweight='light', fontfamily='serif',
                   color='white')

for s in ['top', 'left', 'right', 'bottom']:
    ax.spines[s].set_visible(False)

ax.legend().set_visible(False)
plt.show()


# In[14]:


compare = quality['Quality'].value_counts()
compare


# In[15]:


# Create a bar plot with green and red colors
plt.bar(compare.index, compare, color=['green', 'red'])

# Add labels and title
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Good & Bad bar')

# Show the plot
plt.show() 


# ### Checking Outliners

# In[16]:


plt.boxplot(quality["Size"])

# Add labels and title
plt.xlabel('Size')
plt.ylabel('Values')
plt.title('Outliners for Size')

# Show the plot
plt.show()


# In[17]:


plt.boxplot(quality['Weight'])

# Add labels and title
plt.xlabel('Weight')
plt.ylabel('Values')
plt.title('Outliners for Weight')

# Show the plot
plt.show()


# In[18]:


plt.boxplot(quality['Sweetness'])

# Add labels and title
plt.xlabel('Sweetness')
plt.ylabel('Values')
plt.title('Outliners for Sweetness')

# Show the plot
plt.show()


# In[19]:


plt.boxplot(quality['Crunchiness'])

# Add labels and title
plt.xlabel('Crunchiness')
plt.ylabel('Values')
plt.title('Outliners for Crunchiness')

# Show the plot
plt.show()


# In[20]:


plt.boxplot(quality['Juiciness'])

# Add labels and title
plt.xlabel('Juiciness')
plt.ylabel('Values')
plt.title('Outliners for Juiciness')

# Show the plot
plt.show()


# In[21]:


plt.boxplot(quality['Ripeness'])

# Add labels and title
plt.xlabel('Ripeness')
plt.ylabel('Values')
plt.title('Outliners for Ripeness')

# Show the plot
plt.show()


# In[22]:


# Check the data types of the 'Acidity' column
print(quality['Acidity'].dtype)

# Convert the column to numeric type if necessary
quality['Acidity'] = pd.to_numeric(quality['Acidity'], errors='coerce')

# Check for any NaN values after conversion
print(quality['Acidity'].isna().sum())


# In[23]:


plt.boxplot(quality['Acidity'])

# Add labels and title
plt.xlabel('Acidity')
plt.ylabel('Values')
plt.title('Outliners for Acidity')

# Show the plot
plt.show()


# In[24]:


quality.drop('A_id',axis=1,inplace=True)
quality.head()


# In[25]:


#remove the outliers:
filter_data=quality.iloc[:,[0,1,2,3,4,5,6]]
filter_data.head()


# In[26]:


from sklearn.preprocessing import RobustScaler, StandardScaler
numerical_features = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness',
       'Acidity']

robust_scaler = RobustScaler()

quality[numerical_features] = robust_scaler.fit_transform(quality[numerical_features])

def count_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).sum()

for feature in numerical_features :
    num_outliers = count_outliers(quality[feature])
    print(f'Number of outliers in {feature}: {num_outliers}')


# In[27]:


for column in filter_data.columns:
    Q1 = filter_data[column].quantile(0.25)
    Q3 = filter_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Use .loc to assign values
    quality.loc[(filter_data[column] >= lower_bound) & (filter_data[column] <= upper_bound), column] = filter_data[(filter_data[column] >= lower_bound) & (filter_data[column] <= upper_bound)][column]


# In[28]:


quality.info()


# In[29]:


# Create a box plot
counter=quality.Size.value_counts()
plt.boxplot(counter.index)

# Add labels and title
plt.xlabel('size')
plt.ylabel('Values')
plt.title('Removed Outliers for Size')

# Show the plot
plt.show()


# In[30]:


# Create a box plot
counter=quality.Weight.value_counts()
plt.boxplot(counter.index)
       
# Add labels and title
plt.xlabel('Weight')
plt.ylabel('Values')
plt.title('Removed Outliers for Weight')
       
# Show the plot
plt.show()


# In[31]:


counter=quality.Sweetness.value_counts()
plt.boxplot(counter.index)
        
# Add labels and title
plt.xlabel('Sweetness')
plt.ylabel('Values')
plt.title('Removed Outliers for Sweetness')
        
# Show the plot
plt.show()


# In[32]:


# Create a box plot
counter=quality.Crunchiness.value_counts()
plt.boxplot(counter.index)
       
# Add labels and title
plt.xlabel('Crunchiness')
plt.ylabel('Values')
plt.title('Removed Outliers for Crunchiness')
       
# Show the plot
plt.show()


# In[33]:


# Create a box plot
counter=quality.Juiciness.value_counts()
plt.boxplot(counter.index)
       
# Add labels and title
plt.xlabel('Juiciness')
plt.ylabel('Values')
plt.title('Removed Outliers for Juiciness')
       
# Show the plot
plt.show()


# In[34]:


# Create a box plot
counter=quality.Ripeness.value_counts()
plt.boxplot(counter.index)
       
# Add labels and title
plt.xlabel('Ripeness')
plt.ylabel('Values')
plt.title('Removed Outliers for Ripeness')
       
# Show the plot
plt.show()


# In[35]:


# Create a box plot
counter=quality.Acidity.value_counts()
plt.boxplot(counter.index)
       
# Add labels and title
plt.xlabel('Acidity')
plt.ylabel('Values')
plt.title('Removed Outliers for Acidity')
       
# Show the plot
plt.show()


# In[36]:


for feature in numerical_features :
    num_outliers = count_outliers(quality[feature])
    print(f'Number of outliers in {feature}: {num_outliers}')


# In[37]:


plt.hist(quality['Size'], bins=150, color='green', edgecolor='black',histtype='stepfilled')
plt.xlabel('size values')
plt.ylabel('Frequency')
plt.title('size distribution')
        
# Show the plot
plt.show()


# In[38]:


plt.hist(quality['Weight'], bins=150, color='green', edgecolor='black',histtype='stepfilled')
 # Add labels and title
plt.xlabel('Weight values')
plt.ylabel('Frequency')
plt.title('Weight distribution')
        
# Show the plot
plt.show()


# In[39]:


plt.hist(quality['Sweetness'], bins=150, color='green', edgecolor='black',histtype='stepfilled')
 # Add labels and title
plt.xlabel('Sweetness values')
plt.ylabel('Frequency')
plt.title('Sweetness distribution')
        
# Show the plot
plt.show()


# In[40]:


plt.hist(quality['Crunchiness'], bins=150, color='green', edgecolor='black',histtype='stepfilled')
 # Add labels and title
plt.xlabel('Crunchiness values')
plt.ylabel('Frequency')
plt.title('Crunchiness distribution')
        
# Show the plot
plt.show()


# In[41]:


plt.hist(quality['Juiciness'], bins=150, color='green', edgecolor='black',histtype='stepfilled')
 # Add labels and title
plt.xlabel('Juiciness values')
plt.ylabel('Frequency')
plt.title('Juiciness distribution')
        
# Show the plot
plt.show()


# In[42]:


plt.hist(quality['Ripeness'], bins=150, color='green', edgecolor='black',histtype='stepfilled')
 # Add labels and title
plt.xlabel('Ripeness values')
plt.ylabel('Frequency')
plt.title('Ripeness distribution')
        
# Show the plot
plt.show()


# In[43]:


plt.hist(quality['Acidity'], bins=150, color='green', edgecolor='black',histtype='stepfilled')
 # Add labels and title
plt.xlabel('Acidity values')
plt.ylabel('Frequency')
plt.title('Acidity distribution')
        
# Show the plot
plt.show()


# In[44]:


quality.describe()


# ### Correlation heatmap

# In[45]:


correalation=quality.iloc[:,[0,1,2,3,4,5,6]]
correalation


# In[46]:


correalation.corr()


# In[47]:


# Generate a custom colormap with red and green
colors = [(0, 1, 0), (1, 0, 0)]  # Green to Red
cmap = LinearSegmentedColormap.from_list("RedGreen", colors, N=256)

# Compute correlation and create heatmap
realtion = correalation.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(realtion, annot=True, cmap=cmap, fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()


# # Modeling

# In[48]:


def clean_data(df):
    def label(Quality):

        if Quality == "good":
            return 0
    
        if Quality == "bad":
            return 1
    
        return None
    
    df['Label'] = df['Quality'].apply(label)
    
    df = df.drop(columns=['Quality'])
    
    df = df.astype({'Label': 'int64'})
    
    return df

df_clean = clean_data(quality)
df_clean.head()


# In[49]:


df1 = df_clean.copy()


# In[50]:


X = df1.drop(['Label'],axis=1)
y = df1['Label']


# In[51]:


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler, StandardScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)


# In[52]:


X_resampled.describe().T.style.background_gradient(axis=0, cmap='viridis')


# In[53]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# ### SVC Classification

# In[54]:


param_dist = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1],
}

svc = SVC()

randomized_search = RandomizedSearchCV(svc, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)

randomized_search.fit(X_train, y_train)

best_params = randomized_search.best_params_
print(f"Best Hyperparameters: {best_params}")

best_svc_model = randomized_search.best_estimator_
svc_predicted = best_svc_model.predict(X_test)

svc_acc_score = accuracy_score(y_test, svc_predicted)
svc_conf_matrix = confusion_matrix(y_test, svc_predicted)

print("\nConfusion Matrix:")
print(svc_conf_matrix)
print("\nAccuracy of Support Vector Classifier:", svc_acc_score * 100, '\n')
print("Classification Report:")
print(classification_report(y_test, svc_predicted))


# ### Random forest Classification

# In[55]:


param_dist_rf = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
}

rf = RandomForestClassifier()

randomized_search_rf = RandomizedSearchCV(
    rf,
    param_distributions=param_dist_rf,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

randomized_search_rf.fit(X_train, y_train)

best_params_rf = randomized_search_rf.best_params_
print(f"Best Hyperparameters for Random Forest: {best_params_rf}")

best_rf_model = randomized_search_rf.best_estimator_

rf_predicted = best_rf_model.predict(X_test)

rf_acc_score = accuracy_score(y_test, rf_predicted)
rf_conf_matrix = confusion_matrix(y_test, rf_predicted)

print("\nConfusion Matrix for Random Forest:")
print(rf_conf_matrix)
print("\nAccuracy of Random Forest Classifier:", rf_acc_score * 100, '\n')
print("Classification Report for Random Forest:")
print(classification_report(y_test, rf_predicted))


# ### KNN Classification

# In[56]:


param_dist_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30, 40],
    'p': [1, 2],
}

knn = KNeighborsClassifier()

randomized_search_knn = RandomizedSearchCV(
    knn,
    param_distributions=param_dist_knn,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

randomized_search_knn.fit(X_train, y_train)

best_params_knn = randomized_search_knn.best_params_
print(f"Best Hyperparameters for KNeighborsClassifier: {best_params_knn}")

best_knn_model = randomized_search_knn.best_estimator_

knn_predicted = best_knn_model.predict(X_test)

knn_acc_score = accuracy_score(y_test, knn_predicted)
knn_conf_matrix = confusion_matrix(y_test, knn_predicted)

print("\nConfusion Matrix for KNeighborsClassifier:")
print(knn_conf_matrix)
print("\nAccuracy of KNeighborsClassifier:", knn_acc_score * 100, '\n')
print("Classification Report for KNeighborsClassifier:")
print(classification_report(y_test, knn_predicted))


# ### One Rule Classification

# In[57]:


# Define the feature columns and the target column
feature_cols = quality.columns[:-1]  # all columns except the last one
target_col = quality.columns[-1]  # the last column

# Split the data into features and target
X = quality[feature_cols]
y = quality[target_col]

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[58]:


# Define the OneR algorithm
def oneR_algorithm(X_train, y_train):
    # Find the best attribute by trying to split on each of them
    best_attribute = None
    best_error_rate = float('inf')
    attribute_rules = {}

    for attribute in X_train.columns:
        # Count the frequency of each value of the attribute
        value_counts = X_train[attribute].value_counts().to_dict()

        # For each attribute value, find the most frequent class
        rules = {}
        errors = 0
        for value, count in value_counts.items():
            most_frequent_class = y_train[X_train[attribute] == value].mode()[0]
            rules[value] = most_frequent_class
            # Count errors for the current attribute
            errors += count - y_train[X_train[attribute] == value].value_counts()[most_frequent_class]

        # Calculate the error rate for the attribute
        error_rate = errors / len(X_train)

        # Update the best attribute if the current one is better
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_attribute = attribute
            attribute_rules = rules

    return best_attribute, attribute_rules


# In[59]:


# Train the OneR classifier
best_attribute, rules = oneR_algorithm(X_train, y_train)


# In[60]:


# A function to make predictions using OneR
def oneR_predict(X, best_attribute, rules):
    predictions = []
    for index, row in X.iterrows():
        attribute_value = row[best_attribute]
        predictions.append(rules.get(attribute_value, y_train.mode()[0]))
    return predictions


# In[61]:


# Make predictions on the test set
y_pred = oneR_predict(X_test, best_attribute, rules)


# In[62]:


# Print out the results
print(f"The best attribute is: {best_attribute}")
print("The rules based on the best attribute are:")
print(rules)


# In[63]:


## Good = 0 ; Bad = 1

newApple_data = {
    'Size': [3],  
    'Weight': [-2],  
    'Sweetness': [0],  
    'Crunchines': [1],  
    'Juiciness': [4],  
    'Ripeness': [9],  
    'Acidity': [3]  
}

#Convert the dictionary to a DataFrame
new_data = pd.DataFrame(newApple_data)

# Use the trained OneR classifier to make predictions
new_predictions = oneR_predict(new_data, best_attribute, rules)

# Output the predictions
print("Predictions for the new Apple:\n")
if new_predictions[0] == 0:
    print ("Apple is Good ")
else:
    print("Apple is Bad")


# In[64]:


## Good = 0 ; Bad = 1

newApple_data = {
    'Size': [-0.292023862], 
    'Weight': [-1.351281995], 
    'Sweetness': [-1.738429162],  
    'Crunchines': [-0.342615928],  
    'Juiciness': [2.838635512],  
    'Ripeness': [-0.038033328],  
    'Acidity': [-0.038033328]  
}

#Convert the dictionary to a DataFrame
new_data = pd.DataFrame(newApple_data)

# Use the trained OneR classifier to make predictions
new_predictions = oneR_predict(new_data, best_attribute, rules)

# Output the predictions
# Output the predictions
print("Predictions for the new Apple:\n")
if new_predictions[0] == 0:
    print ("Apple is Good")
else:
    print("Apple is Bad")


# In[ ]:




