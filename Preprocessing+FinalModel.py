# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:58:46 2023

@author: Tony
"""

# Importing Packages
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import nltk
from nltk.corpus import wordnet
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve
from imblearn.over_sampling import SMOTE
import shap
from sklearn.metrics import roc_curve, auc


# Load the data
df = pd.read_csv("C:/Users/Tony/Downloads/linkdin_Job_data.csv")
df = df.drop(columns=[ 'company_id','hiring_person_link','alumni'])
###Task 1: Data Preprocessing
# 1.Drop job id and details id
df=df.drop_duplicates(subset='job_ID', keep='first')

# 2. keep hours since posted
#dummify hiring name given or not column
df['Hiring_person_indicator'] = df['Hiring_person'].notna().astype(int)

# 3.Convert time since posted to common hours

# Function to convert values to hours
def to_hours(value):
    # Check if the value is a string type
    if isinstance(value, str):
        if "minute" in value:
            return int(value.split()[0]) / 60
        elif "day" in value:
            return int(value.split()[0]) * 24
        elif "hour" in value:
            return int(value.split()[0])
    return value

df['posted_day_ago'] = df['posted_day_ago'].apply(to_hours)

print(df)

# 4.Applications/hours for application rate

# Convert no_of_application column to float
df['no_of_application'] = pd.to_numeric(df['no_of_application'], errors='coerce')
df['posted_day_ago'] = pd.to_numeric(df['posted_day_ago'], errors='coerce')

# Calculate Application_rate
df['Application_rate'] = df['no_of_application'] / df['posted_day_ago']

# 5.Create column as 1 or 0 based on above avg or not

# Calculate average of scores
avg_score = df['Application_rate'].mean()

# Create new column 'above_avg'
df['above_avg'] = np.where(df['Application_rate'] > avg_score, 1, 0)

#6.	Create column if hiring person linked in job or not
#df['is_linkedin_job'] = df['hiring_person_link'].apply(lambda x: 1 if "linkedin.com/in/" in str(x) else 0)

#7.	Also add len(job_details) because maybe longer job descriptions get more applications
df['len_job_details'] = df['job_details'].apply(lambda x: len(x) if isinstance(x, str) else 0)

unique_locations_list = df['location'].unique().tolist()
print(unique_locations_list)

# 8.	Extract the state from the location column
states = [
    'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh',
    'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand', 'karnataka',
    'kerala', 'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram',
    'nagaland', 'odisha', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu', 'telangana',
    'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal', 'andaman and nicobar islands',
    'chandigarh', 'dadra and nagar haveli', 'daman and diu', 'delhi', 'lakshadweep',
    'puducherry', 'jammu & kashmir'  # Added 'Jammu & Kashmir'
]

# Function to find the state in the location
def find_state(location):
    if pd.notnull(location):  # Check if 'location' is not NaN
        location_lower = location.lower()
        for state in states:
            if state in location_lower:
                return state.title()
            elif 'bengaluru' in location_lower or 'bangalore' in location_lower:
                 return 'Karnataka'
            elif 'pune' in location_lower or 'mumbai' in location_lower or 'nagpur' in location_lower:
                 return 'Maharashtra'
            elif 'chennai' in location_lower or 'coimbatore' in location_lower:
                 return 'Tamil Nadu'
            elif 'vadodara' in location_lower or 'ahmedabad' in location_lower:
                 return 'Gujarat'
            elif 'kolkata' in location_lower:
                 return 'West Bengal'
            elif 'hyderabad' in location_lower:
                 return 'Telangana'
    return 'Multi-state Jobs'


# Apply the function to create the new 'state' column
df['state'] = df['location'].apply(find_state)

print(df[['location', 'state']])


# Assuming df is your DataFrame with 'location' and 'state' columns already present
multi_state_locations = df[df['state'] == 'Multi-state Jobs']['location'].unique()

# Convert NumPy array to a list
multi_state_locations_list = multi_state_locations.tolist()

print(multi_state_locations_list)


#  9. download the WordNet data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample dataframe
# df = pd.read_csv("C:/Users/vinay/OneDrive/Documents/.spyder-py3/job_cleanData.csv")

# Define keywords and their synonyms for each category
categories = {
    'Benefits-Related': ["bonuses","awards", "healthcare", "retirement", "vacation","flexible","outings"],
    'Company-Culture': ["inclusive", "diverse","balance", "collaborative","innovative",'diversity','equity','inclusion',"teamwork",'people'],
    'Growth and Development':["career","grow","training","development","mentorship","learning","self-development"]
}

# Fetch synonyms for the keywords
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))  # Replace underscore with space for multi-word synonyms
    return list(synonyms)

# Create a dictionary with keywords and their synonyms
category_synonyms = {}
for category, keywords in categories.items():
    synonyms = set()
    for keyword in keywords:
        synonyms.update([keyword])        # Add the keyword itself
        synonyms.update(get_synonyms(keyword))  # Add the synonyms of the keyword
    category_synonyms[category] = synonyms

# Create dummy variables
df['job_details'] = df['job_details'].astype(str)
for category, synonyms in category_synonyms.items():
    df[category] = df['job_details'].apply(lambda x: 1 if any(syn in x for syn in synonyms) else 0)


# 10. Create a column for the number of years of experience required

def experience_to_number(text):
    patterns = [
        r'(\d+)(?:-|\s?to\s?)(\d+)\syears',
        r'(\d+)\s?\+\s?years',
        r'(?:at least\s)?(\d+)\syears',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return max(int(num) for num in match.groups() if num is not None)
    return None

df['experience_num'] = df['job_details'].apply(lambda x: experience_to_number(str(x)))

degree_mapping = {
    'bachelor': ['bachelor', 'b.sc', 'b.s.', 'bs', 'undergraduate'],
    'master': ['master', 'm.sc', 'm.s.', 'ms', 'graduate'],
    'phd': ['phd', 'ph.d.', 'doctorate']
}

def create_degree_dummies(text, degree_mapping):
    text = text.lower()
    dummies = {degree: 0 for degree in degree_mapping.keys()}

    for degree, synonyms in degree_mapping.items():
        if any(synonym in text for synonym in synonyms):
            dummies[degree] = 1

    return pd.Series(dummies)

degree_dummies = df['job_details'].apply(lambda x: create_degree_dummies(str(x), degree_mapping))
df = df.join(degree_dummies)


experience_level = ["Mid-Senior level", "Associate", "Entry level", "Executive", "Director", "Internship"]
job_type = ["Full-time", "Contract", "Internship", "Part-time", "Temporary"]
df['Experience Level'] = df["full_time_remote"].apply(lambda x: next((level for level in experience_level if isinstance(x, str) and level in x), np.nan))
df['Job Type'] = df["full_time_remote"].apply(lambda x: next((job for job in job_type if isinstance(x, str) and job in x), np.nan))

# Extract the number of employees from the no_of_employ column
df['employee_range'] = df['no_of_employ'].str.extract(r'(.*?)\s+employees')
# Extract the industry name similarly from the no_of_employ column
df['industry'] = df['no_of_employ'].str.extract(r'·\s+(.*)')

# 11.Map industry to a predefined category

def map_industry_to_category(industry):
    
    if isinstance(industry, str):
    # IT and Software
        if 'it' in industry.lower() or 'software' in industry.lower() or 'technology' in industry.lower() or 'computer' in industry.lower() or 'internet' in industry.lower() or 'data' in industry.lower():
            return 'IT & Software'

        # Engineering and Manufacturing
        elif 'engineering' in industry.lower() or 'manufacturing' in industry.lower() or 'industrial' in industry.lower() or 'machinery' in industry.lower() or 'semiconductor' in industry.lower():
            return 'Engineering & Manufacturing'

        # Healthcare
        elif 'health' in industry.lower() or 'medical' in industry.lower() or 'pharmaceutical' in industry.lower() or 'biotechnology' in industry.lower():
            return 'Healthcare'

        # Finance and Business
        elif 'financial' in industry.lower() or 'business' in industry.lower() or 'accounting' in industry.lower() or 'banking' in industry.lower() or 'investment' in industry.lower():
            return 'Finance & Business'

        # Education and Research
        elif 'education' in industry.lower() or 'research' in industry.lower() or 'e-learning' in industry.lower() or 'training' in industry.lower():
            return 'Education & Research'

        # Media, Marketing, and Communications
        elif 'media' in industry.lower() or 'marketing' in industry.lower() or 'advertising' in industry.lower() or 'publishing' in industry.lower() or 'communications' in industry.lower():
            return 'Media, Marketing & Communications'

        # Retail and Consumer Services
        elif 'retail' in industry.lower() or 'consumer' in industry.lower() or 'apparel' in industry.lower() or 'luxury' in industry.lower() or 'fashion' in industry.lower():
            return 'Retail & Consumer Services'

        # Energy and Environment
        elif 'energy' in industry.lower() or 'environmental' in industry.lower() or 'oil' in industry.lower() or 'gas' in industry.lower() or 'renewable' in industry.lower():
            return 'Energy & Environment'

        # Transportation and Logistics
        elif 'transportation' in industry.lower() or 'logistics' in industry.lower() or 'aviation' in industry.lower() or 'automotive' in industry.lower() or 'maritime' in industry.lower():
            return 'Transportation & Logistics'

        # Telecommunications
        elif 'telecom' in industry.lower() or 'wireless' in industry.lower():
            return 'Telecommunications'

        # Government and Public Services
        elif 'government' in industry.lower() or 'public' in industry.lower() or 'administration' in industry.lower() or 'military' in industry.lower():
            return 'Government & Public Services'

        # Other
        else:
            return 'Other'

# Apply the mapping function to the industry column to create a new 'industry_category' column
df['industry_category'] = df['industry'].apply(map_industry_to_category)

# Rank the number of employees according to business logic. Rank 4 means very large company and 1 means small.

def map_employee_range_to_rank(range):
    if pd.isna(range):
        return None  
    if isinstance(range, float): 
        return None  
    if isinstance(range, str):
        parts = range.replace('+', '').split('-')
        parts = [int(part.replace(',', '')) for part in parts]
        low = parts[0]
        high = parts[-1] 

        if low <= 50:
            return 1  # Small companies
        elif 51 <= low <= 500:
            return 2  # Medium companies
        elif 501 <= low <= 5000:
            return 3  # Large companies
        else:
            return 4  # Very large companies
    else:
        return None  # for any other unexpected types

# Apply the mapping function to the 'employee_range' column
df['rank'] = df['employee_range'].apply(map_employee_range_to_rank)

print(df)

df.to_csv('C:/Users/Tony/Downloads/out3.csv')

##FINAL SELECTED MODEL CODE - RANDOM FOREST


# ### Add different data roles
df['data_analyst'] = df['job'].str.contains('data analyst', case=False, na=False) 
df['data_engineer'] = df['job'].str.contains('data engineer', case=False, na=False) 
df['data_scientist'] = df['job'].str.contains('data scientist', case=False, regex=True, na=False) 
df['backend'] = df['job'].str.contains('backend', case=False, na=False)


# ### Columns to drop
columns_to_drop = ['Unnamed: 0','Unnamed: 0.1', 'job_ID', 'job', 'location', 'company_name', 
                   'work_type', 'full_time_remote', 'no_of_employ', 'no_of_application', 
                   'posted_day_ago', 'Hiring_person', 'processed_description', 
                   'job_details', 'Application_rate', 'Column1']

df = df.drop(columns=columns_to_drop)

# ### Additional Preprocessing
# 
# 1. LinkedIn followers

df['linkedin_followers'] = df['linkedin_followers'].str.replace(',', '').str.replace(' followers', '')
df['linkedin_followers'] = pd.to_numeric(df['linkedin_followers'], errors='coerce').fillna(0).astype(int)
mean_followers = df.loc[df['linkedin_followers'] != 0, 'linkedin_followers'].mean()

# Imputing mean
df.loc[df['linkedin_followers'] == 0, 'linkedin_followers'] = mean_followers

# 2. Dummifying
categorical = df.select_dtypes(include=['object', 'category']).columns

# Dummify only categorical variables
df_dummies = pd.get_dummies(df[categorical], drop_first=True)
df = df.drop(categorical, axis=1)
df = pd.concat([df, df_dummies], axis=1)

# 3. Splitting to X and y
# Split the data into features and target
X = df.drop('above_avg', axis=1)
y = df['above_avg']

# 4. Imputing missing values`

imputer = SimpleImputer(strategy='mean') 
X_imputed = imputer.fit_transform(X)  



# 5.Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)



# 6.Initialize SMOTE
smote = SMOTE()

 

# Fit and apply SMOTE only on training data
X_train, y_train = smote.fit_resample(X_train, y_train)

# 7.Hyperparameters for GridSearchCV 

param_grid = {
    'n_estimators': [50,100, 150, 200, 250],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],        # Maximum depth of the trees
    'min_samples_split': [2, 4, 6, 8],      # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2,3, 4, 6],       # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 0.5, 0.75,3,4,5,6]  # Number of features to consider for the best split
}

# 8.Initialize the GridSearchCV with the Random Forest model
grid_search = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Using the best estimator found by GridSearchCV
model_rf = grid_search.best_estimator_

# Get probability scores for the precision-recall curve
probabilities = model_rf.predict_proba(X_test)[:, 1]

# Calculate precision-recall pairs for different probability thresholds
precision, recall, thresholds = precision_recall_curve(y_test, probabilities)

# Choose a new threshold for classification
new_threshold = 0.31
modified_predictions = (probabilities >= new_threshold).astype(int)

# Evaluate the model with the modified predictions
new_accuracy = accuracy_score(y_test, modified_predictions)
new_cm = confusion_matrix(y_test, modified_predictions)
new_report = classification_report(y_test, modified_predictions)

# Output the evaluation metrics
print("\nConfusion Matrix:\n", new_cm)
print("\nClassification Report:\n", new_report)
print("\nBest hyperparameters from GridSearchCV:")
print(grid_search.best_params_)





#  Feature Importance Plot
# Calculate feature importances and sort them
feature_importances = model_rf.feature_importances_
sorted_idx = feature_importances.argsort()


X_df = pd.DataFrame(X)

# Take only the top 20 features
top_n = 20
top_sorted_idx = sorted_idx[-top_n:]

plt.figure(figsize=(10, 6))
plt.barh(X_df.columns[top_sorted_idx], feature_importances[top_sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.title("Top 20 Features")
plt.show()


# 3. SHAP Values 
explainer = shap.TreeExplainer(model_rf)
shap_values = explainer.shap_values(X_train)

# Summarize the effects of all the features
shap.summary_plot(shap_values, X_df, plot_type="bar")



# Assuming pred_probs contains the predicted probabilities for the positive class
pred_probs = model_rf.predict_proba(X_test)[:, 1]

# Compute ROC curve data
fpr, tpr, thresholds = roc_curve(y_test, pred_probs)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


