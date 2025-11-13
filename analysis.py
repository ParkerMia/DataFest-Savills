import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree, export_graphviz
import graphviz

# --- Load Data ---
LS = pd.read_csv('Leases.csv')
MMO = pd.read_csv('Major Market Occupancy Data-revised.csv')
PA = pd.read_csv('Price and Availability Data.csv')
UE = pd.read_csv('Unemployment.csv')
IN = pd.read_csv('industry.csv')

# --- Average Rent Analysis ---
# Compute average rent per quarter
avg_cost_per_quarter = LS.groupby(['year', 'quarter'])['overall_rent'].mean().reset_index()
avg_cost_per_quarter.rename(columns={'overall_rent': 'average_cost'}, inplace=True)

# Plot average cost per quarter by year
for year in avg_cost_per_quarter['year'].unique():
    data = avg_cost_per_quarter[avg_cost_per_quarter['year'] == year]
    plt.plot(data['quarter'], data['average_cost'], marker='o', label=f"{year}")
plt.title('Average Cost per Quarter by Year')
plt.xlabel('Quarter')
plt.ylabel('Average Cost')
plt.xticks([1, 2, 3, 4])
plt.legend(title='Year')
plt.tight_layout()
plt.show()

# Compute average rent per month
avg_cost_per_month = LS.groupby(['year', 'monthsigned'])['overall_rent'].mean().reset_index()
avg_cost_per_month.rename(columns={'overall_rent': 'average_cost'}, inplace=True)

# Plot average cost per month by year
for year in avg_cost_per_month['year'].unique():
    data = avg_cost_per_month[avg_cost_per_month['year'] == year]
    plt.plot(data['monthsigned'], data['average_cost'], marker='o', label=f"{year}")
plt.title('Average Cost per Month by Year')
plt.xlabel('Month')
plt.ylabel('Average Cost')
plt.xticks(range(1, 13))
plt.legend(title='Year')
plt.tight_layout()
plt.show()

# --- Industry Data Cleaning ---
# Remove missing and TBD industry entries
IN = IN[IN['internal_class'] != 'Na']
IN = IN[IN['internal_industry'] != 'TBD']

# Convert categorical variables to numeric codes
IN['cbd_suburban'] = IN['cbd_suburban'].replace({'CBD': 1, 'Suburban': 0})
IN['internal_industry'] = IN['internal_industry'].replace({
    'Financial Services and Insurance':1,
    'Construction, Engineering and Architecture':2,
    'Technology, Advertising, Media, and Information':3,
    'Manufacturing (except Pharmaceutical, Retail, and Computer Tech)':4,
    'Associations and Non-profit Organizations (except Education and Non-profit Hospitals)':5,
    'Transportation':6,
    'Coworking and Executive Suite Companies':7,
    'Business, Professional, and Consulting Services (except Financial and Legal) - Including Accounting':8,
    'Education':9,
    'Legal Services':10,
    'Real Estate (except coworking providers)':11,
    'Healthcare':12,
    'Personal Services and Recreation':13,
    'Government':14,
    'Retail':15,
    'Energy & Utilities':16,
    'Pharmaceuticals':17,
    'Agriculture, Forestry, Fishing, Metal & Mineral Mining':18,
    'Unclassifiable':19
})
IN['state'] = IN['state'].replace({
    'AZ':1, 'CA':2, 'CO':3, 'DC':4, 'DE':5, 'FL':6, 'GA':7, 
    'IL':8, 'MA':9, 'MD':10, 'MI':11, 'NC':12, 'NH':13, 'NJ':14, 
    'NY':15, 'PA':16, 'SC':17, 'TN':18, 'TX':19, 'UT':20, 'VA':21, 'WA':22
})

# --- Exploratory Data Analysis ---
# Count of industries in finance and technology
finance = IN[IN['internal_industry']==4]
sns.countplot(data=finance, x='state')
plt.show()

tech = IN[IN['internal_industry']==3]
sns.countplot(data=tech, x='market')
plt.show()

# Countplot of all industry types
plt.rcParams['font.family'] = 'serif'
sns.countplot(x='internal_industry', data=IN, color='#c80d0f')
plt.title('Count by Industry Type')
plt.xlabel('Industry')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# --- Random Forest Model for Industry Prediction ---
X = IN[['zip', 'leasedsf', 'overall_rent', 'cbd_suburban', 'state']]
y = IN['internal_industry']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
plt.figure(figsize=(40, 20))
plot_tree(model.estimators_[0], feature_names=X.columns, max_depth=2, fontsize=20)
plt.title("Example Tree from Random Forest")
plt.show()

# Export tree to Graphviz format
single_tree = model.estimators_[0]
Names = ['Zip Code', 'Square Feet', 'Overall Rent', 'CBD/Suburban', 'State']
dot_data = export_graphviz(
    single_tree,
    feature_names=Names,
    class_names=[str(cls) for cls in model.classes_],
    rounded=True,
    impurity=False,
    max_depth=2
)
graph = graphviz.Source(dot_data)
graph.render(filename='random_forest_tree', format='png', cleanup=True)

# Feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(feature_importance_df.head(2))

# --- Random Forest Models for Specific Industries ---
for industry_code, label in [(1, 'Finance'), (3, 'Technology'), (8, 'Business'), (10, 'Legal')]:
    IN[f'is_{label.lower()}'] = IN['internal_industry'].apply(lambda x: 1 if x == industry_code else 0)
    X = IN[['zip', 'leasedsf', 'overall_rent', 'cbd_suburban', 'state']]
    if label == 'Legal':
        X = IN[['cbd_suburban', 'leasedsf', 'overall_rent', 'state', 'zip']]
    y = IN[f'is_{label.lower()}']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    print(f"{label} Accuracy:", accuracy_score(y_test, y_pred))
    print(feature_importance_df.head(2))

# --- Correlation Heatmap ---
sns.heatmap(LS[['overall_rent', 'leasedSF', 'available_space']].corr(),
            annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()
