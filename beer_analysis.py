import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as st
import datetime

beers_table = pd.read_csv('beers_data.csv')

beers_table['Style'] = pd.to_numeric(beers_table['Style'].str.replace('American IPARanked #', ''), errors = 'coerce')
beers_table.rename(columns = {'Style' : 'Ranking'}, inplace = True)

beers_table['ABV'] = pd.to_numeric(beers_table['ABV'].str.replace('%', ''), errors = 'coerce')/100
beers_table['Score'] = pd.to_numeric(beers_table['Score'].str.split('R', n = 1).str[0])

beers_table[['Avg', 'Deviation']] = beers_table['Avg'].str.split('|',expand = True)
beers_table['Avg'] = beers_table['Avg'].astype(float)
beers_table['Deviation'] = pd.to_numeric(beers_table['Deviation'].str.replace('pDev:' , '', regex = True).str.replace( '%', '', regex = True))

beers_table[['Ratings','Reviews']] = beers_table['Ratings'].str.split('|', expand = True)
beers_table['Ratings'] = beers_table['Ratings'].str.replace(',', '', regex = True).astype(float)
beers_table['Status'] = (beers_table['Status'] == 'Active').astype(int)

beers_table = beers_table.drop(columns = ['Rated'])

beers_table['Added'] = pd.to_datetime(beers_table['Added'], errors = 'coerce')
today = datetime.datetime.now()
beers_table['Days_on_site'] = (today - beers_table['Added']).dt.days

beers_table['Wants'] =  beers_table['Wants'].str.replace(',', '', regex = True).astype(int)
beers_table['Gots'] =  beers_table['Gots'].str.replace(',', '', regex = True).astype(int)

beers_table['Reviews'] = beers_table['Reviews'].str.replace('reviews:', '', regex = True).str.replace(',', '', regex = True).astype(int)
beers_table['Country'] = pd.Series(dtype='str')


for index, row in beers_table.iterrows():
    if ',' in row['Region']:
        beers_table.loc[index, 'State'] = row['Region'].split(',')[0].strip()
        beers_table.loc[index, 'Country'] = row['Region'].split(',')[1].strip()
    else:
        beers_table.loc[index, 'State'] = np.nan
        beers_table.loc[index, 'Country'] = row['Region']
beers_table.drop(columns = ['Region', 'Ranking'], inplace =True)

# Check for missing values
print(beers_table.isnull().sum())
# thowing rows where ABV is missing, because the number of them is smaller than 5% of the total dataset
beers_table = beers_table.dropna(subset = ['ABV'])

beers_table.info()

beers_table['Hype'] = beers_table['Wants'] / (beers_table['Gots'] + 1)
model_features = ['ABV', 'Ratings', 'Status', 'Hype', 'Deviation','Days_on_site']

correlation_matrix = beers_table[model_features + ['Avg']].corr()
corrplot = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix') 
plt.show()

numeric_features = ['ABV', 'Score','Avg', 'Ratings', 'Hype', 'Deviation','Days_on_site']
beers_table[numeric_features].describe().round(3)

for feature in numeric_features:
    first_quartile = beers_table[feature].quantile(0.25)
    third_quartile = beers_table[feature].quantile(0.75)
    interquartile_range = third_quartile - first_quartile
    lower = first_quartile - 1.5 * interquartile_range
    upper = third_quartile + 1.5 * interquartile_range
    plt.figure(figsize = (4,2))
    sns.histplot(beers_table[(beers_table[feature] > lower) & (beers_table[feature] < upper)][feature],kde = True, bins = 10)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.show()



from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb


X = beers_table[model_features]   
y = beers_table['Avg'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 324)
scaler = StandardScaler()
scaler.set_output(transform = 'pandas')

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Support Vector Regression (SVR)": SVR(),
    "Random Forest": RandomForestRegressor( random_state=63, n_jobs=-1),
    "XGBoost": xgb.XGBRegressor( random_state=90, n_jobs=-1),
    "LightGBM": lgb.LGBMRegressor( random_state=12, verbose=-1, n_jobs=-1)
}

param_grids = {
    "Ridge Regression": {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    },
    "Lasso Regression": {
        'alpha': [0.001, 0.01, 0.1, 1.0]
    },
    "Support Vector Regression (SVR)": {
        'kernel': ['rbf', 'linear'],
        'C': [0.1, 1.0, 10.0],
        'epsilon': [0.01, 0.1, 0.2]
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9]
    },
    "LightGBM": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [-1, 5, 10]
    },
    "Linear Regression" :
    {}
}

results = []

for name, model in models.items():
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=5,                     
        scoring='neg_mean_squared_error', 
        n_jobs=-1                  
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    
    if name == "Lasso Regression":
        print("\n=== Coefficients Estimated by the Lasso Model ===")
        for col, coef in zip(X_train.columns, best_model.coef_):
            print(f"{col:<15} : {coef:.4f}")
        print("="*47 + "\n")

    predictions = best_model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    results.append({
        "Model": name,
        "Best Params": grid_search.best_params_,  
        "MSE": round(mse, 4),
        "R2 Score": round(r2, 4)
    })

df_grid_results = pd.DataFrame(results).sort_values(by="MSE")
print("\n=== Result of best estimators ===")
for index, row in df_grid_results.iterrows():
    print(f"\nModel: {row['Model']}")
    print(f"Best Params: {row['Best Params']}")
    print(f"MSE: {row['MSE']} | R2 Score: {row['R2 Score']}")





def calculate_lower_CI(row, confidence_level = 0.95):
    sigma = row['Deviation']
    mu = row['Avg']
    n = row['Ratings']

    if n < 2 or sigma == 0:
        return 0
    
    std_err = sigma / np.sqrt(n)

    lower_bound = st.t.interval(confidence_level, df = n-1, loc = mu, scale = std_err)[0]
    return max(0,lower_bound)


beers_table['lower_CI_score'] = beers_table.apply(calculate_lower_CI, axis = 1)

good_beers = beers_table.sort_values(by = 'lower_CI_score', ascending = False)
print(good_beers.head(10))
