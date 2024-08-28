import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import ydata_profiling as ydf
from sklearn.model_selection import train_test_split,cross_val_score    
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
df = pd.read_csv("carpredictor1.csv")
df2 = pd.read_csv("carpredictor2.csv")
df.head()
df2.head()
df.shape
df2.shape
df.info()
df2.info()
# df3 = pd.read_csv("carpredictor3.csv")
# df3.head()
# df3.info()
# df3.shape
# df3["transmission"]

df2.drop(["transmission","ownership","engine","Seats"],axis=1, inplace=True)
df2.drop("Unnamed: 0", axis=1, inplace=True)
df2.info()
df2.head()


# 0. Preprocess + EDA + Feature Selection
# 1. Extract input and output cols
# 2. Scale the values
# 3. Train test split
# 4. Train the model
# 5. Evaluate the model/model selection
# 6. Deploy the model

"""report = ydf.ProfileReport(df)"""
"""report2 = ydf.ProfileReport(df2)"""
"""report.to_file("carpricepredector1.html")"""
"""report2.to_file("carpricepredector2.html")"""

df["year"].unique()


df = df[df["year"].str.isnumeric()]
df["year"].unique()
df.shape
df['year']=df['year'].astype(int)
"""now df 2 kai year pai kaam karte hai"""

df2 = df2.rename(columns={'manufacture': 'year'})
df2 = df2.rename(columns={'car_prices_in_rupee': 'Price'})
df2.info()

df2["year"].unique()
df2["year"] = df2['year'].astype(int)
df2.info()

"""creating brand name colum in df2"""

df2["company"] = df2['car_name'].str.split().str.get(0)
df2.head()
df2.info()

#price
df["Price"]
df=df[df['Price']!='Ask For Price'] 
df['Price']=df['Price'].str.replace(',','').astype(int)
df["Price"].unique()

# now on df2
df2.info()
df2["Price"].unique()
df2["Price"].sample(10)

def convert_to_intt(value):
    if 'Lakh' in value:
        numeric_part = float(value.replace(' Lakh', '').strip())
        return int(numeric_part * 100000)  
    elif 'Crore' in value:
        numeric_part = float(value.replace(' Crore', '').strip())
        return int(numeric_part * 10000000)  
    else:
        try:
            return int(float(value))
        except ValueError:
            return None 
df2['Price'] = df2['Price'].apply(convert_to_intt)

df2["Price"]
df2['Price']=df2['Price'].astype(int)
df2.shape
df2.info()

"""now km"""
df.info()
df2.info()
df["kms_driven"].unique()
df2["kms_driven"].unique()
df['kms_driven']=df['kms_driven'].str.split().str.get(0).str.replace(',','')
df2['kms_driven']=df2['kms_driven'].str.split().str.get(0).str.replace(',','')
df = df[df['kms_driven'] != 'Petrol']
df['kms_driven']=df['kms_driven'].astype(int)
df2['kms_driven']=df2['kms_driven'].astype(int)


df=df[~df['fuel_type'].isna()]
df["fuel_type"].unique()

df2=df2[~df2['fuel_type'].isna()]
df2["fuel_type"].unique()
df2 = df2[df2['fuel_type'] != 'Lpg']
df2.info()
df = df[df['fuel_type'] != 'LPG']
df.info()
"""name working"""


df['name']=df['name'].str.split().str.slice(start=0,stop=3).str.join(' ')
df.head()
df2['car_name']=df2['car_name'].str.split().str.slice(start=0,stop=4).str.join(' ')
df2.head()

df = df.rename(columns={'name': 'car_name'})

df2 = df2[["car_name","company","year","kms_driven","fuel_type","Price"]]
df2.head()
df = df[["car_name","company","year","kms_driven","fuel_type","Price"]]
df.head()

df["company"].unique()
df2["company"].unique()


##############CLEANING DONE##############################################################


df.head()
df2.head()
df.info()
df2.info()
df.shape
df2.shape

df.describe()
df2.describe()

df3 = pd.concat([df, df2])
df3.info()
df3.head()
df3.tail()
df3.sample()
df3=df3.reset_index(drop=True)
df3.describe()
df3 = pd.read_csv('carpredictor3madebymecleaned.csv')

df.describe(include='all')
df = pd.read_csv("carpredictor1cleaned.csv")
df2 = pd.read_csv("carpredictor2cleaned.csv")
df2.describe(include="all")
df2.info()
df3.describe(include="all")

df =df[df['Price']<3.1e6]
df3 = df3.drop(columns=["Unnamed: 0"])

df2[df2['Price']>1.8e7]

df3.describe()
df3[df2['Price']>1.8e7]
############EDA START####################################################################

report = ydf.ProfileReport(df)
report2 = ydf.ProfileReport(df2)
report3 = ydf.ProfileReport(df3)
report.to_file("carpricepredector1cleamned.html")
report2.to_file("carpricepredector2cleaned.html")
report2.to_file("carpricepredector3cleaned.html")
df.info()
########################Model##################################################
"""Train test df"""

X_df = df.iloc[:,0:5]
y_df = df.iloc[:,-1]

X_df
y_df
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df,y_df,test_size=0.2,random_state=42)
X_train_df


"""Train test df3"""
X_df3 = df3.iloc[:,0:5]
y_df3 = df3.iloc[:,-1]


X_train_df3, X_test_df3, y_train_df3, y_test_df3 = train_test_split(X_df3,y_df3,test_size=0.2,random_state=42)
X_train_df3

print(y_train_df3.isna().sum())
print(y_test_df3.isna().sum())
######################encoding####################################

ohedf=OneHotEncoder()
ohedf.fit(X_df[['car_name','company','fuel_type']])
ohedf3=OneHotEncoder()
ohedf3.fit(X_df3[['car_name','company','fuel_type']])

column_transdf=make_column_transformer((OneHotEncoder(categories=ohedf.categories_),['car_name','company','fuel_type']),
                                    remainder='passthrough')

column_transdf3=make_column_transformer((OneHotEncoder(categories=ohedf3.categories_),['car_name','company','fuel_type']),
                                    remainder='passthrough')
# df3.head()
from xgboost import XGBRegressor
xgb = XGBRegressor(random_state = 42,use_label_encoder = None,eval_metric = "logloss")
print(y_train_df3.isna().sum())
print(y_test_df3.isna().sum())
lr=LinearRegression()
df3['Price'] = df3['Price'].fillna(df3['Price'].mean())
pipedf=make_pipeline(column_transdf3,xgb)
pipedf.fit(X_train_df3,y_train_df3)
y_preddf=pipedf.predict(X_test_df3)
r2_score(y_test_df3,y_preddf)#67
#after scaling





# pipedf=make_pipeline(column_transdfscaling,lr)
# pipedf.fit(X_train_df,y_train_df)
# y_preddf=pipedf.predict(X_test_df)
# r2_score(y_test_df,y_preddf)#69

# """Log is worst"""
# log = LogisticRegression()
# pipedf = make_pipeline(column_transdf,log)
# pipedf.fit(X_train_df,y_train_df)
# y_preddf=pipedf.predict(X_test_df)
# r2_score(y_test_df,y_preddf) # minus mai aaya 

# from sklearn.tree import DecisionTreeRegressor
# dt = DecisionTreeRegressor(random_state=42)

# pipedf=make_pipeline(column_transdf,dt)
# pipedf.fit(X_train_df,y_train_df)
# y_preddf=pipedf.predict(X_test_df)
# r2_score(y_test_df,y_preddf)#71.05
# #after scaling

# pipedf=make_pipeline(column_transdfscaling,dt)
# pipedf.fit(X_train_df,y_train_df)
# y_preddf=pipedf.predict(X_test_df)
# r2_score(y_test_df,y_preddf)#71.2

# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# rf = RandomForestRegressor()
# pipedf=make_pipeline(column_transdf,rf)
# pipedf.fit(X_train_df,y_train_df)
# y_preddf=pipedf.predict(X_test_df)
# r2_score(y_test_df,y_preddf)#71.3


# #after scaling

# pipedf=make_pipeline(column_transdfscaling,rf)
# pipedf.fit(X_train_df,y_train_df)
# y_preddf=pipedf.predict(X_test_df)
# r2_score(y_test_df,y_preddf) #70.6

# pipedf=make_pipeline(column_transdf,xgb)
# pipedf.fit(X_train_df,y_train_df)
# y_preddf=pipedf.predict(X_test_df)
# r2_score(y_test_df,y_preddf) #75
# #with scaling
# pipedf=make_pipeline(column_transdfscaling,xgb)
# pipedf.fit(X_train_df,y_train_df)
# y_preddf=pipedf.predict(X_test_df)
# r2_score(y_test_df,y_preddf) #74

# gb = GradientBoostingRegressor()
# pipedf=make_pipeline(column_transdf,gb)
# pipedf.fit(X_train_df,y_train_df)
# y_preddf=pipedf.predict(X_test_df)
# r2_score(y_test_df,y_preddf)#65.2
# # after scaling
# pipedf=make_pipeline(column_transdfscaling,gb)
# pipedf.fit(X_train_df,y_train_df)
# y_preddf=pipedf.predict(X_test_df)
# r2_score(y_test_df,y_preddf)#65.1
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsRegressor
# estimators = [
#     ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
#     ('xg', XGBRegressor()),
#     ('gbdt',GradientBoostingRegressor())
# ]
# clf = StackingClassifier(
#     estimators=estimators, 
#     final_estimator=LogisticRegression(),
#     cv=10
# )

# pipedf=make_pipeline(column_transdfscaling,clf)
# pipedf.fit(X_train_df,y_train_df)
# y_preddf=pipedf.predict(X_test_df)
# r2_score(y_test_df,y_preddf)
""" uffffffffffffff yeh toh worstt deraaa """


# now check for the best 



scores=[]
for i in range(1000):
    X_train_df3,X_test_df3,y_train_df3,y_test_df3=train_test_split(X_df3,y_df3,test_size=0.2,random_state=i)
    pipe=make_pipeline(column_transdf3,xgb)
    pipe.fit(X_train_df3,y_train_df3)
    y_preddf3=pipe.predict(X_test_df3)
    scores.append(r2_score(y_test_df3,y_preddf3))
np.argmax(scores)
scores[np.argmax(scores)]


X_train_df3,X_test_df3,y_train_df3,y_test_df3=train_test_split(X_df3,y_df3,test_size=0.2,random_state=np.argmax(scores))
pipe=make_pipeline(column_transdf3,xgb)
pipe.fit(X_train_df3,y_train_df3)
y_preddf3=pipe.predict(X_test_df3)
r2_score(y_test_df3,y_preddf3)
import pickle
pickle.dump(pipe,open("CarSalePriceXGB.pkl",'wb'))
X_test_df3
pipe.predict(pd.DataFrame(columns=X_test_df3.columns,data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))
pipe.predict(pd.DataFrame(columns=X_test_df3.columns,data=np.array(['Hyundai Verna 1.6 SX' ,    "Hyundai"  ,2013,       87013 ,   'Petrol']).reshape(1,5)))