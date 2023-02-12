# Overview

This project analyses the housing in Kings County to identify what features have a strong relationship with price. 

# Data Understanding and Preparation


```python
# Import Standard Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
from statsmodels.formula.api import ols


sales = pd.read_csv('data/kc_house_data.csv')
pd.options.display.float_format = '{:,.0f}'.format
sales.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221,900</td>
      <td>3</td>
      <td>1</td>
      <td>1180</td>
      <td>5650</td>
      <td>1</td>
      <td>nan</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0</td>
      <td>98178</td>
      <td>48</td>
      <td>-122</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538,000</td>
      <td>3</td>
      <td>2</td>
      <td>2570</td>
      <td>7242</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1,991</td>
      <td>98125</td>
      <td>48</td>
      <td>-122</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180,000</td>
      <td>2</td>
      <td>1</td>
      <td>770</td>
      <td>10000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>nan</td>
      <td>98028</td>
      <td>48</td>
      <td>-122</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604,000</td>
      <td>4</td>
      <td>3</td>
      <td>1960</td>
      <td>5000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0</td>
      <td>98136</td>
      <td>48</td>
      <td>-122</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510,000</td>
      <td>3</td>
      <td>2</td>
      <td>1680</td>
      <td>8080</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0</td>
      <td>98074</td>
      <td>48</td>
      <td>-122</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
sales.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 21 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   id             21597 non-null  int64  
     1   date           21597 non-null  object 
     2   price          21597 non-null  float64
     3   bedrooms       21597 non-null  int64  
     4   bathrooms      21597 non-null  float64
     5   sqft_living    21597 non-null  int64  
     6   sqft_lot       21597 non-null  int64  
     7   floors         21597 non-null  float64
     8   waterfront     19221 non-null  float64
     9   view           21534 non-null  float64
     10  condition      21597 non-null  int64  
     11  grade          21597 non-null  int64  
     12  sqft_above     21597 non-null  int64  
     13  sqft_basement  21597 non-null  object 
     14  yr_built       21597 non-null  int64  
     15  yr_renovated   17755 non-null  float64
     16  zipcode        21597 non-null  int64  
     17  lat            21597 non-null  float64
     18  long           21597 non-null  float64
     19  sqft_living15  21597 non-null  int64  
     20  sqft_lot15     21597 non-null  int64  
    dtypes: float64(8), int64(11), object(2)
    memory usage: 3.5+ MB
    


```python
#check for any NAN value
sales.isna().sum()
```




    id                  0
    date                0
    price               0
    bedrooms            0
    bathrooms           0
    sqft_living         0
    sqft_lot            0
    floors              0
    waterfront       2376
    view               63
    condition           0
    grade               0
    sqft_above          0
    sqft_basement       0
    yr_built            0
    yr_renovated     3842
    zipcode             0
    lat                 0
    long                0
    sqft_living15       0
    sqft_lot15          0
    dtype: int64



Based on the above, Waterfront and Yr Renovated require data cleasning 


```python
#drop columns that are not useful to the purpose of this data analysis
sales.drop(sales.iloc[:, 16:21], inplace=True, axis=1)
sales.drop(sales.iloc[:, 12:14], inplace=True, axis=1)
sales.drop(sales.iloc[:, 9:10], inplace=True, axis=1)
sales.drop(sales.iloc[:, 0:2], inplace=True, axis=1)

#replace NAN value 
sales["waterfront"] = sales["waterfront"].fillna(0)
sales["yr_renovated"] = sales["yr_renovated"].fillna(0)
#add "age" column 
sales['age']=[2023] - sales.loc[:,"yr_built"]

```


```python
sales.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 12 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   price         21597 non-null  float64
     1   bedrooms      21597 non-null  int64  
     2   bathrooms     21597 non-null  float64
     3   sqft_living   21597 non-null  int64  
     4   sqft_lot      21597 non-null  int64  
     5   floors        21597 non-null  float64
     6   waterfront    21597 non-null  float64
     7   condition     21597 non-null  int64  
     8   grade         21597 non-null  int64  
     9   yr_built      21597 non-null  int64  
     10  yr_renovated  21597 non-null  float64
     11  age           21597 non-null  int64  
    dtypes: float64(5), int64(7)
    memory usage: 2.0 MB
    

# Base Model 


```python
X = sales.drop('price', axis=1)
y=sales['price']
import statsmodels.api as sm
X_int = sm.add_constant(X)
model = sm.OLS(y,X_int).fit()
model.summary()

```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.646</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.646</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   3936.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 12 Feb 2023</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>17:49:56</td>     <th>  Log-Likelihood:    </th> <td>-2.9618e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21597</td>      <th>  AIC:               </th>  <td>5.924e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21586</td>      <th>  BIC:               </th>  <td>5.925e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    10</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>        <td>    1.3376</td> <td>    0.033</td> <td>   41.137</td> <td> 0.000</td> <td>    1.274</td> <td>    1.401</td>
</tr>
<tr>
  <th>bedrooms</th>     <td>-4.238e+04</td> <td> 2051.101</td> <td>  -20.661</td> <td> 0.000</td> <td>-4.64e+04</td> <td>-3.84e+04</td>
</tr>
<tr>
  <th>bathrooms</th>    <td> 4.935e+04</td> <td> 3480.907</td> <td>   14.178</td> <td> 0.000</td> <td> 4.25e+04</td> <td> 5.62e+04</td>
</tr>
<tr>
  <th>sqft_living</th>  <td>  177.3030</td> <td>    3.303</td> <td>   53.682</td> <td> 0.000</td> <td>  170.829</td> <td>  183.777</td>
</tr>
<tr>
  <th>sqft_lot</th>     <td>   -0.2437</td> <td>    0.037</td> <td>   -6.626</td> <td> 0.000</td> <td>   -0.316</td> <td>   -0.172</td>
</tr>
<tr>
  <th>floors</th>       <td> 2.055e+04</td> <td> 3461.733</td> <td>    5.936</td> <td> 0.000</td> <td> 1.38e+04</td> <td> 2.73e+04</td>
</tr>
<tr>
  <th>waterfront</th>   <td> 7.518e+05</td> <td> 1.84e+04</td> <td>   40.893</td> <td> 0.000</td> <td> 7.16e+05</td> <td> 7.88e+05</td>
</tr>
<tr>
  <th>condition</th>    <td>  2.02e+04</td> <td> 2515.446</td> <td>    8.031</td> <td> 0.000</td> <td> 1.53e+04</td> <td> 2.51e+04</td>
</tr>
<tr>
  <th>grade</th>        <td> 1.299e+05</td> <td> 2155.845</td> <td>   60.276</td> <td> 0.000</td> <td> 1.26e+05</td> <td> 1.34e+05</td>
</tr>
<tr>
  <th>yr_built</th>     <td> -535.6120</td> <td>    8.685</td> <td>  -61.669</td> <td> 0.000</td> <td> -552.636</td> <td> -518.588</td>
</tr>
<tr>
  <th>yr_renovated</th> <td>   14.3579</td> <td>    4.309</td> <td>    3.332</td> <td> 0.001</td> <td>    5.913</td> <td>   22.803</td>
</tr>
<tr>
  <th>age</th>          <td> 3241.6102</td> <td>   67.284</td> <td>   48.178</td> <td> 0.000</td> <td> 3109.728</td> <td> 3373.493</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>15847.393</td> <th>  Durbin-Watson:     </th>  <td>   1.976</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>1006948.729</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 2.936</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>35.932</td>   <th>  Cond. No.          </th>  <td>2.06e+18</td>  
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 9.92e-24. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
import scipy.stats as stats
residuals = model.resid
sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
plt.show()
```


    
![png](output_11_0.png)
    


# Transformation and validation


```python
#Check categorical variables

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16,10), sharey=True)

categoricals = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'condition', 'grade']

for col, ax in zip(categoricals, axes.flatten()):
    (sales.groupby(col)               # group values together by column of interest
         .mean()['price']        # take the mean of the saleprice for each group
         .sort_values()              # sort the groups in ascending order
         .plot
         .bar(ax=ax))                # create a bar graph on the ax
    
    ax.set_title(col)                # Make the title the name of the column
    
fig.tight_layout()
```


    
![png](output_13_0.png)
    



```python
#drop 'waterfront' "Floors" and "conditions" as they do not show a significant relationship to price
sales.drop(sales.iloc[:, 9:11], inplace=True, axis=1)
sales.drop(sales.iloc[:, 7:8], inplace=True, axis=1)
sales.drop(sales.iloc[:, 6:7], inplace=True, axis=1)

sales.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>grade</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221,900</td>
      <td>3</td>
      <td>1</td>
      <td>1180</td>
      <td>5650</td>
      <td>1</td>
      <td>7</td>
      <td>68</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538,000</td>
      <td>3</td>
      <td>2</td>
      <td>2570</td>
      <td>7242</td>
      <td>2</td>
      <td>7</td>
      <td>72</td>
    </tr>
    <tr>
      <th>2</th>
      <td>180,000</td>
      <td>2</td>
      <td>1</td>
      <td>770</td>
      <td>10000</td>
      <td>1</td>
      <td>6</td>
      <td>90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604,000</td>
      <td>4</td>
      <td>3</td>
      <td>1960</td>
      <td>5000</td>
      <td>1</td>
      <td>7</td>
      <td>58</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510,000</td>
      <td>3</td>
      <td>2</td>
      <td>1680</td>
      <td>8080</td>
      <td>1</td>
      <td>8</td>
      <td>36</td>
    </tr>
  </tbody>
</table>
</div>




```python
#split bedrooms, bathrooms and grade into 2 groups to create dummies and drop the original columns 

sales['beds'] = ['1-4beds' if x <=4 else '4+beds' for x in sales['bedrooms']]
sales['baths']=['1-5baths' if x <=5 else '5+baths' for x in sales['bathrooms']]
sales['quality']=['low' if x <=8 else 'high' for x in sales['grade']]
sales= sales.drop(['bedrooms','bathrooms','floors','grade'], axis=1)

sales.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>age</th>
      <th>beds</th>
      <th>baths</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221,900</td>
      <td>1180</td>
      <td>5650</td>
      <td>68</td>
      <td>1-4beds</td>
      <td>1-5baths</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538,000</td>
      <td>2570</td>
      <td>7242</td>
      <td>72</td>
      <td>1-4beds</td>
      <td>1-5baths</td>
      <td>low</td>
    </tr>
    <tr>
      <th>2</th>
      <td>180,000</td>
      <td>770</td>
      <td>10000</td>
      <td>90</td>
      <td>1-4beds</td>
      <td>1-5baths</td>
      <td>low</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604,000</td>
      <td>1960</td>
      <td>5000</td>
      <td>58</td>
      <td>1-4beds</td>
      <td>1-5baths</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510,000</td>
      <td>1680</td>
      <td>8080</td>
      <td>36</td>
      <td>1-4beds</td>
      <td>1-5baths</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>




```python
#create dummies: 

categoricals = ['beds', 'baths', 'quality']
sales_ohe=pd.get_dummies(sales[categoricals], prefix=categoricals, drop_first=True)
```


```python
# check distribution of the continuous variables 

continuous = ['sqft_living','sqft_lot','age','price']
sales_cont = sales[continuous]
pd.plotting.scatter_matrix(sales_cont, figsize=(10,12));
```


    
![png](output_17_0.png)
    



```python
# log features
log_names = [f'{column}_log' for column in sales_cont.columns]

sales_log = np.log(sales_cont)
sales_log.columns = log_names

# normalize (subract mean and divide by std)

def normalize(feature):
    return (feature - feature.mean()) / feature.std()

sales_log_norm = sales_log.apply(normalize)
```


```python
pd.plotting.scatter_matrix(sales_log_norm, figsize=(10,12));
```


    
![png](output_19_0.png)
    



```python
preprocessed = pd.concat([sales_log_norm, sales_ohe], axis=1)
preprocessed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sqft_living_log</th>
      <th>sqft_lot_log</th>
      <th>age_log</th>
      <th>price_log</th>
      <th>beds_4+beds</th>
      <th>baths_5+baths</th>
      <th>quality_low</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>-0</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2</td>
      <td>0</td>
      <td>1</td>
      <td>-2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0</td>
      <td>0</td>
      <td>-0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#rerun the model 
X = preprocessed.drop('price_log', axis=1)
y = preprocessed['price_log']
```


```python
import statsmodels.api as sm
X_int = sm.add_constant(X)
model = sm.OLS(y,X_int).fit()
model.summary()

```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>price_log</td>    <th>  R-squared:         </th> <td>   0.549</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.549</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   4379.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 12 Feb 2023</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>17:50:02</td>     <th>  Log-Likelihood:    </th> <td> -22047.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21597</td>      <th>  AIC:               </th> <td>4.411e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21590</td>      <th>  BIC:               </th> <td>4.416e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     6</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>           <td>    0.6483</td> <td>    0.012</td> <td>   52.155</td> <td> 0.000</td> <td>    0.624</td> <td>    0.673</td>
</tr>
<tr>
  <th>sqft_living_log</th> <td>    0.6116</td> <td>    0.006</td> <td>   96.263</td> <td> 0.000</td> <td>    0.599</td> <td>    0.624</td>
</tr>
<tr>
  <th>sqft_lot_log</th>    <td>   -0.1587</td> <td>    0.005</td> <td>  -31.369</td> <td> 0.000</td> <td>   -0.169</td> <td>   -0.149</td>
</tr>
<tr>
  <th>age_log</th>         <td>    0.2048</td> <td>    0.005</td> <td>   39.568</td> <td> 0.000</td> <td>    0.195</td> <td>    0.215</td>
</tr>
<tr>
  <th>beds_4+beds</th>     <td>   -0.0541</td> <td>    0.017</td> <td>   -3.167</td> <td> 0.002</td> <td>   -0.088</td> <td>   -0.021</td>
</tr>
<tr>
  <th>baths_5+baths</th>   <td>    0.9173</td> <td>    0.104</td> <td>    8.852</td> <td> 0.000</td> <td>    0.714</td> <td>    1.120</td>
</tr>
<tr>
  <th>quality_low</th>     <td>   -0.8034</td> <td>    0.014</td> <td>  -55.903</td> <td> 0.000</td> <td>   -0.832</td> <td>   -0.775</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.610</td> <th>  Durbin-Watson:     </th> <td>   1.968</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.737</td> <th>  Jarque-Bera (JB):  </th> <td>   0.586</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.005</td> <th>  Prob(JB):          </th> <td>   0.746</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.024</td> <th>  Cond. No.          </th> <td>    30.5</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
import scipy.stats as stats
residuals = model.resid
sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
plt.show()
```


    
![png](output_23_0.png)
    



```python
#drop ages and check if it has improved the model R square 
preprocessed= preprocessed.drop(['age_log'], axis=1)

```


```python
X = preprocessed.drop('price_log', axis=1)
y = preprocessed['price_log']
import statsmodels.api as sm
X_int = sm.add_constant(X)
model = sm.OLS(y,X_int).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>price_log</td>    <th>  R-squared:         </th> <td>   0.516</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.516</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   4608.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 12 Feb 2023</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>17:50:02</td>     <th>  Log-Likelihood:    </th> <td> -22803.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21597</td>      <th>  AIC:               </th> <td>4.562e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21591</td>      <th>  BIC:               </th> <td>4.567e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>           <td>    0.5613</td> <td>    0.013</td> <td>   44.298</td> <td> 0.000</td> <td>    0.536</td> <td>    0.586</td>
</tr>
<tr>
  <th>sqft_living_log</th> <td>    0.5399</td> <td>    0.006</td> <td>   85.608</td> <td> 0.000</td> <td>    0.527</td> <td>    0.552</td>
</tr>
<tr>
  <th>sqft_lot_log</th>    <td>   -0.1005</td> <td>    0.005</td> <td>  -20.049</td> <td> 0.000</td> <td>   -0.110</td> <td>   -0.091</td>
</tr>
<tr>
  <th>beds_4+beds</th>     <td>    0.0122</td> <td>    0.018</td> <td>    0.691</td> <td> 0.489</td> <td>   -0.022</td> <td>    0.047</td>
</tr>
<tr>
  <th>baths_5+baths</th>   <td>    0.9037</td> <td>    0.107</td> <td>    8.421</td> <td> 0.000</td> <td>    0.693</td> <td>    1.114</td>
</tr>
<tr>
  <th>quality_low</th>     <td>   -0.7024</td> <td>    0.015</td> <td>  -47.955</td> <td> 0.000</td> <td>   -0.731</td> <td>   -0.674</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>67.267</td> <th>  Durbin-Watson:     </th> <td>   1.975</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  65.111</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.116</td> <th>  Prob(JB):          </th> <td>7.27e-15</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.862</td> <th>  Cond. No.          </th> <td>    30.2</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
import scipy.stats as stats
residuals = model.resid
sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
plt.show()
```


    
![png](output_26_0.png)
    



```python
preprocessed= preprocessed.drop(['beds_4+beds'], axis=1)
X = preprocessed.drop('price_log', axis=1)
y = preprocessed['price_log']

```


```python
import statsmodels.api as sm
X_int = sm.add_constant(X)
model = sm.OLS(y,X_int).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>price_log</td>    <th>  R-squared:         </th> <td>   0.516</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.516</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   5760.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 12 Feb 2023</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>17:50:02</td>     <th>  Log-Likelihood:    </th> <td> -22803.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21597</td>      <th>  AIC:               </th> <td>4.562e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21592</td>      <th>  BIC:               </th> <td>4.566e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>           <td>    0.5617</td> <td>    0.013</td> <td>   44.398</td> <td> 0.000</td> <td>    0.537</td> <td>    0.587</td>
</tr>
<tr>
  <th>sqft_living_log</th> <td>    0.5411</td> <td>    0.006</td> <td>   89.843</td> <td> 0.000</td> <td>    0.529</td> <td>    0.553</td>
</tr>
<tr>
  <th>sqft_lot_log</th>    <td>   -0.1006</td> <td>    0.005</td> <td>  -20.074</td> <td> 0.000</td> <td>   -0.110</td> <td>   -0.091</td>
</tr>
<tr>
  <th>baths_5+baths</th>   <td>    0.9087</td> <td>    0.107</td> <td>    8.487</td> <td> 0.000</td> <td>    0.699</td> <td>    1.119</td>
</tr>
<tr>
  <th>quality_low</th>     <td>   -0.7016</td> <td>    0.015</td> <td>  -48.041</td> <td> 0.000</td> <td>   -0.730</td> <td>   -0.673</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>67.763</td> <th>  Durbin-Watson:     </th> <td>   1.975</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  65.605</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.116</td> <th>  Prob(JB):          </th> <td>5.68e-15</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.862</td> <th>  Cond. No.          </th> <td>    30.1</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.






```python
import scipy.stats as stats
residuals = model.resid
sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
plt.show()
```


    
![png](output_30_0.png)
    


QQ plot shows that the normality assumption of the residuals seems fulfilled.


```python
fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "sqft_living_log", fig=fig)
plt.show()

```


    
![png](output_32_0.png)
    


From the first and second plot in the first row, we see the residuals appear to be equal across the regression line which is a sign of Homoscedasticity. 


```python
fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "sqft_lot_log", fig=fig)
plt.show()
```


    
![png](output_34_0.png)
    


However, for the Lot area,  the residuals do not appear to be equal across the regression line which is a sign of heteroscedasticity i.e. the residuals are heteroscedastic. This violates an assumption.

Train-Test Split:
Perform a train-test split with a test set of 20% and a random state of 4.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```


```python
#Fit a linear regression model on the training set

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

```




    LinearRegression()




```python
#Generate Predictions on Training and Test Sets
y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)
```


```python
#Calculate the Mean Squared Error (MSE)
from sklearn.metrics import mean_squared_error
train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
print('Train Mean Squared Error:', train_mse)
print('Test Mean Squared Error: ', test_mse)
```

    Train Mean Squared Error: 0.48344525331522825
    Test Mean Squared Error:  0.48512251743393336
    

The difference between Test MSE and Train MSE is quite small (appx 0.35%)


```python
##Evaluate the effect of train tesst split size: 
#Iterate over a range of train-test split sizes from .5 to .9
#For each train-test split size, generate 10 iterations of models/errors and save the average train/test error. 
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

categoricals = ['baths_5+baths', 'quality_low']
continuous = ['sqft_living_log','sqft_lot_log']

log_transformer = FunctionTransformer(np.log, validate=True)
ohe = OneHotEncoder(drop='first', sparse=False)

train_mses = []
test_mses = []

t_sizes = np.linspace(0.5, 0.9, 10)
for t_size in t_sizes:
    
    inner_train_mses = []
    inner_test_mses = []
    for i in range(10):
        # Create new split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=i)

        # Fit model
        linreg.fit(X_train, y_train)

        # Append metrics to their respective lists
        y_hat_train = linreg.predict(X_train)
        y_hat_test = linreg.predict(X_test)
        inner_train_mses.append(mean_squared_error(y_train, y_hat_train))
        inner_test_mses.append(mean_squared_error(y_test, y_hat_test))

    train_mses.append(np.mean(inner_train_mses))
    test_mses.append(np.mean(inner_test_mses))

fig, ax = plt.subplots()
ax.scatter(t_sizes, train_mses, label='Average Training Error')
ax.scatter(t_sizes, test_mses, label='Average Testing Error')
ax.legend();
```


    
![png](output_42_0.png)
    


As above, the average testing error is generally lower than the training error. This suggests that the model may generalize well to future cases.

# Interpretation 

The model summary shows that the R-squared value is 0.516 i.e. 51.6% of the house price can be explained by the dependent variables. 

sqft_living_log - footage of the home; ; sqft_lot_log - footage of the lot; number of bathrooms and grade are found to tatistically significant to the pricing of a property in King County. 

Footage of the home and the number of bathrooms have a positive relationship with the house price i.e. # increase in home footage or a home with more than 5 bathroom results in an increase in the house price. Similarly, quality_low - grade - a lower grade of a property is associated with a lower value in house price. 

Wheras, sqft_lot_log - footage of the lot has an inversed relationship with the house price. Increase in footage of the lot is associated with a decrease in price of the house. 

Model validation: 
QQ plot shows that the normality assumption of the residuals seems fulfilled. 

In further analysis, for living footage, we can see the residuals appear to be equal across the regression line which is a sign of Homoscedasticity. However, for the Lot area, the residuals do not appear to be equal across the regression line which is a sign of heteroscedasticity i.e. the residuals are heteroscedastic. Therefore, living footage may be a better indicator/feature to be used. 

The first train test shows that the difference between Test MSE and Train MSE is quite small (appx 0.35%). Similarly when run the train over 10 iterations over a range of train-test split sizes from .5 to .9, the Testing Error are better than the training error and the difference is also quite small. This suggests that the model is fitting well. 
