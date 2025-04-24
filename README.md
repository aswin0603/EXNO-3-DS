## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
### 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  ### 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```python
import pandas as pd
df = pd.read_csv('/content/Encoding Data.csv')
df
```
![image](https://github.com/user-attachments/assets/95084145-686f-45e0-b3df-d68f46532ac0)

```python
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm = ['Hot', 'Warm', 'Cold']
e1 = OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/756e82eb-13f9-42a1-8a6a-28139837d842)

```python
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/2cc96097-514a-407f-9cea-dbbe0be850c9)

```python
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/e167c289-b637-4a32-b646-be24243953b2)

```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output = False)
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/59c62032-e061-4f63-aea6-35d3293c386a)

```python
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/24b130dd-d4f0-4ee5-be46-e038e9edf975)

```python
pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![image](https://github.com/user-attachments/assets/5aea9959-4d9a-4237-a439-8525df5311b4)

```python
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![image](https://github.com/user-attachments/assets/d873d98d-9d18-49af-9b23-29520d2e5236)

```python
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/33faf187-69d0-41c0-b944-d5130a5e0c70)

```python
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/d30f02da-7e24-455a-b252-4e46511122af)

```python
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/f3fe465b-0b9f-4a85-831d-a14a60f90212)

```python
df.skew()
```
![image](https://github.com/user-attachments/assets/f47f0857-117e-414d-aa33-da15e5cc5389)

```python
np.log(df["Highly Positiove Skew"])
```
![image](https://github.com/user-attachments/assets/c8ee33b8-86be-4470-954e-9418a4ee7fa5)

```python
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/1ca04769-bb8f-49c7-aac5-20d309aedffe)

```python
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/07fed1e2-dafc-4a27-a36c-3895946fe4bd)

```python
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/6e7037fe-1948-4ef9-b25b-126c0f792aaa)

```python
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/91028290-b775-49a4-8175-5b326477936d)

```python
df.skew()
```
![image](https://github.com/user-attachments/assets/db702350-f8b8-4d53-a800-6f7441db1028)

```python
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/5154c639-0786-404f-a272-8e75b6ca285e)

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/c478c52c-9b55-4459-a759-2fbea05ef3f6)

```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/56281ab2-7f7a-4ec6-be0d-817474af8eef)

```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/4913e7c8-e456-4238-bd8f-b865e731162e)

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/a1751655-8884-474b-b05a-47318191eb3e)

```python
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/acc341a3-eb42-4f95-93c9-047cc9f5ae1c)

```python
dt=pd.read_csv("titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/a7f3664d-b71a-44dd-a72d-59f9dec136c5)

```python
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/174e02ef-4a08-40ea-b516-05b72fc956e5)


# RESULT:
Thus the given data is read and performed Feature Encoding and Transformation process and saved the data to the file.

## NAME : ASWIN B
## Register Number : 212224110007
