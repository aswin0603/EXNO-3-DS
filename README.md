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











































# CODING AND OUTPUT:
       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
# RESULT:
       # INCLUDE YOUR RESULT HERE

       
