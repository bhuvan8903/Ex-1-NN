<H3>ENTER YOUR NAME : Bhuvaneshwaran H</H3>
<H3>ENTER YOUR REGISTER NO. 212223240018 </H3>
<H3>EX. NO.1</H3>
<H3>DATE:16.05.2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#import libraries
from google.colab import files
import pandas as pd
import io
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset from drive
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded[list(uploaded.keys())[0]]))


#Display first 5 rows
df.head()


# Finding Missing Values
print(df.isnull().sum())


#Handling Missing values


#Fill numerical missing values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

#Fill categorical missing values with mode
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
print("Missing values handled")


#Check for Duplicates
print("Duplicate Rows:", df.duplicated().sum())

#Remove duplicate rows
df = df.drop_duplicates()
print("Duplicates removed")


#Detect Outliers
#Using IQR Method
Q1 = df.select_dtypes(include=np.number).quantile(0.25)
Q3 = df.select_dtypes(include=np.number).quantile(0.75)
IQR = Q3 - Q1
outliers = ((df.select_dtypes(include=np.number) < (Q1 - 1.5 * IQR)) |
            (df.select_dtypes(include=np.number) > (Q3 + 1.5 * IQR)))
print("Outliers detected:")
print(outliers.sum())


#Normalize the dataset
#Selecting numerical columns
num_cols = df.select_dtypes(include=np.number).columns


#Min-Max Normalization
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print("Dataset normalized")


#split the dataset into input and output
#Assuming last column is target/output
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print("Input Features:")
print(X.head())
print("Output Variable:")
print(y.head())


#splitting the data for training & Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
print("Training and Testing data created")


#Print the training data and testing data
print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)
print("\nTraining Data:")
print(X_train.head())
print("\nTesting Data:")
print(X_test.head())

```

## OUTPUT:
SHOW YOUR OUTPUT HERE

### Read the dataset from drive

### Display first 5 rows

<img width="1644" height="346" alt="image" src="https://github.com/user-attachments/assets/f7b245b3-46df-429e-9678-8f02c5aa794f" />

### Finding Missing Values

<img width="255" height="373" alt="image" src="https://github.com/user-attachments/assets/bdaafbc2-dcba-4745-86f2-c1dcbe3b5be1" />


### Check for Duplicates
<img width="277" height="67" alt="image" src="https://github.com/user-attachments/assets/1a218d30-9bde-4fd6-8efd-7d53e92f575b" />

### Detect Outliers

<img width="274" height="298" alt="image" src="https://github.com/user-attachments/assets/30228bf8-4df5-4cfd-9768-cb960d4ece1c" />

### split the dataset into input and output

<img width="838" height="465" alt="image" src="https://github.com/user-attachments/assets/2e1728c2-752e-42d1-af3c-9b2d05c6d05e" />

### Print the training data and testing data
<img width="606" height="719" alt="image" src="https://github.com/user-attachments/assets/18409e8e-9c84-4810-8b29-92736b839090" />

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


