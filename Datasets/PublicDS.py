import re
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, PowerTransformer, OneHotEncoder


from Utils import Util

warnings.filterwarnings('ignore')





def get_raisin_ds(path, seed):
    #https://www.kaggle.com/datasets/nimapourmoradi/raisin-binary-classification?resource=download
    df = pd.read_csv(path, header=0)  # Ensure the first row is treated as the header

    # Drop the first row which contains the actual header (if needed)
    df = df.drop(0)
    df.Class = [1 if each == "Kecimen" else 0 for each in df.Class]

    df = shuffle(df, random_state=seed)

    # print(df.head())
    # print(df.info())
    # print()
    # print(df["Class"].value_counts())
    # df.Class = [1 if each == "Kecimen" else 0 for each in data.Class]
    # print(df["Class"])

    # Separate features (X) and target (y)
    X = df.drop('Class', axis=1).values  # Convert X to a 2D array (list of lists)
    y = df['Class'].values  # Convert y to a 1D array (list)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X,y





# def get_cats_dogs_ds(base_path, seed):
#     # Define paths for cats and dogs
#     cat_path = os.path.join(base_path, 'Cat')
#     dog_path = os.path.join(base_path, 'Dog')
#
#     # Initialize data and labels lists
#     data = []
#     labels = []
#
#     # Process cat images and assign label 0
#     for img_name in os.listdir(cat_path):
#         img_path = os.path.join(cat_path, img_name)
#         img = image.load_img(img_path, target_size=(224, 224))
#         img_array = image.img_to_array(img)
#         img_array = preprocess_input(img_array)
#         data.append(img_array)
#         labels.append(0)  # Cat label is 0
#
#     # Process dog images and assign label 1
#     for img_name in os.listdir(dog_path):
#         img_path = os.path.join(dog_path, img_name)
#         img = image.load_img(img_path, target_size=(224, 224))
#         img_array = image.img_to_array(img)
#         img_array = preprocess_input(img_array)
#         data.append(img_array)
#         labels.append(1)  # Dog label is 1
#
#     # Convert to numpy arrays
#     X = np.array(data)
#     y = np.array(labels)
#
#     # Shuffle the dataset
#     X, y = shuffle(X, y, random_state=seed)
#
#     # Use pre-trained VGG16 to extract features
#     vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#     X_features = vgg_model.predict(X)
#
#     # Reshape features for classifier input
#     X_features = X_features.reshape(X_features.shape[0], -1)
#
#     return X_features, y


def get_credit_card_ds(seed):
    # https://www.kaggle.com/datasets/rohitudageri/credit-card-details
    path1 = Util.get_dataset_path('06_CREDIT_CARD\\006_Credit_card.csv')
    path2 = Util.get_dataset_path('06_CREDIT_CARD\\006_Credit_card_label.csv')
    df = pd.read_csv(path1)
    df_label = pd.read_csv(path2)
    df = pd.merge(df, df_label, on='Ind_ID', how='outer')
    #Dealing with Null Values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    cols_to_impute_numeric = ['Annual_income', 'Birthday_count']
    df[cols_to_impute_numeric] = imputer.fit_transform(df[cols_to_impute_numeric])
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    cols_to_impute_string = ['GENDER', 'Type_Occupation']
    df[cols_to_impute_string] = imputer.fit_transform(df[cols_to_impute_string])
    #Drop Unnecessary Columns
    df = df.drop(columns=['Ind_ID', 'Mobile_phone'], axis=1)
    #Encoding Categorical Features
    cols_to_le = ['GENDER', 'Car_Owner', 'Propert_Owner']
    cols_to_ohe = ['Type_Income', 'EDUCATION', 'Marital_status', 'Housing_type', 'Type_Occupation']
    le = LabelEncoder()
    for i in cols_to_le:
        df[i] = le.fit_transform(df[i])

    df = pd.get_dummies(df, columns=cols_to_ohe, dtype=int)
    df = shuffle(df, random_state=seed)
    X = df.drop(columns=['label'], axis=1).values
    y = df['label'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def get_email_span_ds(seed):
    #https://www.kaggle.com/datasets/colormap/spambase
    path = Util.get_dataset_path('07_SPAM_BASE\\007_SPAM_BASE.csv')
    df = pd.read_csv(path)
    df = shuffle(df, random_state=seed)
    # Split into features (X) and target variable (y)
    X = df.drop('spam', axis=1).values
    y = df['spam'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


# def get_amazon_customer_review_ds():
#     path = Util.get_dataset_path('13_MNIST_Still\\013_MNIST_Still.csv')
#     df = pd.read_csv(path)
#     df['Helpful %'] = np.where(df['HelpfulnessDenominator'] > 0,
#                                df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'], -1)
#     df['Upvote %'] = pd.cut(df['Helpful %'], bins=[-1, 0, 0.2, 0.4, 0.6, 0.8, 1],
#                             labels=['Empty', '0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
#     df.groupby(['Score', 'Upvote %']).agg('count')
#     df_s = df.groupby(['Score', 'Upvote %']).agg({'Id': 'count'}).reset_index()
#     pivot = df_s.pivot(index='Upvote %', columns='Score')
#
#     # apply BOW
#     df['Score'].unique()
#     df2 = df[df['Score'] != 3]
#     X = df2['Text']
#     df2['Score'].unique()
#     y_dict = {1: 0, 2: 0, 4: 1, 5: 1}
#     y = df2['Score'].map(y_dict)
#     # convert text
#     c = CountVectorizer(stop_words='english')
#     X = c.fit_transform(X)
#    # use this (scaler = StandardScaler()
# #     X = scaler.fit_transform(X)) before return X,y
#     return X, y

import numpy as np
import pandas as pd


# def get_amazon_customer_review_ds():
#     path = Util.get_dataset_path('13_MNIST_Still\\013_MNIST_Still.csv')
#     df = pd.read_csv(path)
#
#     # Calculate Helpful %
#     df['Helpful %'] = np.where(df['HelpfulnessDenominator'] > 0,
#                                df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'], -1)
#
#     # Categorize Upvote %
#     df['Upvote %'] = pd.cut(df['Helpful %'], bins=[-1, 0, 0.2, 0.4, 0.6, 0.8, 1],
#                             labels=['Empty', '0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
#
#     # Grouping and counting without warning
#     df_s = df.groupby(['Score', 'Upvote %'], observed=True).size().reset_index(name='count')
#
#     # Create a pivot table
#     pivot = df_s.pivot(index='Upvote %', columns='Score', values='count')
#
#     # Filter and prepare data for model
#     df2 = df[df['Score'] != 3]
#     X = df2['Text']
#     y_dict = {1: 0, 2: 0, 4: 1, 5: 1}
#     y = df2['Score'].map(y_dict)
#
#     # Convert text to BOW (Bag of Words)
#     c = CountVectorizer(stop_words='english')
#     X = c.fit_transform(X)
#   # use this (scaler = StandardScaler()
#     X = scaler.fit_transform(X)) before return X,y
#     return X, y.values

def get_iris_ds():
    path = Util.get_dataset_path('13_MNIST_Still\\013_MNIST_Still.csv')
    df = pd.read_csv(path)
    # Define feature columns and label column
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    label_column = 'species'

    # Extract features and labels using column names
    X = df[feature_columns].values.tolist()  # Convert to list of lists

    # Convert labels to numeric values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_column]).tolist()  # Convert to list of integers
    return X, y


def stemming(content):
    portStemmer = PorterStemmer()
    content = str(content)
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [portStemmer.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


def get_fake_and_real_news_ds():
    # https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    nltk.download('stopwords')
    path_fake = Util.get_dataset_path('08_FAKE_REAL_NEWS\\FAKE.csv')
    path_true = Util.get_dataset_path('08_FAKE_REAL_NEWS\\TRUE.csv')


    # Load the data from the CSV files
    fake_data = pd.read_csv(path_fake)
    true_data = pd.read_csv(path_true)

    # Add the target labels
    fake_data["target"] = 0  # Assign label 0 for fake data
    true_data["target"] = 1  # Assign label 1 for true data

    # Combine the two datasets
    combined_data = pd.concat([fake_data, true_data], ignore_index=True)

    # Shuffle the combined data (optional but recommended)
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Display the combined DataFrame
    # print(combined_data.head())
    combined_data.drop(columns=['date'], axis=1, inplace=True)
    combined_data.drop(columns=['subject'], axis=1, inplace=True)
    # print(combined_data.head())

    portStemmer = PorterStemmer()
    # stemming the title column
    combined_data['title'] = combined_data['title'].apply(stemming)
    combined_data.drop(columns=('text'), inplace=True)

    X = combined_data['title'].values
    y = combined_data['target'].values

    vectorizer = TfidfVectorizer()
    vectorizer.fit(X)

    X = vectorizer.transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # lr = LogisticRegression()
    # lr.fit(X_train, y_train)
    # y_pred = lr.predict(X_test)
    # accuracy = accuracy_score(y_pred, y_test)
    # print(accuracy)
    return X, y

######################################## MultiClass Datasets##########################

# def get_credit_score_brackets_ds(seed):
#     #https://www.kaggle.com/datasets/sudhanshu2198/processed-data-credit-score
#     # Load dataset
#     path = Util.get_dataset_path('13_CREDIT_SCORE_BRACKETS\\013_CREDIT_SCORE_BRACKETS.csv')
#     df = pd.read_csv(path)
#     df = shuffle(df, random_state=seed)
#
#     # Split into X and y
#     X = df.drop("Credit_Score", axis=1)  # Keep this as a DataFrame for now
#     y = df["Credit_Score"].values  # Convert target to 1D array
#
#     # Define categorical and numeric columns
#     categorical = ['Payment_of_Min_Amount', 'Credit_Mix']
#     numeric = X.select_dtypes(exclude="object").columns
#
#     # Further categorize numeric columns into low and high cardinality
#     low_cardinality = [col for col in numeric if df[col].nunique() <= 30]
#     high_cardinality = [col for col in numeric if df[col].nunique() > 30]
#
#     # Label encode the target variable (Credit_Score)
#     label = LabelEncoder()
#     y = label.fit_transform(df["Credit_Score"])  # Convert target variable to numeric form
#
#     # Power transform high-cardinality numeric features
#     transformer = PowerTransformer()
#     numeric_transformed = transformer.fit_transform(df[high_cardinality])  # Scale numeric features
#
#     # One-hot encode categorical variables
#     encoding = OneHotEncoder(drop="first", sparse_output=False)  # Disable sparse output to return array
#     one_hot = encoding.fit_transform(df[categorical])  # One-hot encode categorical features
#
#     # Handle ordinal variables (low-cardinality numeric)
#     ordinal = df[low_cardinality].values  # No transformation, direct conversion to NumPy
#
#     # Concatenate all features (numeric, one-hot encoded, and ordinal)
#     X = np.concatenate([numeric_transformed, one_hot, ordinal], axis=1)
#
#     return X, y





def get_credit_score_brackets_ds(seed):
    #https://www.kaggle.com/datasets/sudhanshu2198/processed-data-credit-score
    # Load dataset
    path = Util.get_dataset_path('13_CREDIT_SCORE_BRACKETS\\013_CREDIT_SCORE_BRACKETS.csv')
    df = pd.read_csv(path)
    df = shuffle(df, random_state=seed)

    # Split into X and y
    X = df.drop("Credit_Score", axis=1)  # Keep this as a DataFrame for now
    y = df["Credit_Score"].values  # Convert target to 1D array

    # Define categorical and numeric columns
    categorical = ['Payment_of_Min_Amount', 'Credit_Mix']
    numeric = X.select_dtypes(exclude="object").columns

    # Further categorize numeric columns into low and high cardinality
    low_cardinality = [col for col in numeric if df[col].nunique() <= 30]
    high_cardinality = [col for col in numeric if df[col].nunique() > 30]

    # Label encode the target variable (Credit_Score)
    label = LabelEncoder()
    y = label.fit_transform(df["Credit_Score"])  # Convert target variable to numeric form

    # Power transform high-cardinality numeric features
    transformer = PowerTransformer()
    numeric_transformed = transformer.fit_transform(df[high_cardinality])  # Scale numeric features

    # One-hot encode categorical variables
    encoding = OneHotEncoder(drop="first", sparse_output=False)  # Disable sparse output to return array
    one_hot = encoding.fit_transform(df[categorical])  # One-hot encode categorical features

    # Handle ordinal variables (low-cardinality numeric)
    ordinal = df[low_cardinality].values  # No transformation, direct conversion to NumPy

    # Concatenate all features (numeric, one-hot encoded, and ordinal)
    X = np.concatenate([numeric_transformed, one_hot, ordinal], axis=1)

    return X, y

def get_human_activity_recognition_with_smartphones_ds(seed):
    #https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
    path_train = Util.get_dataset_path("15_HUMAN_ACTIVITY_RECOGNITION_WITH_SMARTPHONES\\015_HARWS_TRAIN.csv")
    path_test = Util.get_dataset_path("15_HUMAN_ACTIVITY_RECOGNITION_WITH_SMARTPHONES\\015_HARWS_TEST.csv")
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)
    # Combine train and test dataframes
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    df = shuffle(df_combined, random_state=seed)

    X = df.drop(columns=['Activity'], axis=1).values
    y = df['Activity'].values

    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    return X, y

from sklearn import datasets
def get_handwritten_digits_ds(seed):
    # https://www.kaggle.com/code/satishgunjal/multiclass-logistic-regression-using-sklearn
    digits_df = datasets.load_digits()
    X = digits_df.data
    y = digits_df.target
    return X, y

if __name__ == "__main__":
    seed = 42
    # DS05 call
    # path = Util.get_dataset_path('05_RAISIN\\005_raisin.csv')
    # get_raisin_ds(path, seed)

    # # DS06 call
    # get_credit_card_ds(seed)

    # # DS07 call
    # get_email_span_ds(seed)

    #DS8 call
    # X, y = get_amazon_customer_review_ds()
    # print(X)
    # # print(y)

    # # DS8 call
    # X,y = get_iris_ds()
    # print(X)
    # print(y)

    # DS9 call:
    # X, y = get_fake_and_real_news_ds()
    # print(X)
    # print(y)







