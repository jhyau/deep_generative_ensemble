from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def load_adult_census(as_frame=True, path='data/adult.csv'):
    try:
        df = pd.read_csv(path, encoding='latin-1')
    except:
        raise FileNotFoundError('Could not find adult.csv in data folder.')
    
    # Remove space in column names
    df.rename(columns=lambda x: x.strip(), inplace=True)
    
    # Show top few examples
    print(df.head())
    print("Columns in the data:")
    for col in df.columns:
        print(col)
        # Remove space in column values
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    print("after removing spaces:")
    print(df.head())

    df[df==' ?'] = np.nan
    for col in ['workclass', 'occupation', 'native.country']:
        df[col].fillna(df[col].mode()[0], inplace=True)
   
    print("original income column: \n", df['income'])
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1, ' <=50K': 0, ' >50K': 1, ' <=50K.': 0, ' >50K.': 1})
    print("after mapping: \n", df['income'])

    # drop education because it is already encoded in education.num
    X, y = df.drop(['income', 'education'], axis=1), df['income']
    categorical = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    
    for feature in categorical:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature])
    
    if as_frame:
        X = pd.DataFrame(X)
        y = pd.Series(y)
    print("input features: \n", X)
    print("income label should be boolean 0 or 1: \n", y)

    print("any nans: ", df['income'].isnull().values.any())
    print("how many nans: ", df['income'].isnull().sum())
    return X, y


if __name__ == '__main__':
    X, y = load_adult_census()
    print(X.head())
    print(y.head())
    print('Mean\n',X.mean(axis=0))
    print('Shape', X.shape)
