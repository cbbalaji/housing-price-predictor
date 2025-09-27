import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


def load_data(path):
    df = pd.read_csv(path)
    return df


def new_features(df):
    # People per household
    df['population_per_household'] = df['population'] / df['households']
    # Rooms per household
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    # Bedrooms per room
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    '''Removing the features that were highly correlated with each other 
    and that was used to create the new features'''
    df.drop(['total_rooms', 'total_bedrooms', 'population', 'households'], 
            axis=1, inplace=True)
    return df


def loc_cluster(df):
    loc = df[['latitude', 'longitude']]
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['loc_cluster'] = kmeans.fit_predict(loc)
    # Keeping the clustered column and remove the latitude / longtitude
    df.drop(['longitude', 'latitude'], axis=1, inplace=True)
    return df


def preprocess(df):
    df = df.dropna()
    df = new_features(df)
    df = loc_cluster(df)
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    return train_test_split(X, y, test_size=0.2, random_state=42)