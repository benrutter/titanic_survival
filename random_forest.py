from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd


def read_and_process_data():
    survival_df = pd.read_csv('train.csv')
    final_df = pd.read_csv('test.csv')

    categorical_cols = ['Sex', 'Embarked']
    linear_cols = ['Pclass', 'Age', 'Fare']

    for column in linear_cols:
        column_mean = survival_df[column].mean()
        survival_df[column] = survival_df[column].fillna(value=column_mean)
        final_df[column] = final_df[column].fillna(value=column_mean)

    X = pd.merge(
        pd.get_dummies(survival_df[categorical_cols]),
        survival_df[linear_cols],
        left_index=True,
        right_index=True,
    )
    X_final = pd.merge(
        pd.get_dummies(final_df[categorical_cols]),
        final_df[linear_cols],
        left_index=True,
        right_index=True,
    )

    y = survival_df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=7,
    )

    return X_train, X_test, y_train, y_test, X_final, final_df[['PassengerId']]


if __name__ == '__main__':

    X_train, X_test, y_train, y_test, X_final, output_df = read_and_process_data()

    parameters = {
        'n_estimators': [1, 5, 10, 15, 25, 50, 75, 100, 200],
        'max_depth': list(range(3, 10)),
        'random_state': [7],
    }

    grid_search = GridSearchCV(RandomForestClassifier(), parameters)
    grid_search.fit(X_train, y_train)

    model = RandomForestClassifier(**grid_search.best_params_)
    model.fit(X_train, y_train)

    output_df['Survived'] = model.predict(X_final)
    output_df.to_csv(
        'predictions.csv',
        index=False,
    )

