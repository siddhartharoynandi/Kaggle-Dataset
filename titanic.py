import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import f1_score

if __name__ == '__main__':
    #Load the data in a panda dataframe. Sequencial attributes are removed.
    train_data = pd.read_csv('/Users/siddhartharoynandi/Desktop/titanic/train.csv')[['Pclass','Sex','Age','SibSp','Parch','Survived']]

    # Preprocessing steps : First check which columns are eligible for null.
    # Take the Male and Female data seperately
    male_rows = train_data.loc[train_data['Sex'] == 'male']
    female_rows = train_data.loc[train_data['Sex'] == 'female']

    #For different genders replace the missing age values by average of that population
    male_rows['Age'].fillna((male_rows['Age'].mean()), inplace=True)
    female_rows['Age'].fillna((female_rows['Age'].mean()), inplace=True)

    #Concatenate the preprocessed data
    processed_train_data = pd.concat([male_rows,female_rows])
    #Find the pearson correlation coefficient to determine significant features
    print processed_train_data.corr()
    #Map the categorical values to numeric values
    sex_map = {'male': 1, 'female': 0}
    processed_train_data['Sex'] = processed_train_data["Sex"].map(sex_map)

    #Create Training and Testing sets
    training,testing = train_test_split(processed_train_data,test_size=0.3)

    training_x = training[['Pclass','Sex','Age','SibSp','Parch']]
    training_y = training['Survived']
    test_x = testing[['Pclass','Sex','Age','SibSp','Parch']]
    test_y = testing['Survived']

    #Prepare Decision Tree model
    model = tree.DecisionTreeClassifier()
    model = model.fit(training_x,training_y)
    pred_y = model.predict(test_x)

    #Output F1-Score of the classification
    print f1_score(test_y,pred_y)
    exit(0)

