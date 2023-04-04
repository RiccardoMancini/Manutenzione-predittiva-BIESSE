import pandas as pd
import openpyxl
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from datetime import datetime

class PredManClass:
    def __init__(self):
        self.vib_foot = pd.read_excel('VibrationFootprints.xlsx', sheet_name='Foglio1')

        # conversione in timestamp
        self.vib_foot['startDate'] = self.vib_foot['startDate'].astype('int64') // 10**9
        self.vib_foot['endDate'] = self.vib_foot['endDate'].astype('int64') // 10 ** 9

    def some_stats(self):
        v = self.vib_foot['classe'].value_counts()
        print('N° sample x classe:')
        print(v)

    def decisionTree_classifier(self):
        # rimuoviamo temporaneamente il campione della classe con un solo campione
        single_sample_class = None
        for c in self.vib_foot['classe'].unique():
            print(c)
            if (self.vib_foot[self.vib_foot['classe'] == c]).sum() == 1:
                single_sample_class = self.vib_foot.loc[self.vib_foot['classe'] == c]
                self.vib_foot = self.vib_foot.loc[self.vib_foot['classe'] != c]
                break

        # Split the data into training and testing set        
        X_train, X_test, y_train, y_test = train_test_split(self.vib_foot.drop(['classe'], axis=1),
                                                            self.vib_foot['classe'],
                                                            stratify=self.vib_foot['classe'],
                                                            test_size=0.2,
                                                            random_state=42)

        # choose from different tunable hyperparameters
        clf = tree.DecisionTreeClassifier(max_depth=3, criterion='entropy')

        # Creating the model on Training Data
        dTree = clf.fit(X_train, y_train)
        prediction = dTree.predict(X_test)

        # Measuring accuracy on Testing Data
        print(metrics.classification_report(y_test, prediction))
        print(metrics.confusion_matrix(y_test, prediction))

        # Plotting the feature importance for Top 10 most important columns
        feature_importances = pd.Series(dTree.feature_importances_, index='Predictors')
        feature_importances.nlargest(10).plot(kind='barh')

        # Printing some sample values of prediction
        '''TestingDataResults = pd.DataFrame(data=X_test, columns=Predictors)
        TestingDataResults &  # 91;'TargetColumn']=y_test
        TestingDataResults &  # 91;'Prediction']=prediction
        TestingDataResults.head()'''

if __name__ == "__main__":
    predManObj = PredManClass()
    # predManObj.some_stats()
    predManObj.decisionTree_classifier()
