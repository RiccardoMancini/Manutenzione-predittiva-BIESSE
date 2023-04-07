import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import resample
from scipy.stats import weibull_min
import datetime as dt


class PredManClass:

    def __init__(self):
        self.vib_foot = pd.read_excel('VibrationFootprints.xlsx', sheet_name='Foglio1')

        # conversione in timestamp
        self.vib_foot['startDate'] = self.vib_foot['startDate'].astype('int64') // 10 ** 9
        self.vib_foot['endDate'] = self.vib_foot['endDate'].astype('int64') // 10 ** 9

        print(self.vib_foot.shape)

    def some_stats(self):
        v = self.vib_foot['classe'].value_counts()
        print('N° sample x classe:')
        print(v)

    def decisionTree_classifier(self):
        # rimuoviamo temporaneamente il campione della classe con un solo campione
        single_sample_class = None
        for c in self.vib_foot['classe'].unique():
            # print(c)
            if self.vib_foot[self.vib_foot['classe'] == c].shape[0] == 1:
                single_sample_class = self.vib_foot.loc[self.vib_foot['classe'] == c]
                # print(self.vib_foot.loc[self.vib_foot['classe'] == c])
                self.vib_foot = self.vib_foot.loc[self.vib_foot['classe'] != c]

        # print(self.vib_foot.shape[0])

        # Oversampling delle classi 3 e 4 e undersampling della classe 5
        df_2 = resample(self.vib_foot[self.vib_foot['classe'] == 3], replace=True, n_samples=19, random_state=42)
        df_1 = resample(self.vib_foot[self.vib_foot['classe'] == 4], replace=True, n_samples=19, random_state=42)
        df_3 = self.vib_foot[self.vib_foot['classe'] == 5].sample(19)


        # Combinare i dataframes oversampled con il dataframe originale
        df_balanced = pd.concat([df_3, df_2, df_1])

        # Verificare che le classi siano bilanciate
        print(df_balanced['classe'].value_counts())
        self.vib_foot = df_balanced.drop(['snMacchina', 'snEm'], axis=1)
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
        '''metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(y_test, prediction)).plot()
        plt.show()'''


        # Plotting the feature importance for Top 10 most important columns
        feature_importances = pd.Series(dTree.feature_importances_, index=X_train.columns)
        feature_importances.nlargest(10).plot(kind='barh', fontsize=6)
        plt.show()

    def weibullDist(self):
        df = self.vib_foot

        # Calcola i parametri della distribuzione di Weibull
        shape, loc, scale = weibull_min.fit(df['deltaDateHour'], floc=0)

        # Stima il tempo di vita rimanente per ogni componente
        df['remainingLifeTime'] = weibull_min.ppf(0.9, shape, loc=loc, scale=scale) - df['deltaDateHour']

        # Converti il tempo di vita rimanente in giorni
        df['remainingLifeTime'] = df['remainingLifeTime'] / 24

        # Stampa il risultato
        print(df[['snMacchina', 'remainingLifeTime']])


        # Calcola la distribuzione di Weibull con i parametri shape, loc e scale
        x = np.linspace(0, 10000, 10000)
        pdf = weibull_min.pdf(x, shape, loc=loc, scale=scale)

        # Grafica la distribuzione di Weibull
        plt.plot(x, pdf)
        plt.xlabel('Tempo di vita (ore)')
        plt.ylabel('Densità di probabilità')
        plt.title('Distribuzione di Weibull')
        plt.show()


if __name__ == "__main__":
    predManObj = PredManClass()
    # predManObj.some_stats()
    # predManObj.decisionTree_classifier()
    predManObj.weibullDist()
