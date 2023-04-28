import numpy as np
import pandas as pd
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import weibull_min
from lifelines import WeibullFitter, WeibullAFTFitter, KaplanMeierFitter, ExponentialFitter, LogNormalFitter, \
    LogLogisticFitter
from lifelines.utils import k_fold_cross_validation, median_survival_times

import datetime as dt


class PredManClass:

    def __init__(self):
        self.vib_foot = pd.read_excel('VibrationFootprints.xlsx', sheet_name='VibrationFootprints')

        # conversione in timestamp
        self.vib_foot['startDate'] = self.vib_foot['startDate'].astype('int64') // 10 ** 9
        self.vib_foot['endDate'] = self.vib_foot['endDate'].astype('int64') // 10 ** 9

        print(self.vib_foot.head())

    def some_stats(self):
        df = self.vib_foot
        v = self.vib_foot['classe'].value_counts()
        print('N° sample x classe:')
        print(v)

        col = 'Ore_lav_totali'
        # Istogramma
        sns.histplot(data=df, x=col, bins=50, kde=True, alpha=0.5)
        plt.xlabel('Valori')
        plt.ylabel('Frequenza')
        plt.title('Istogramma dei dati')
        plt.show()

        # Some visualization...
        kmf = KaplanMeierFitter()
        kmf.fit(durations=df[col])
        kmf.plot_survival_function()
        plt.show()
        kmf.plot_cumulative_density()
        plt.show()

        median_ = kmf.median_survival_time_
        median_confidence_interval_ = median_survival_times(kmf.confidence_interval_)
        print(median_)
        print(median_confidence_interval_)

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
        df = self.reduce_n_range()
        df = df.iloc[:, 6:8]
        df["fail"] = self.vib_foot['classe'].apply(lambda x: 0 if x == 5 else 1)

        print(df.head())

        # Standardizziamo le covariate
        '''scaler = StandardScaler()
        covariates_df_standardized = scaler.fit_transform(df[['deltaDateHour', 'deltaDateHour']])
        covariates_df = pd.DataFrame(covariates_df_standardized, columns=['deltaDateHour', 'percentualiLavorazione'])
        # Concateniamo i due DataFrame
        df = pd.concat([covariates_df, time_df], axis=1)'''




        '''T = df["oreLavorazione"]
        E = df['fail']
        plt.hist(T, bins=50)
        plt.show()

        kmf = KaplanMeierFitter()
        kmf.fit(durations=T, event_observed=E)
        kmf.plot_survival_function()
        plt.show()

        median_ = kmf.median_survival_time_
        median_confidence_interval_ = median_survival_times(kmf.confidence_interval_)
        print(median_)
        print(median_confidence_interval_)

        # Instantiate each fitter
        wb = WeibullFitter()
        ex = ExponentialFitter()
        log = LogNormalFitter()
        loglogis = LogLogisticFitter()
        # Fit to data
        for model in [wb, ex, log, loglogis]:
            model.fit(durations=T, event_observed=E)
            # Print AIC
            print("The AIC value for", model.__class__.__name__, "is", model.AIC_)'''


        # FIRST IMPLEMENTATION
        weibull_aft = WeibullAFTFitter()
        scores = k_fold_cross_validation(weibull_aft, df, 'oreLavorazione', event_col='fail', k=5,
                                         scoring_method="concordance_index")
        print(scores)

        '''weibull_aft.fit(df, duration_col='oreLavorazione', event_col='fail')
        #weibull_aft.print_summary(3)
        '''
        scale = np.exp(weibull_aft.params_['lambda_']['Intercept'])
        shape = np.exp(weibull_aft.params_['rho_']['Intercept'])
        print(shape, scale)

        print(weibull_aft.median_survival_time_)
        print(weibull_aft.mean_survival_time_)




        new_data = pd.DataFrame({
            'deltaDateHour': [825],
            'fail': [0]
        })

        predicted_expectation = weibull_aft.predict_expectation(new_data)
        print(predicted_expectation)


        # prendo la baseline, quindi un comportamento medio
        n = df.mean()
        n['oreLavorazione'] = 1000
        sf = weibull_aft.predict_survival_function(n)
        sf.plot()
        plt.title('Funzione di sopravvivenza stimata')
        plt.xlabel('Tempo (ore)')
        plt.ylabel('Probabilità di sopravvivenza')
        plt.show()

        # Converti l'indice in un array numpy e seleziona l'indice del valore più vicino a 1000 ore
        time_of_work = 1000
        time_idx = np.abs(sf.index.to_numpy() - time_of_work).argmin()
        # print(time_idx)
        # Seleziona la probabilità di sopravvivenza corrispondente all'indice trovato
        prob_sopravvivenza = sf.iloc[time_idx, 0]
        print(f"Probabilità di sopravvivenza: {prob_sopravvivenza:.2%}")

        '''
        # SECOND IMPLEMENTATION
        wf = WeibullFitter()
        wf.fit(df['oreLavorazione'])
        scale = wf.lambda_
        shape = wf.rho_
        print(shape, scale)

        # THIRD IMPLEMENTATION
        shape, _, scale = weibull_min.fit(df['oreLavorazione'], floc=0)
        print(shape, scale)'''

        # per confrontare l'andamento in funzione delle classi
        '''ax = plt.subplot(111)
        m = (df["classe"] == 5)
        kmf.fit(durations=df['oreLavorazione'][m], label="Class 5")
        kmf.plot_survival_function(ax=ax)
        kmf.fit(df['oreLavorazione'][~m], label="Others class")
        kmf.plot_survival_function(ax=ax, at_risk_counts=True)
        plt.title("Survival of components")
        plt.show()'''

        # Calcola la distribuzione di Weibull con i parametri shape, loc e scale
        '''x = np.linspace(0, 10000, 10000)
        pdf = weibull_min.pdf(x, shape, scale=scale)

        # Grafica la distribuzione di Weibull
        plt.plot(x, pdf)
        plt.xlabel('Tempo di vita (ore)')
        plt.ylabel('Densità di probabilità')
        plt.title('Distribuzione di Weibull')
        plt.show()'''

    def reduce_n_range(self):
        df = self.vib_foot
        c = 0
        for i in range(6):
            end = 19 if i != 5 else 20
            # print(df.columns[9:end])
            c = c + 10 if i != 0 else float(i) + 9.5
            df[f'[{float(i) if i == 0 else c - 10}-{c if i != 5 else c + 1})'] = df.iloc[:, 9:end].sum(axis=1)
            df = df.drop(df.columns[9:end], axis=1)

        # conversione secondi in ore
        df.iloc[:, 9:16] = df.iloc[:, 9:16].apply(lambda x: x / 3600, axis=0)
        return df

    def vibration_footprint_matrix(self):
        df = self.reduce_n_range().iloc[:, 9:16]

        vibration_ranges = df.columns.values.tolist()
        vibration_sums = df.sum(axis=0).round(3).to_numpy()
        print(df.sum(axis=0))

        # creazione grafico
        norm = plt.Normalize(vmin=vibration_sums.min(), vmax=vibration_sums.max())
        plt.imshow(vibration_sums[:, np.newaxis].T, cmap='Reds', norm=norm)
        plt.colorbar()
        plt.title('Ore trascorse in ogni range')
        plt.xticks(np.arange(len(vibration_ranges)), vibration_ranges, fontsize=8, rotation=45)
        plt.yticks([])
        plt.tight_layout()
        # aggiungo i numeri all'interno dei quadranti del vettore
        for i in range(len(vibration_ranges)):
            text = plt.text(i, 0, vibration_sums[i], ha="center", va="center", color="black", fontsize=10)

        plt.show()


if __name__ == "__main__":
    predManObj = PredManClass()

    predManObj.some_stats()

    # predManObj.decisionTree_classifier()

    # predManObj.weibullDist()

    # predManObj.vibration_footprint_matrix()
