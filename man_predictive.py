import numpy as np
import pandas as pd
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import weibull_min
from lifelines import WeibullFitter, WeibullAFTFitter, KaplanMeierFitter, ExponentialFitter, LogNormalFitter, \
    LogLogisticFitter
from lifelines.utils import k_fold_cross_validation, median_survival_times
from collections import Counter
import datetime as dt


class PredManClass:

    def __init__(self):
        self.vib_foot = pd.read_excel('VibrationFootprints.xlsx', sheet_name='VibrationFootprints')
        self.pre_process()

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

    def pre_process(self):
        # conversione in timestamp
        self.vib_foot['startDate'] = self.vib_foot['startDate'].astype('int64') // 10 ** 9
        self.vib_foot['endDate'] = self.vib_foot['endDate'].astype('int64') // 10 ** 9

        # converti secondi in ore negli intervalli delle vibrazioni
        self.vib_foot.iloc[:, 13:] = self.vib_foot.iloc[:, 13:].apply(lambda x: x / 3600, axis=0)

        # aggiungi le ore mancanti
        self.vib_foot['Ore_lav_totali'] = self.vib_foot['Ore_lav_totali'] + self.vib_foot['Ore lav manc']

        # rimuovi colonne inutili
        self.vib_foot = self.vib_foot.drop(['snMacchina', 'snEm', 'tp', 'startDate', 'endDate',
                                            'Perc ore lav', 'Lav mancanti', 'Perc lav manc',
                                            'Ore lav manc', 'Perc ore manc'], axis=1)

        # print(self.vib_foot.shape, self.vib_foot.head())

    def overSample(self):

        def oversample_with_gaussian_noise(data, n_samples, noise_ratio=0.3):
            """
            Oversampling con gaussian noise.

            Parameters
            ----------
            data : pandas.DataFrame
                Il dataframe contenente i dati originali.
            n_samples : int
                Il numero di campioni sintetici da generare.
            noise_ratio : float, optional (default=0.1)
                Il rapporto tra la deviazione standard del rumore e quella della feature.

            Returns
            -------
            pandas.DataFrame
                Il dataframe contenente i dati originali e i campioni sintetici generati.
            """

            # Creazione del dataframe vuoto per i dati sintetici
            synthetic_data = pd.DataFrame(columns=data.columns)

            # Calcolo delle deviazioni standard per le feature
            stds = data.std()

            # Generazione di rumore gaussiano con deviazioni standard proporzionali
            noise = np.clip(np.random.normal(0, stds*noise_ratio, size=(n_samples, len(data.columns))), a_min=0, a_max=None)

            # Ripetizione n_samples volte
            for i in range(n_samples):
                # Estrazione random di un sample dal dataframe dei dati originali
                sample = data.sample(n=1, replace=True)

                # Aggiunta del rumore gaussiano al sample estratto
                synthetic_sample = sample.values + noise[i, :]

                # Creazione del dataframe per il campione sintetico
                synthetic_sample_df = pd.DataFrame(synthetic_sample, columns=data.columns)

                # Aggiunta del campione sintetico al dataframe dei dati sintetici
                synthetic_data = pd.concat([synthetic_data, synthetic_sample_df], axis=0)

            # Unione dei dati originali e sintetici
            oversampled_data = pd.concat([data, synthetic_data], axis=0)

            return oversampled_data

        vibration_de = self.vib_foot[self.vib_foot['classe'] == 3].iloc[:, 1:]\
            .drop(['Ore_lav_totali'], axis=1)
        print('N° sample before oversampling: ', vibration_de.shape)

        df_res = oversample_with_gaussian_noise(vibration_de, 20)
        df_res['total'] = df_res[list(df_res.columns[1:])].sum(axis=1)
        print('N° sample after oversampling: ', df_res.shape)

        df_res.to_excel('overS.xlsx')

        return df_res

    def component_correlation(self):
        vibration_al = self.vib_foot[self.vib_foot['classe'] != 3].reset_index().drop(['index'],axis=1)
        vibration_de = self.vib_foot[self.vib_foot['classe'] == 3].reset_index().drop(['index'],axis=1)

        al = vibration_al.iloc[:, 3:]
        de = vibration_de.iloc[:, 3:]

        print(vibration_de.head(), de.head())
        #print(vibration_de.shape, vibration_al.shape)

        corMat = []
        c = []
        for i in range(0, de.shape[0]):
            for j in range(0, al.shape[0]):
                corMat.append(np.linalg.norm(al.iloc[j] - de.iloc[i]))
            c.append(corMat)
            corMat = []

        matpvalue = pd.DataFrame(c)
        matpvalue = (matpvalue - matpvalue.min()) / (matpvalue.max() - matpvalue.min())
        matpvalue = matpvalue.transpose()

        matpvalue.to_excel("dist.xlsx", sheet_name='Euclidean_dist')
        print(matpvalue.shape)

        matpvalue = matpvalue.drop(matpvalue[(matpvalue.mean() > 0.51) & (matpvalue.median() > 0.57)])
        print(matpvalue.shape)
        # vibration_al = vibration_al[vibration_al.index.isin(matpvalue.index)]
        # concatenazione vibration_al e de

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

        '''df = self.vib_foot[self.vib_foot['classe'] == 3].iloc[:, 1:8]
        total = df['Ore_lav_totali']
        df = df.drop(['Ore_lav_totali'], axis=1)
        #df = (df - df.min()) / (df.max() - df.min())
        df['total'] = total'''

        # fail data
        df = self.overSample().reset_index(drop=True)
        total_fail = df['total']
        df = df.iloc[:, :10]

        # no-fail data
        test_data = self.vib_foot[self.vib_foot['classe'] != 3].iloc[:, 1:12]
        total_no_fail = test_data['Ore_lav_totali']
        test_data = test_data.drop(['Ore_lav_totali'], axis=1)
        test_data = test_data.iloc[:, :10]

        # calcolare il minimo e il massimo dei due dataframe uniti
        min_val = pd.concat([df, test_data], axis=0).min()
        max_val = pd.concat([df, test_data], axis=0).max()

        # normalizzare i dataframe utilizzando il minimo e il massimo dei due dataframe uniti
        df = (df - min_val) / (max_val - min_val)
        test_data = (test_data - min_val) / (max_val - min_val)

        test_data['total'] = total_no_fail
        df['total'] = total_fail

        print(df.head())
        print(test_data.head())

        '''
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
        weibull_aft.fit(df, duration_col='total')
        # weibull_aft.print_summary(3)

        scale = np.exp(weibull_aft.params_['lambda_']['Intercept'])
        shape = np.exp(weibull_aft.params_['rho_']['Intercept'])
        print(shape, scale)

        print(weibull_aft.median_survival_time_)
        print(weibull_aft.mean_survival_time_)




        new_data = test_data.iloc[[1]].drop(['total'], axis=1)

        predicted_expectation = weibull_aft.predict_expectation(new_data)
        print(predicted_expectation)

        # prendo la baseline, quindi un comportamento medio
        n = df.mean()
        #n['Ore_lav_totali'] = 1000
        sf = weibull_aft.predict_survival_function(new_data)
        sf.plot()
        plt.title('Funzione di sopravvivenza stimata')
        plt.xlabel('Tempo (ore)')
        plt.ylabel('Probabilità di sopravvivenza')
        plt.show()

        hz = weibull_aft.predict_hazard(new_data)
        hz.plot()
        plt.title('Funzione di rischio stimata')
        plt.xlabel('Tempo (ore)')
        plt.ylabel('Funzione di hazard')
        plt.show()



        # Converti l'indice in un array numpy e seleziona l'indice del valore più vicino a 1000 ore
        time_of_work = 455
        time_idx = np.abs(sf.index.to_numpy() - time_of_work).argmin()
        # print(time_idx)
        # Seleziona la probabilità di sopravvivenza corrispondente all'indice trovato
        prob_sopravvivenza = sf.iloc[time_idx, 0]
        print(f"Probabilità di sopravvivenza: {prob_sopravvivenza:.2%}")


        # SECOND IMPLEMENTATION
        '''wf = WeibullFitter()
        wf.fit(df['total'])
        scale = wf.lambda_
        shape = wf.rho_
        print(shape, scale)'''



        # THIRD IMPLEMENTATION
        '''shape, _, scale = weibull_min.fit(df['total'], floc=0)
        print(shape, scale)'''

        # Calcola la distribuzione di Weibull con i parametri shape, loc e scale
        x = np.linspace(0, 200, 200)
        pdf = weibull_min.pdf(x, shape, scale=scale)

        # Grafica la distribuzione di Weibull
        plt.plot(x, pdf)
        plt.xlabel('Tempo di vita (ore)')
        plt.ylabel('Densità di probabilità')
        plt.title('Distribuzione di Weibull')
        plt.show()

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

    # predManObj.some_stats()

    # predManObj.overSample()

    predManObj.component_correlation()

    # predManObj.decisionTree_classifier()

    # predManObj.weibullDist()

    # predManObj.vibration_footprint_matrix()
