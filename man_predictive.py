import os

import numpy as np
import pandas as pd
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, ShuffleSplit, GridSearchCV
from sklearn import metrics
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from scipy.stats import weibull_min
from lifelines import WeibullFitter, WeibullAFTFitter, KaplanMeierFitter, ExponentialFitter, LogNormalFitter, \
    LogLogisticFitter
from lifelines.utils import k_fold_cross_validation, median_survival_times
from collections import Counter
import datetime as dt
import warnings

warnings.filterwarnings("ignore")


class PredManClass:

    def __init__(self):
        self.vib_foot = pd.read_excel('VibrationFootprints.xlsx', sheet_name='VibrationFootprints')
        self.pre_process()

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

    def some_stats(self):
        df = pd.read_excel('Dataset.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)
        train, test = df[df['classe'] == 3].drop(['classe'], axis=1), df[df['classe'] == 5].drop(['classe'], axis=1)

        col = 'total'
        # Istogramma
        sns.histplot(data=train, x=col, bins=50, kde=True, alpha=0.5)
        plt.xlabel('Valori')
        plt.ylabel('Frequenza')
        plt.title('Istogramma dei dati')
        plt.show()

        # Some visualization...
        kmf = KaplanMeierFitter()
        kmf.fit(durations=train[col])
        kmf.plot_survival_function()
        plt.show()
        kmf.plot_cumulative_density()
        plt.show()

        median_ = kmf.median_survival_time_
        median_confidence_interval_ = median_survival_times(kmf.confidence_interval_)
        print(median_)
        print(median_confidence_interval_)

    def feature_subset(self):
        df_filt = self.vib_foot[self.vib_foot['classe'] == 3]

        dfr = df_filt.drop(['classe'], axis=1)
        print(dfr.head(), dfr.shape)

        # compute pearson's
        target_correlation = dfr.corr()[['Ore_lav_totali']]
        target = target_correlation.drop('Ore_lav_totali')
        valfin = target[abs(target) > 0.5].dropna()
        # print(valfin, self.vib_foot.head())

        # val1 = valfin.sort_values('Ore_lav_totali',ascending=False)
        '''plt.figure(figsize=(7, 5))
        sns.heatmap(valfin, annot=True, cmap=plt.cm.Reds)
        plt.show()'''
        return valfin

    def overSample(self, n_samples):

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
            noise = np.clip(np.random.normal(0, stds * noise_ratio, size=(n_samples, len(data.columns))), a_min=0,
                            a_max=None)

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

        '''df = self.component_correlation()
        vibration_de = df[df['classe'] == 3].iloc[:, 1:].drop(['Ore_lav_totali'], axis=1)
        print('N° sample before oversampling: ', vibration_de.shape)'''

        df = pd.read_excel('reduce_dimensionNEW.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)
        df = df.drop(['classe'], axis=1)
        print('N° sample before oversampling: ', df.shape)
        df_res = oversample_with_gaussian_noise(df,n_samples)
        print('N° sample after oversampling: ', df_res.shape)
        df_res.to_excel('reduce_dimensionNEW.xlsx')

    def reduce_n_range(self):
        feature = self.feature_subset()
        df = self.overSample()

        print(feature.index.values.tolist())
        print(df.shape)

        col = [x for x in feature.index.values.tolist()]
        for x in ['total', 'deltaDateHour', 'classe']:
            col.append(x)
        dfN = df[df.columns.intersection(col)]
        print(dfN.shape, dfN.head())

        dfN.to_excel('Dataset.xlsx')

    def merge_columns(self, df):
        # df = self.overSample()

        dfN = df.drop(['total', 'deltaDateHour', 'classe', 'ore_lav_rim'], axis=1)
        # print(dfN.head())

        g1 = dfN.columns[6:11]
        # print(g1)

        red = dfN[g1].sum(axis=1)

        dfN, dfN['[5.5-10.5)'] = dfN.iloc[:, :6], red
        dfN['deltaDateHour'], dfN['classe'], dfN['total'], dfN['ore_lav_rim'] = \
            df['deltaDateHour'], df['classe'], df['total'], df['ore_lav_rim']

        dfN.to_excel('reduce_dimensionNEW.xlsx')

    def component_correlation(self):
        vibration_al = self.vib_foot[self.vib_foot['classe'] != 3].reset_index().drop(['index'], axis=1)
        vibration_de = self.vib_foot[self.vib_foot['classe'] == 3].reset_index().drop(['index'], axis=1)

        al = vibration_al.iloc[:, 3:]
        de = vibration_de.iloc[:, 3:]
        print(al.shape)
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
        # matpvalue.to_excel("dist.xlsx", sheet_name='Euclidean_dist')

        # prendo le righe con dist euclidea media minore di 0.5
        matpvalue = matpvalue[(matpvalue.mean(axis=1) < 0.5)]

        # filtro i componenti no-fail che ricadono in questa distanza
        vibration_al = vibration_al[vibration_al.index.isin(matpvalue.index)]

        # li ri-concateno con quelli fail iniziali
        result = pd.concat([vibration_al, vibration_de])
        print(result.head(), result.shape)
        return result

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

    def discretize_columns(self, dataframe, columns, bin_width=24):
        # Calcola il minimo e il massimo globali tra le colonne specificate
        global_min = dataframe[columns].min().min()
        global_max = dataframe[columns].max().max()

        for column in columns:
            # Calcola gli intervalli per la discretizzazione
            bins = np.arange(global_min, global_max + bin_width, bin_width)
            # print(bins)
            # Discretizza la colonna
            dataframe[column] = pd.cut(dataframe[column], bins=bins, labels=False, include_lowest=True)

        return dataframe

    def get_new_data(self):
        glob_df = None
        for file in os.listdir('VibrationFootprints_time'):
            df = pd.read_csv(f'./VibrationFootprints_time/{file}')
            df.rename(columns={'delta_date_hour': 'deltaDateHour', 'tempo_lav': 'total'}, inplace=True)
            df.drop(df.tail(1).index, inplace=True)
            if glob_df is not None:
                glob_df = pd.concat([glob_df, df])
            else:
                glob_df = df
            #print(glob_df.shape)

        glob_df.iloc[:, 3:-1] = glob_df.iloc[:, 3:-1].apply(lambda x: x / 3600, axis=0)
        glob_df['classe'] = 3
        self.merge_columns(glob_df)

    def weibullDist2_0(self):
        self.get_new_data()
        self.overSample(52)

        df = pd.read_excel('reduce_dimensionNEW.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(df.drop(['ore_lav_rim'], axis=1),
                                                            df['ore_lav_rim'],
                                                            test_size=0.2,
                                                            random_state=42)
        X_train['ore_lav_rim'] = y_train
        weibull_aft = WeibullAFTFitter()
        weibull_aft.fit(X_train, duration_col='ore_lav_rim')
        # weibull_aft.print_summary()

        scale = np.exp(weibull_aft.params_['lambda_']['Intercept'])
        shape = np.exp(weibull_aft.params_['rho_']['Intercept'])
        print(shape, scale)

        print(weibull_aft.median_survival_time_)
        print(weibull_aft.mean_survival_time_)
        # print(weibull_aft.confidence_intervals_)

        predictions = []
        for i, t in X_test.iterrows():
            #print(i, t)
            predict = weibull_aft.predict_expectation(X_test.loc[[i]])
            predictions.append(predict.item())

        '''sf = weibull_aft.predict_survival_function(X_test)
        sf.plot()
        plt.title('Funzione di sopravvivenza stimata')
        plt.xlabel('Tempo (ore)')
        plt.ylabel('Probabilità di sopravvivenza')
        plt.show()'''

        data = {'ore_lav_rimanenti predette': predictions,
                'ore_lav_rimanenti reali': y_test.tolist()}
        confronto = pd.DataFrame(data)
        confronto.to_excel('comparationWeibullDistNEW.xlsx')
        # print(confronto.head())

    def SVM2_0(self):
        df = pd.read_excel('reduce_dimensionNEW.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)
        target = df['ore_lav_rim']
        df = df.drop(['ore_lav_rim'], axis=1)

        # normalizzare i dataframe
        df = (df - df.min()) / (df.max() - df.min())
        df['ore_lav_rim'] = target
        X_train, X_test, y_train, y_test = train_test_split(df.drop(['ore_lav_rim'], axis=1),
                                                            df['ore_lav_rim'],
                                                            test_size=0.2,
                                                            random_state=42)

        # modello generale
        param = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'C': [1, 5, 10], 'degree': [3, 8],
                 'coef0': [0.01, 10, 0.5], 'gamma': ('auto', 'scale')}

        model = SVR()
        svr_cv_modelgen = GridSearchCV(model, param, cv=5, verbose=10)
        svr_tuned_gen = svr_cv_modelgen.fit(X_train, y_train)

        print(svr_tuned_gen.best_params_)
        # {'C': 10, 'coef0': 10, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}

        y_pred_gen = list(svr_tuned_gen.predict(X_test))

        data = {'ore_lav_rimanenti predette': y_pred_gen,
                'ore_lav_rimanenti reali': y_test.tolist()}
        confronto = pd.DataFrame(data)
        confronto.to_excel('comparationSvmNEW.xlsx')


    def weibullDistNEW(self):
        # load data
        df = pd.read_excel('reduce_dimensionNEW.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)
        target = df['ore_lav_rim']
        df = df.drop(['ore_lav_rim'], axis=1)

        print(df.head())

        # normalizzare i dataframe
        df = (df - df.min()) / (df.max() - df.min())
        df['ore_lav_rim'] = target
        train, test = df[df['classe'] == 0].drop(['classe'], axis=1), df[df['classe'] == 1].drop(['classe'], axis=1)

        x_test, y_test = test.drop(['ore_lav_rim'], axis=1), test['ore_lav_rim']
        # print(train.head(), test.head())

        # FIRST IMPLEMENTATION
        weibull_aft = WeibullAFTFitter()
        weibull_aft.fit(train, duration_col='ore_lav_rim')
        # weibull_aft.print_summary()

        scale = np.exp(weibull_aft.params_['lambda_']['Intercept'])
        shape = np.exp(weibull_aft.params_['rho_']['Intercept'])
        print(shape, scale)

        print(weibull_aft.median_survival_time_)
        print(weibull_aft.mean_survival_time_)
        # print(weibull_aft.confidence_intervals_)

        predictions = []
        probability = []
        for i, t in x_test.iterrows():
            # print(i, t)
            predict = weibull_aft.predict_expectation(x_test.iloc[[i]])
            predictions.append(predict.item())

            '''sf = weibull_aft.predict_survival_function(x_test.iloc[[i]])
            time_of_work = y_test.iloc[[i]].item()
            # TODO: il valore assoluto è probabilmente il motivo per cui c'è una prob. non coerente con la previsione
            time_idx = np.abs(sf.index.to_numpy() - time_of_work).argmin()
            print(time_idx)
            prob_sopravvivenza = sf.iloc[time_idx, 0]
            probability.append(prob_sopravvivenza)'''
        print(predictions)
        '''sf = weibull_aft.predict_survival_function(test.iloc[[11]])
        sf.plot()
        plt.title('Funzione di sopravvivenza stimata')
        plt.xlabel('Tempo (ore)')
        plt.ylabel('Probabilità di sopravvivenza')
        plt.show()'''

        app = pd.read_excel('reduce_dimensionNEW.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)
        data = {'Predizione ore_lav_rimanenti': predictions,
                'Ore lavoro svolte': app[app['classe'] == 5]['total'].tolist()}
        confronto = pd.DataFrame(data)
        confronto.to_excel('comparationWeibullDistNEW.xlsx')
        print(confronto.head())

        '''new_data = test.iloc[[x]].drop(['total'], axis=1)
        print(new_data)

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
        plt.show()'''

        # SECOND IMPLEMENTATION
        '''wf = WeibullFitter()
        wf.fit(train['total'])
        scale = wf.lambda_
        shape = wf.rho_
        print(shape, scale)
        wf.survival_function_.plot()
        plt.show()'''

        # THIRD IMPLEMENTATION
        '''shape, _, scale = weibull_min.fit(df['total'], floc=0)
        print(shape, scale)'''

        # Calcola la distribuzione di Weibull con i parametri shape, loc e scale
        '''x = np.linspace(0, 200, 200)
        pdf = weibull_min.pdf(x, shape, scale=scale)

        # Grafica la distribuzione di Weibull
        plt.plot(x, pdf)
        plt.xlabel('Tempo di vita (ore)')
        plt.ylabel('Densità di probabilità')
        plt.title('Distribuzione di Weibull')
        plt.show()'''

    def weibullDist(self):
        # load data
        df = pd.read_excel('reduce_dimension.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)
        target = df['total']
        df = df.drop(['total'], axis=1)

        print(df.head())

        # normalizzare i dataframe
        df = (df - df.min()) / (df.max() - df.min())
        df['total'] = target
        train, test = df[df['classe'] == 0].drop(['classe'], axis=1), df[df['classe'] == 1].drop(['classe'], axis=1)

        x_test, y_test = test.drop(['total'], axis=1), test['total']

        # FIRST IMPLEMENTATION
        weibull_aft = WeibullAFTFitter()
        weibull_aft.fit(train, duration_col='total')
        # weibull_aft.print_summary()

        scale = np.exp(weibull_aft.params_['lambda_']['Intercept'])
        shape = np.exp(weibull_aft.params_['rho_']['Intercept'])
        print(shape, scale)

        print(weibull_aft.median_survival_time_)
        print(weibull_aft.mean_survival_time_)
        # print(weibull_aft.confidence_intervals_)

        predictions = []
        probability = []
        for i, t in x_test.iterrows():
            print(i, t)
            predict = weibull_aft.predict_expectation(x_test.iloc[[i]])
            predictions.append(predict.item())

            sf = weibull_aft.predict_survival_function(x_test.iloc[[i]])
            time_of_work = y_test.iloc[[i]].item()
            # TODO: il valore assoluto è probabilmente il motivo per cui c'è una prob. non coerente con la previsione
            time_idx = np.abs(sf.index.to_numpy() - time_of_work).argmin()
            print(time_idx)
            prob_sopravvivenza = sf.iloc[time_idx, 0]
            probability.append(prob_sopravvivenza)

        '''sf = weibull_aft.predict_survival_function(test.iloc[[11]])
        sf.plot()
        plt.title('Funzione di sopravvivenza stimata')
        plt.xlabel('Tempo (ore)')
        plt.ylabel('Probabilità di sopravvivenza')
        plt.show()'''

        data = {'Predizione': predictions, 'Reale': y_test.tolist(), 'Prob. sopravvivenza': probability}
        confronto = pd.DataFrame(data)
        # confronto.to_excel('comparationWeibullDist.xlsx')
        print(confronto.head())

        '''new_data = test.iloc[[x]].drop(['total'], axis=1)
        print(new_data)

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
        plt.show()'''

        # SECOND IMPLEMENTATION
        '''wf = WeibullFitter()
        wf.fit(train['total'])
        scale = wf.lambda_
        shape = wf.rho_
        print(shape, scale)
        wf.survival_function_.plot()
        plt.show()'''

        # THIRD IMPLEMENTATION
        '''shape, _, scale = weibull_min.fit(df['total'], floc=0)
        print(shape, scale)'''

        # Calcola la distribuzione di Weibull con i parametri shape, loc e scale
        '''x = np.linspace(0, 200, 200)
        pdf = weibull_min.pdf(x, shape, scale=scale)

        # Grafica la distribuzione di Weibull
        plt.plot(x, pdf)
        plt.xlabel('Tempo di vita (ore)')
        plt.ylabel('Densità di probabilità')
        plt.title('Distribuzione di Weibull')
        plt.show()'''

    def SVM(self):
        # load data
        dfT = pd.read_excel('reduce_dimension.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)
        target = dfT['total']
        dfT = dfT.drop(['total'], axis=1)

        print(dfT.head())

        bin_widths = [12, 24, 48]
        for bin in bin_widths:
            app = dfT.iloc[:, :7]

            print(app.head())

            df = self.discretize_columns(app, app.columns, bin)
            df['deltaDateHour'], df['classe'] = dfT['deltaDateHour'], dfT['classe']
            df.to_excel('./grid_results/discr' + str(bin) + '.xlsx')

            # normalizzare i dataframe
            df = (df - df.min()) / (df.max() - df.min())
            df['total'] = target
            train, test = df[df['classe'] == 0].drop(['classe'], axis=1), df[df['classe'] == 1].drop(['classe'], axis=1)

            x_train, y_train = train.drop(['total'], axis=1), train['total']
            x_test, y_test = test.drop(['total'], axis=1), test['total']

            param = {'kernel': ('linear', 'poly'), 'C': [10, 5], 'degree': [3, 8],
                     'coef0': [0.01, 10, 0.5], 'gamma': ('auto', 'scale')}

            model = SVR()
            gridModel = GridSearchCV(model, param, cv=5)

            tunedModel = gridModel.fit(x_train, y_train)

            # Previsioni sui dati di test per ogni modello generato
            residuals = []
            for i in range(len(gridModel.cv_results_['params'])):
                model = SVR(**gridModel.cv_results_['params'][i])
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                residuals.append(y_test - y_pred)

            # Calcolo del valore assoluto dei residui per ogni modello
            absolute_errors = [np.mean(np.abs(res)) for res in residuals]

            # Individuazione degli indici dei modelli con i residui più grandi
            indices_of_outliers = np.argsort(absolute_errors)[-2:]
            # print(indices_of_outliers)

            # Stampa dei modelli con i residui più grandi
            for i, index in enumerate(indices_of_outliers):
                print("Modello con residuo elevato:")
                params = gridModel.cv_results_['params'][index]
                print("Parametri: ", params)
                print("Miglior score (R2): ", gridModel.cv_results_['mean_test_score'][index])
                # print("Residuo: ", residuals[index])
                print("---------------------------------")

                model = SVR(**params)
                model.fit(x_train, y_train)
                y_pred_gen = list(model.predict(x_test))

                data = {'Predizione': y_pred_gen, 'Reale': y_test}
                confronto = pd.DataFrame(data)
                confronto.to_excel('./grid_results/comparationSVM_discr' + str(bin) + '_' + str(i) + '.xlsx')


if __name__ == "__main__":
    predManObj = PredManClass()

    # predManObj.merge_columns()

    # predManObj.some_stats()

    # predManObj.feature_subset()

    # predManObj.overSample()

    # predManObj.component_correlation()

    # predManObj.decisionTree_classifier()

    # predManObj.weibullDist()

    # predManObj.get_new_data()
    # predManObj.weibullDistNEW()

    #predManObj.weibullDist2_0()
    predManObj.SVM2_0()

    # predManObj.SVM()

    # predManObj.reduce_n_range()
