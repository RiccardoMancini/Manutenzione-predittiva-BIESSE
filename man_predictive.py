import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from scipy.stats import weibull_min
from lifelines import WeibullAFTFitter
import warnings

warnings.filterwarnings("ignore")


class PredManClass:

    def __init__(self):
        self.vib_foot = pd.read_excel('VibrationFootprintsD1.xlsx', sheet_name='VibrationFootprints')
        self.pre_process()

    def pre_process(self):
        # converti secondi in ore negli intervalli delle vibrazioni
        self.vib_foot.iloc[:, 13:] = self.vib_foot.iloc[:, 13:].apply(lambda x: x / 3600, axis=0)

        # aggiungi le ore mancanti
        self.vib_foot['Ore_lav_totali'] = self.vib_foot['Ore_lav_totali'] + self.vib_foot['Ore lav manc']

        # rimuovi colonne inutili
        self.vib_foot = self.vib_foot.drop(['snMacchina', 'snEm', 'tp', 'startDate', 'endDate',
                                            'Perc ore lav', 'Lav mancanti', 'Perc lav manc',
                                            'Ore lav manc', 'Perc ore manc'], axis=1)

    @staticmethod
    def some_stats():
        df = pd.read_excel('reduce_dimension.xlsx', sheet_name='Sheet1')
        # train, test = df[df['classe'] == 3].drop(['classe'], axis=1), df[df['classe'] == 5].drop(['classe'], axis=1)

        col = 'total'
        # Istogramma
        sns.histplot(data=df, x=col, bins=50, kde=True, alpha=0.5)
        plt.xlabel('Valori')
        plt.ylabel('Frequenza')
        plt.title('Istogramma dei dati')
        plt.show()

    def feature_subset_Pearson(self):
        df_filt = self.vib_foot[self.vib_foot['classe'] == 3]

        dfr = df_filt.drop(['classe'], axis=1)
        print(dfr.head(), dfr.shape)

        # compute pearson's correlation
        target_correlation = dfr.corr()[['Ore_lav_totali']]
        target = target_correlation.drop('Ore_lav_totali')
        valfin = target[abs(target) > 0.5].dropna()

        return valfin

    @staticmethod
    def overSample(n_samples):

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

        df = pd.read_excel('dataset2.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)
        df = df.drop(['classe'], axis=1)
        print('N° sample before oversampling: ', df.shape)
        df_res = oversample_with_gaussian_noise(df, n_samples)
        print('N° sample after oversampling: ', df_res.shape)
        df_res.to_excel('dataset2.xlsx')

    @staticmethod
    def plot_synt_data():
        df = pd.read_excel('Dataset1.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)
        df = df[df['classe'] == 3]
        # print(df.head())
        originale = df.iloc[:5]

        rumore = df.iloc[6:]

        # ndf = pd.DataFrame(rumore)
        porig = originale['[0.5-1.5)']
        xo = originale['[3.5-4.5)']

        prum = rumore['[0.5-1.5)']
        xr = rumore['[3.5-4.5)']

        plt.scatter(xo, list(porig), c="red")

        plt.scatter(xr, list(prum), c="green", marker="^")
        # print(porig)
        plt.xlabel("Intervallo vibrazione [3.5-4.5)")
        plt.ylabel("Intervallo vibrazione [0.5-1.5)")
        legend = plt.legend(['Campioni reali', 'Campioni sintetici'])
        legend._legend_box.sep = 20
        plt.show()

    def reduce_n_range(self):
        feature = self.feature_subset_Pearson()
        df = self.overSample()

        print(feature.index.values.tolist())
        print(df.shape)

        col = [x for x in feature.index.values.tolist()]
        for x in ['total', 'deltaDateHour', 'classe']:
            col.append(x)
        dfN = df[df.columns.intersection(col)]
        print(dfN.shape, dfN.head())

        dfN.to_excel('Dataset.xlsx')

    @staticmethod
    def merge_columns(df):
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

    @staticmethod
    def discretize_columns(dataframe, columns, bin_width=24):
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
        for file in os.listdir('VibrationFootprintsD2'):
            df = pd.read_csv(f'./VibrationFootprintsD2/{file}')
            df.rename(columns={'delta_date_hour': 'deltaDateHour', 'tempo_lav': 'total'}, inplace=True)
            df.drop(df.tail(1).index, inplace=True)
            if glob_df is not None:
                glob_df = pd.concat([glob_df, df])
            else:
                glob_df = df
            # print(glob_df.shape)

        glob_df.iloc[:, 3:-1] = glob_df.iloc[:, 3:-1].apply(lambda x: x / 3600, axis=0)
        glob_df['classe'] = 3
        self.merge_columns(glob_df)

    @staticmethod
    def weibullLeaveOneOut():
        target_col = 'ore_lav_rim'
        df = pd.read_excel('Dataset2.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)
        X = df
        y = df[target_col]
        n_samples = df.shape[0]
        mse_errors = []
        mae_errors = []
        sse_errors = []

        for i in tqdm(range(n_samples)):
            X_train = X.drop([i])
            X_test = X.loc[[i]].drop([target_col], axis=1)
            y_test = y.loc[[i]]
            '''print(X_train.head())
            print(X_test.head())
            print(y_test.head())'''

            weibull_aft = WeibullAFTFitter()
            weibull_aft.fit(X_train, duration_col='ore_lav_rim')
            y_pred = weibull_aft.predict_expectation(X_test)
            # print(y_pred)

            mse = (y_test - y_pred) ** 2
            rmse = np.sqrt(mse)
            mae = abs(y_test - y_pred)
            sse_sing = (y_test - y_pred) ** 2

            # print(mse, rmse, mae, sse_sing)
            mse_errors.append(mse)
            mae_errors.append(mae)
            sse_errors.append(sse_sing)

        avg_mse = np.mean(mse_errors)
        avg_rmse = np.sqrt(avg_mse)
        avg_mae = np.mean(mae_errors)
        sse = np.sum(sse_errors)
        print('WeibullDist')
        print(f'MSE: {round(avg_mse, 2)}; RMSE: {round(avg_rmse, 2)}; MAE: {round(avg_mae, 2)}; SSE: {round(sse, 2)};')

    @staticmethod
    def svmLeaveOneOut():
        target_col = 'ore_lav_rim'
        df = pd.read_excel('Dataset2.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)
        target = df['ore_lav_rim']
        df = df.drop(['ore_lav_rim'], axis=1)

        # normalizzare i dataframe
        df = (df - df.min()) / (df.max() - df.min())
        df['ore_lav_rim'] = target

        X = df
        y = df[target_col]
        n_samples = df.shape[0]
        mse_errors = []
        mae_errors = []
        sse_errors = []

        for i in tqdm(range(n_samples)):
            X_train = X.drop([i]).drop([target_col], axis=1)
            y_train = y.drop([i])
            X_test = X.loc[[i]].drop([target_col], axis=1)
            y_test = y.loc[[i]]
            '''print(X_train.head())
            print(y_train.head())
            print(X_test.head())
            print(y_test.head())'''

            model = SVR(C=10, coef0=10, degree=3, gamma='scale', kernel='poly')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # print(y_pred)

            mse = (y_test - y_pred) ** 2
            mae = abs(y_test - y_pred)
            sse_sing = (y_test - y_pred) ** 2

            # print(mse, rmse, mae, sse_sing)
            mse_errors.append(mse)
            mae_errors.append(mae)
            sse_errors.append(sse_sing)

        avg_mse = np.mean(mse_errors)
        avg_rmse = np.sqrt(avg_mse)
        avg_mae = np.mean(mae_errors)
        sse = np.sum(sse_errors)
        print('SVR')
        print(f'MSE: {round(avg_mse, 2)}; RMSE: {round(avg_rmse, 2)}; MAE: {round(avg_mae, 2)}; SSE: {round(sse, 2)};')

    @staticmethod
    def weibullDist():
        # load data
        df = pd.read_excel('Dataset1.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)
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
        dfT = pd.read_excel('Dataset1.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)
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

    @staticmethod
    def weibullDist2_0():
        # self.get_new_data()
        # self.overSample(52)

        df = pd.read_excel('Dataset2.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)
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
            # print(i, t)
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

    @staticmethod
    def SVM2_0():
        df = pd.read_excel('Dataset2.xlsx', sheet_name='Sheet1').drop(['Unnamed: 0'], axis=1)
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


if __name__ == "__main__":
    predManObj = PredManClass()

    # Dataset 1
    # predManObj.weibullDist()
    # predManObj.SVM()

    # Dataset 2
    # predManObj.weibullDist2_0()
    # predManObj.SVM2_0()

    # Metriche di valutazione su Dataset 2
    # predManObj.weibullLeaveOneOut()
    # predManObj.svmLeaveOneOut()

    predManObj.plot_synt_data()
