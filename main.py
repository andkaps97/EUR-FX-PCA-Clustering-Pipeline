import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
import plotly.express as px
from statsmodels.tools.eval_measures import rmse
from dcor import distance_correlation
import seaborn as sns


def adf_test(series):
    """Return p-value from ADF test."""
    return adfuller(series, autolag='AIC')[1]


def pca_analysis_with_threshold(data, columns_for_pca, variance_threshold=0.6667):
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    pca = PCA().fit(data_standardized)
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    components_needed = (cumulative_variance < variance_threshold).sum() + 1
    plot_pca_loadings(pca, columns_for_pca)
    return components_needed, cumulative_variance


def plot_pca_loadings(pca, columns_for_pca, components_to_plot=5):
    loadings = pca.components_
    num_features = loadings.shape[1]
    plt.figure(figsize=(10, 6))
    for i in range(components_to_plot):
        plt.subplot(1, components_to_plot, i + 1)
        plt.barh(range(num_features), loadings[i, :])
        plt.title(f'PCA Component {i+1}')
        plt.yticks(range(num_features), columns_for_pca)
    plt.tight_layout()
    plt.show()


def check_pca_components_stationarity(pca_components):
    p_values = [adf_test(pca_components[:, i]) for i in range(pca_components.shape[1])]
    non_stationary_components = np.sum(np.array(p_values) > 0.05)
    print(f"Number of non-stationary components: {non_stationary_components}")
    return p_values


class ExchangeRatePipeline:
    def __init__(self, filepath, selected_currencies):
        self.filepath = filepath
        self.selected = selected_currencies

    def load(self):
        df = pd.read_csv(self.filepath, index_col='Date')
        df.index = pd.to_datetime(df.index)
        df = df.dropna(thresh=len(df)*0.5, axis=1)
        return df

    def preprocess(self, df):
        cleaned = df[self.selected].copy()
        cleaned = cleaned.fillna(method='ffill').fillna(method='bfill')
        non_stat = [cur for cur in self.selected if adf_test(cleaned[cur]) > 0.05]
        for cur in non_stat:
            cleaned[f'{cur}_r'] = cleaned[cur].diff()
        cleaned_diff = cleaned.dropna().reset_index(drop=True)
        columns_for_pca = [f'{cur}_r' for cur in non_stat]
        data_for_pca = cleaned_diff[columns_for_pca]
        return cleaned_diff, data_for_pca, columns_for_pca

    def run_pca(self, data_for_pca, columns_for_pca):
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data_for_pca)
        components_needed, cumulative_variance = pca_analysis_with_threshold(data_for_pca, columns_for_pca)
        print("Components needed:", components_needed)
        print("Cumulative Variance Explained:", cumulative_variance)
        # Scree plot
        plt.figure(figsize=(10,6))
        plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linestyle='--')
        plt.axvline(x=components_needed, color='red', linestyle='--', label=f'Elbow Point at Component {components_needed}')
        plt.title('Scree Plot')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance Explained')
        plt.legend()
        plt.show()
        # PCA transform
        pca = PCA(n_components=components_needed)
        data_pca = pca.fit_transform(data_standardized)
        return pca, data_pca

    def cluster(self, data_pca, num_clusters=5):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(data_pca)
        labels = kmeans.labels_
        plt.figure(figsize=(10,6))
        for i in range(num_clusters):
            plt.scatter(data_pca[labels==i,0], data_pca[labels==i,1], label=f'Cluster {i+1}')
        plt.title('Clusters of Exchange Rates based on PCA Components')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.show()
        return labels


class MacroFinancePipeline:
    def __init__(self, base_index):
        self.base_index = base_index

    def load_data(self):
        fed_funds = pd.read_csv('FEDFUNDS.csv', parse_dates=['Date'], index_col='Date').sort_index()
        interest_rates = pd.read_csv('rates.csv', parse_dates=['Date'], index_col='Date').sort_index()
        dxy = pd.read_csv('DX-Y.NYB.csv', parse_dates=['Date'], index_col='Date').sort_index()
        gold = pd.read_csv('gold.csv', parse_dates=['Date'], index_col='Date', thousands=',').sort_index()
        msci = pd.read_csv('MSCI World Historical Data.csv', parse_dates=['Date'], index_col='Date').sort_index()
        gdp = pd.read_csv('gross domestic protact.csv', parse_dates=['Date'], index_col='Date').sort_index()
        EU_inflation = pd.read_csv('euro inflation rate.csv', parse_dates=['Date'], index_col='Date').sort_index()
        stocks = pd.read_csv('eu stock.csv', parse_dates=['Date'], index_col='Date').sort_index()
        t10_2 = pd.read_excel('T10Y2Y.xls', index_col=0, parse_dates=True).sort_index()
        vix = pd.read_excel('VIXCLS.xls', index_col=0, parse_dates=True).sort_index()
        oil_prices = pd.read_excel('oil.xls', index_col=0, parse_dates=True).sort_index()
        financial_data = pd.DataFrame({
            'DXY': dxy['Close DXY'].reindex(self.base_index).interpolate(),
            'Gold': gold['Price'].reindex(self.base_index).interpolate(),
            'T10Y2Y': t10_2['T10Y2Y'].reindex(self.base_index).interpolate(),
            'VIX': vix['VIXCLS'].reindex(self.base_index).interpolate(),
            'Oil': oil_prices['DCOILWTICO'].reindex(self.base_index).interpolate(),
            'rate_diff': fed_funds['FEDFUNDS'].reindex(self.base_index).interpolate() - interest_rates['rates'].reindex(self.base_index).interpolate(),
            'Inflation': EU_inflation['HICP '].reindex(self.base_index).interpolate(),
            'EU STOXX': stocks['EURO STOXX'].reindex(self.base_index).interpolate(),
            'GDP': gdp['GDP'].reindex(self.base_index).interpolate()
        })
        financial_data = financial_data.fillna(method='bfill').fillna(method='ffill')
        return financial_data

    def make_stationary(self, financial_data):
        for column in financial_data.columns:
            if adf_test(financial_data[column]) > 0.05:
                financial_data[column] = financial_data[column].diff()
                print(f"Differencing {column} to achieve stationarity")
        financial_data = financial_data.dropna().reset_index(drop=True)
        return financial_data

    def compute_correlations(self, pca_components_df, financial_data_standardized_df):
        aligned = pca_components_df.join(financial_data_standardized_df, how='inner').bfill()
        correlations = {}
        spearman_correlations = {}
        for column in financial_data_standardized_df.columns:
            correlations[column] = []
            spearman_correlations[column] = []
            for i in range(pca_components_df.shape[1]):
                series_x = aligned.iloc[:, i]
                series_y = aligned[column]
                if len(series_x) >= 2 and len(series_y) >= 2:
                    dist_corr = distance_correlation(series_x, series_y)
                    _, _ = pearsonr(series_x, series_y)
                    corrs, _ = spearmanr(series_x, series_y)
                    spearman_correlations[column].append(corrs)
                    correlations[column].append(dist_corr)
                else:
                    correlations[column].append(np.nan)
                    spearman_correlations[column].append(np.nan)
        correlation_df = pd.DataFrame(correlations, index=[f'PC{i+1}' for i in range(pca_components_df.shape[1])])
        plt.figure(figsize=(17,8))
        sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation between Principal Components and Financial/Macroeconomic Series")
        plt.show()
        return correlation_df

    def fit_ar_models(self, equity_returns, pca_components_df):
        lags_to_test = [1,2,3]
        results = []
        for lags in lags_to_test:
            model_ar, rmse_ar, t_stats_ar, pseudo_r_squared_ar, coefs_ar = fit_ar_model(equity_returns, lags)
            model_with_pcs, rmse_with_pcs, coefs_with_pcs = fit_ar_model_with_pcs(equity_returns, pca_components_df, lags)
            results.append({
                'AR_lags': lags,
                'RMSE_AR': rmse_ar,
                'RMSE_AR_with_PCs': rmse_with_pcs,
                'AIC_AR': model_ar.aic,
                'AIC_AR_with_PCs': model_with_pcs.aic,
                'BIC_AR': model_ar.bic,
                'BIC_AR_with_PCs': model_with_pcs.bic,
                'R_squared_AR': pseudo_r_squared_ar,
                'R_squared_AR_with_PCs': model_with_pcs.rsquared,
            })
        df_results = pd.DataFrame(results)
        print(df_results)
        return df_results


def fit_ar_model(equity_returns, lags):
    model_ar = AutoReg(equity_returns, lags=lags).fit()
    predictions_ar = model_ar.predict(start=lags, end=len(equity_returns)-1, dynamic=False)
    rmse_ar = rmse(equity_returns[lags:], predictions_ar)
    ss_res = np.sum((equity_returns[lags:]-predictions_ar)**2)
    ss_tot = np.sum((equity_returns[lags:]-np.mean(equity_returns[lags:]))**2)
    pseudo_r_squared = 1 - ss_res/ss_tot
    coefs_ar = model_ar.params
    return model_ar, rmse_ar, model_ar.tvalues, pseudo_r_squared, coefs_ar


def fit_ar_model_with_pcs(equity_returns, pca_components_df, lags):
    df = pd.concat([equity_returns.shift(i) for i in range(1, lags+1)], axis=1).join(pca_components_df.shift(lags))
    df.columns = [f'Lag_{i}' for i in range(1, lags+1)] + [f'PC_{i+1}' for i in range(pca_components_df.shape[1])]
    df['Equity_Returns'] = equity_returns
    df = df.dropna()
    X = df.drop('Equity_Returns', axis=1)
    y = df['Equity_Returns']
    model_with_pcs = sm.OLS(y, sm.add_constant(X)).fit()
    rmse_with_pcs = rmse(y, model_with_pcs.predict(sm.add_constant(X)))
    return model_with_pcs, rmse_with_pcs, model_with_pcs.params


def plot_pca_loadings_dotplot(pca, columns_for_pca):
    loadings = pca.components_
    num_components, num_features = loadings.shape
    loadings_df = pd.DataFrame(loadings.T, index=columns_for_pca, columns=[f'Component {i+1}' for i in range(num_components)])
    df_melt = loadings_df.reset_index().melt(id_vars='index', var_name='Component', value_name='Loading')
    fig = px.scatter(df_melt, x='Loading', y='index', color='Component', symbol='Component', title='PCA Component Loadings Dot Plot')
    fig.update_traces(marker=dict(size=12))
    fig.show()


def pca_biplot(score, coeff, labels=None):
    xs, ys = score[:,0], score[:,1]
    n = coeff.shape[0]
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(xs, ys, alpha=0.5)
    for i in range(n):
        ax.arrow(0,0, coeff[i,0], coeff[i,1], color='r', alpha=0.5)
        if labels is not None:
            ax.text(coeff[i,0]*1.15, coeff[i,1]*1.15, labels[i], color='green', ha='center', va='center')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); plt.grid(); plt.show()


def main():
    # Exchange Rate Pipeline
    selected_currencies = ['USD','JPY','GBP','AUD','CAD','CHF','CNY','INR','BRL','IDR',
                           'ZAR','MXN','SGD','NZD','NOK','SEK','TRY','KRW','HKD','MYR']
    erp = ExchangeRatePipeline('eurofxref-hist.csv', selected_currencies)
    df = erp.load()
    cleaned_diff, data_for_pca, columns_for_pca = erp.preprocess(df)
    pca, data_pca = erp.run_pca(data_for_pca, columns_for_pca)
    erp.cluster(data_pca)

    # Macro Finance Pipeline
    mfp = MacroFinancePipeline(df.index)
    financial_data = mfp.load_data()
    stationary_fin = mfp.make_stationary(financial_data)
    pca_components_df = pd.DataFrame(data_pca, index=cleaned_diff.index, columns=[f'PC{i+1}' for i in range(data_pca.shape[1])])
    financial_data_standardized_df = stationary_fin.copy()
    mfp.compute_correlations(pca_components_df, financial_data_standardized_df)
    mfp.fit_ar_models(stationary_fin['EU STOXX'], pca_components_df)

    # Additional Visualizations
    plot_pca_loadings_dotplot(pca, columns_for_pca)
    pca_df = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(data_pca.shape[1])], index=cleaned_diff.index)
    fig = px.line(pca_df, y='PC1', title='Time Series of PCA Component 1')
    fig.update_xaxes(title='Date'); fig.update_yaxes(title='Value'); fig.show()
    pca_biplot(data_pca[:,:2], np.transpose(pca.components_[:2,:]), labels=columns_for_pca)

    # Rolling correlations
    df_aligned = df.iloc[1:].copy()
    pca_components_df.index = df_aligned.index
    financial_data_standardized_df.index = df_aligned.index
    window_size = 120
    roll_corr1 = -pca_components_df['PC1'].rolling(window=window_size).corr(financial_data_standardized_df['DXY']).rolling(window=5).mean()
    plt.figure(figsize=(15,7)); plt.plot(roll_corr1.index, roll_corr1, linewidth=2, label='Rolling Correlation'); plt.title('Rolling Correlation PC1 vs DXY'); plt.legend(); plt.show()
    roll_corr2 = pca_components_df['PC2'].rolling(window=window_size).corr(financial_data_standardized_df['Oil']).rolling(window=5).mean()
    plt.figure(figsize=(15,7)); plt.plot(roll_corr2.index, roll_corr2, linewidth=2, label='Rolling Correlation'); plt.title('Rolling Correlation PC2 vs Oil'); plt.legend(); plt.show()

if __name__ == '__main__':
    main()
