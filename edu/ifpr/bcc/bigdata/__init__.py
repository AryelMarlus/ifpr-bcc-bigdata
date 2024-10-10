import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import shapiro
from scipy.stats import kstest
import scipy.stats as stats

pk = pd.read_excel(r'C:\Users\aryel\OneDrive\Documentos\IFPR\Aulas\Big Data\PokemonGO.xlsx', sheet_name=0)
print(pk.columns)

#Shapiro-Wilk
stat, p = shapiro(pk['DPS'])
print(p)
stat, p = shapiro(pk['TDO'])
print(p)

#Kolmogorov-Smirnov
stat, p = kstest(pk['DPS'], 'norm')
print(p)
stat, p = kstest(pk['TDO'], 'norm')
print(p)

media_x = pk['DPS'].mean()
media_y = pk['TDO'].mean()

plt.figure()
pk['DPS'].sort_values().plot(kind='bar')
plt.axhline(media_x, color='red')
plt.show()

pk['TDO'].sort_values().plot(kind='bar')
plt.axhline(media_y, color='red')
plt.show()

stats.probplot(pk['DPS'], dist="norm", plot=plt)
plt.show()
stats.probplot(pk['TDO'], dist="norm", plot=plt)
plt.show()

plt.figure()
plt.scatter(pk['DPS'], pk['TDO'])
plt.axvline(media_x, color='red')
plt.axhline(media_y, color='red')
plt.show(block=False)

plt.figure()
scaler = StandardScaler()
plt.scatter(scaler.fit_transform(pk[['DPS']]), scaler.fit_transform(pk[['TDO']]))
plt.show()

plt.figure()
pca = PCA(n_components=2)
componentesPrincipais = pca.fit_transform(pk[['DPS','TDO','Score']])
plt.scatter(componentesPrincipais[:,0], componentesPrincipais[:,1])
plt.show(block=False)
print("variância PCA")
print(sum(pca.explained_variance_ratio_))

plt.figure()
pca = PCA(n_components=2)
componentesPrincipais = pca.fit_transform(pk[['DPS','TDO']])
print(componentesPrincipais)
plt.scatter(componentesPrincipais[:,0], componentesPrincipais[:,1])
plt.show()

# Marcando pontos
plt.figure()
semPCA = pk[['DPS', 'TDO']]
plt.scatter(semPCA['DPS'], semPCA['TDO'], label='Dados Normais', color='lightgray')
indices_maior_media = pk['DPS'] > media_x
plt.scatter(semPCA.loc[indices_maior_media, 'DPS'], semPCA.loc[indices_maior_media, 'TDO'],
            label='DPS > Média', color='red', edgecolor='black', s=100)

plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid(True)
plt.show(block =False)

plt.figure()
plt.scatter(semPCA['DPS'], semPCA['TDO'], label='Dados Normais', color='lightgray')
indices_maior_media = pk['TDO'] > media_y
plt.scatter(semPCA.loc[indices_maior_media, 'DPS'], semPCA.loc[indices_maior_media, 'TDO'],
            label='TDO > Média', color='red', edgecolor='black', s=100)

plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid(True)
plt.show()

# Marcando pontos pca
plt.figure()

plt.scatter(componentesPrincipais[:, 0], componentesPrincipais[:, 1], label='Dados Normais', color='lightgray')
indices_maior_media = pk['DPS'] > media_x
plt.scatter(componentesPrincipais[indices_maior_media, 0], componentesPrincipais[indices_maior_media, 1],
            label='DPS > Média', color='red', edgecolor='black', s=100)
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid(True)
plt.show(block =False)

plt.figure()

plt.scatter(componentesPrincipais[:, 0], componentesPrincipais[:, 1], label='Dados Normais', color='lightgray')
indices_maior_media = pk['TDO'] > media_y
plt.scatter(componentesPrincipais[indices_maior_media, 0], componentesPrincipais[indices_maior_media, 1],
            label='TDO > Média', color='red', edgecolor='black', s=100)
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.legend()
plt.grid(True)
plt.show()
