#!/usr/bin/env python
# coding: utf-8

# # PROJET 9 

# # IMPORT DES LIBRAIRIES, IMPORT DES FICHIERS, CENTRAGE ET REDUCTION

# In[ ]:


# importer les packages nécessaires
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.stats import skew
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import decomposition
from scipy.cluster.hierarchy import fcluster
from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import dendrogram
from pandas.plotting import parallel_coordinates
import seaborn as sns


# In[71]:


# importer les données
df_chicken = pd.read_excel('datachicken.xlsx')
df_final = df_chicken[['pays', 'importation_qt', 'nourriture_mille_to', 'production_mille_to', 'population', 'distance', 'PIB_tête_$', 'stabilite_politique']]

# utilisation du dataframe df en 7 variables ; et indexage en pays
df = df_chicken[['pays', 'importation_qt', 'nourriture_mille_to', 'production_mille_to', 'population', 'distance', 'PIB_tête_$', 'stabilite_politique']]
df = df.set_index('pays')


# In[72]:


# Afficher df
display(df)


# In[73]:


# Centrage, Réduction et Dendrogramme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


# # CAH

# In[74]:


# Clustering hiérarchique
Z = linkage(X_scaled, method='ward', metric='euclidean')

# Affichage du dendrogramme
plt.figure(figsize=(35, 9))
plt.title('Dendrogramme', fontsize=20)
plt.ylabel('Distance')
dendrogram(Z, labels=df.index, leaf_font_size=12, color_threshold=10, orientation='top')
plt.show()


# In[75]:


# 5 clusters avec le nombre de pays pour chacun
plt.figure(figsize=(15, 5))
plt.title('Dendrogramme en 5 clusters', fontsize=20)
plt.xlabel('distance', fontsize=15)
dendrogram(Z, labels=df.index, p=5, truncate_mode='lastp', leaf_font_size=15, orientation='left', show_contracted=True)
plt.show()


# In[76]:


# On récupère les données du dendrogramme
# DataFrame avec la liste des pays associée à leur cluster

cluster = fcluster(Z, t=5, criterion='maxclust')
pays_clustered = pd.DataFrame({'pays' : df.index.tolist(), 'cluster' : cluster})
pays_clustered = pays_clustered.sort_values(['cluster', 'pays'])
pays_clustered.head()

# jointure du dataframe avec la colonne cluster
pays_clustered = pd.merge(df, pays_clustered, on='pays')
pays_clustered.head()


# In[77]:


# Création des dataframes pour chaque cluster
clusters = [pays_clustered[pays_clustered['cluster'] == i] for i in range(1, 6)]

cluster1_cah = clusters[0]
cluster2_cah = clusters[1]
cluster3_cah = clusters[2]
cluster4_cah = clusters[3]
cluster5_cah = clusters[4]


# In[78]:


# Affichage du contenu des 5 clusters cah
print('Cluster 1 :', 'Nombre de pays:', len(clusters[0]), cluster1_cah['pays'].unique())
print('Cluster 2 :', 'Nombre de pays:', len(clusters[1]), cluster2_cah['pays'].unique())
print('Cluster 3 :', 'Nombre de pays:', len(clusters[2]), cluster3_cah['pays'].unique())
print('Cluster 4 :', 'Nombre de pays:', len(clusters[3]), cluster4_cah['pays'].unique())
print('Cluster 5 :', 'Nombre de pays:', len(clusters[4]), cluster5_cah['pays'].unique())


# In[38]:


# Boxplots : visualisation des clusters par variable
plt.figure(figsize=(25, 20)) 
sns.set(style="whitegrid")

plt.subplot(331)
sns.boxplot(data=pays_clustered, x='cluster', y='importation_qt')

plt.subplot(332)
sns.boxplot(data=pays_clustered, x='cluster', y='distance')

plt.subplot(333)
sns.boxplot(data=pays_clustered, x='cluster', y='production_mille_to')

plt.subplot(334)
sns.boxplot(data=pays_clustered, x='cluster', y='population')

plt.subplot(335)
sns.boxplot(data=pays_clustered, x='cluster', y='PIB_tête_$')

plt.subplot(336)
sns.boxplot(data=pays_clustered, x='cluster', y='stabilite_politique')

plt.subplot(337)
sns.boxplot(data=pays_clustered, x='cluster', y='nourriture_mille_to')

plt.show()


# In[79]:


# visualisation des centroides
df_cah_mean = pays_clustered.groupby('cluster').mean().round (decimals=6) 
display(df_cah_mean)


# L'analyse via boxplots des clusters sur les variables et la visualisation des centroides nous permet de nous focaliser sur les 2 clusters les plus intéressants ici : le cluster 3 et le cluster 5 ; cumulant les meilleures statistiques d'importation, les meilleures statistiques en terme de PIB par tête ainsi que les meilleures statistiques en terme de stabilité politique.

# In[80]:


# Regroupement des clusters 3-5 et visualisation des pays par importation, avec un seuil de population min > 1 000 000
cah_retenus = cluster5_cah.append(cluster3_cah)
cah_import = cah_retenus.sort_values('importation_qt', ascending=False)
pop_cah = cah_import[cah_import['population'] > 1000000]
pop_cah.head(10)


# En tête, on retrouve hong-kong, les émirats détenants de forts taux d'importations, tandis que les pays-bas et la belgique détiennent de bons taux d'importations, moins élevés mais géographiquement beaucoup plus avantageux d'un point de vu distance. Tous ces pays ont une population conséquente, une bonne stabilité politique et un PIB/tête élevé. Les deux pays en tête produisent moins que les deux suivants, mais consomment plus.

# # ACP

# In[81]:


#fonction pour afficher les composantes principales
def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

# Calcul des composantes principales
n_comp = 7
pca = decomposition.PCA(n_components=n_comp)
pca.fit(X_scaled)

eigenvalues = pca.explained_variance_
significant_eigenvalues = eigenvalues[eigenvalues > 1]
num_components = len(significant_eigenvalues)

# Eboulis des valeurs propres
display_scree_plot(pca)

#critere kaiser
print(significant_eigenvalues)
print("Nombre de composantes principales conservées selon le critère de Kaiser :", num_components)


# In[82]:


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from pandas.plotting import parallel_coordinates
import seaborn as sns


def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(7,6))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)


# In[83]:


from sklearn.decomposition import PCA
from sklearn import decomposition, preprocessing

n_comp=3

# Création du PCA pour réduire les données à 2 dimensions pour la visualisation
pca = PCA(n_components=n_comp)
pca.fit(X_scaled)

# Cercle des corrélations
pcs = pca.components_
display_circles(pcs, n_comp, pca, [(0,1),(0,2),(1,2)], labels = np.array(df.columns))


# F1 : Cette composante est fortement influencée par nourriture_mille_to (0.53), stabilite_politique (0.50), PIB_tête_$ (0.45) et importation_qt (0.42). Ces variables ont des loadings positifs, ce qui signifie qu’une augmentation de ces variables entraîne une augmentation de F1.
# F2 : Cette composante est principalement influencée par distance (0.60) et importation_qt (0.50). Une augmentation de ces variables entraîne une augmentation de F2. Cependant, production_mille_to (-0.48) et PIB_tête_$ (-0.37) ont des loadings négatifs, ce qui signifie qu’une augmentation de ces variables entraîne une diminution de F2.
# F3 : Cette composante est principalement influencée par distance (0.55), production_mille_to (0.56) et population (0.48). Une augmentation de ces variables entraîne une augmentation de F3. Cependant, importation_qt (-0.21) a un loading négatif, ce qui signifie qu’une augmentation de cette variable entraîne une diminution de F3.

# In[84]:


# Projection des individus (nouvel espace vectoriel)
X_projected = pca.transform(X_scaled)

# Affiche un scatter plot des points des data dans le nouvel espace vectoriel
display_factorial_planes(X_projected, n_comp, pca, [(0,1),(0,2),(1,2)], labels=None, illustrative_var = cluster, alpha = 0.8)

plt.savefig("PCA.png")
plt.show()


# On constate de manière évidente sur nos espaces vectoriels (F1-F2, F1-F3, F2-F3) que les clusters dominants quant aux statistiques d'importations sont le 3 et le 5. Dans une très, très moindre mesure, le 2.

# In[85]:


#Centroides
X2 = df_cah_mean.values
std_scale = preprocessing.StandardScaler().fit(X2)
X2_scaled = std_scale.transform(X2)

# Projection des centroïdes des clusters sur les deux composantes
X2_projected = pca.transform(X2_scaled)
display_factorial_planes(X2_projected, n_comp, pca, [(0,1),(0,2),(1,2)], illustrative_var=df_cah_mean.index)
plt.savefig("centroid.png")
plt.show()


# La visualisation des centroides sur nos espaces vectoriels confirme nos observations précédentes, les clusters 3 et 5 influent sur les importations, de même que le 5 est souvent à l'opposé de distance, qui confirme la proximité des pays du cluster 5 comme la belgique ou les pays-bas.

# # KMEANS

# In[86]:


from sklearn.cluster import KMeans
import sklearn.metrics as metrics

#Liste pour stocker nos coefficients
silhouettes = [] 

#Boucle itérative de 2 à 10 clusters pour tester toutes les possibilités de k
for k in range(2, 10): 
    #Création et ajustement d'un modèle pour chaque k
    cls = KMeans(n_clusters=k, random_state=0)  
    cls.fit(X_scaled)
    
    #Stockage des coefficients associés
    silh = metrics.silhouette_score(X_scaled, cls.labels_)
    silhouettes.append(silh)
    
# Visualisation des valeurs de coefficient de silhouette pour chaque nombre de cluster
plt.plot(range(2, 10), silhouettes, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Score de silhouette')
plt.title('Score de silhouette en fonction du nombre de clusters')
plt.savefig('silhouette.png')
plt.show()


# Le score de silhouette est une mesure utilisée pour évaluer la cohérence et la compacité des clusters obtenus à partir d'un algorithme de clustering comme K-means. Il fournit une indication de la qualité de la séparation entre les clusters. Plus le score est haut et proche de 1, plus cela indique que les instances sont bien regroupées dans leur propre cluster et sont éloignées des autres clusters, ce qui est souhaitable. Le score le plus haut de silhouette définit notre nombre de clusters idéal, soit 5.

# In[87]:


#Méthode du coude
#On crée une liste dans laquelle on stocke les inerties
inerties=[]

#On fait une boucle de 2 à 9 pour tester toutes ces possibiliéts
for k in range(2, 10):
    #pour chaque k, on crée un modèle et on l’ajuste
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(X_scaled)
    #on stocke l’inertie associée
    inerties.append(km.inertia_)

#Visualisation des valeurs d'inertie pour chaque nombre de cluster
plt.plot(range(2, 10), inerties, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
plt.savefig('coude.png')
plt.show()


# Le point où la diminution de l'inertie ralentit brusquement est au nombre de clusters de 5, où on peut observer la cassure, ce qui nous confirme ce que Silhouette nous indiquait.

# In[88]:


# Création du modèle k-means 
kmeans = KMeans(n_clusters=5, n_init=20, init='k-means++', random_state=1)
kmeans.fit(X_scaled)

# Détermine à quel cluster appartient chaque point (kmeans.labels_)
clusters =  kmeans.predict(X_scaled)

# nombre d'individus (pays) dans chaque cluster
np.unique(kmeans.labels_, return_counts=True)


# In[89]:


#Ajout d'une nouvelle colonne qui affecte à chaque pays un numéro de cluster
df_km=df.copy()
df_km['clusters_km'] = kmeans.labels_
df_km.head()


# In[90]:


df_km=df_km.reset_index()
# Dataframe de chaque cluster
clusterkm0=df_km[df_km['clusters_km']==0]
clusterkm1=df_km[df_km['clusters_km']==1]
clusterkm2=df_km[df_km['clusters_km']==2]
clusterkm3=df_km[df_km['clusters_km']==3]
clusterkm4=df_km[df_km['clusters_km']==4]


# In[91]:


# Affichage des pays de chaque clusters kmeans
print('Cluster 0 :', 'Nombre de pays:', len(clusterkm0), clusterkm0['pays'].unique())
print('Cluster 1 :', 'Nombre de pays:', len(clusterkm1), clusterkm1['pays'].unique())
print('Cluster 2 :', 'Nombre de pays:', len(clusterkm2), clusterkm2['pays'].unique())
print('Cluster 3 :', 'Nombre de pays:', len(clusterkm3), clusterkm3['pays'].unique())
print('Cluster 4 :', 'Nombre de pays:', len(clusterkm4), clusterkm4['pays'].unique())


# Avec le clustering kmeans, les pays semble visuellement mieux répartis (un cluster à 110 pays avec CAH).

# In[92]:


#Affichage en boxplots des clusters kmeans pour chaque variable
plt.figure(figsize=(25, 15))
sns.set(style="whitegrid")

plt.subplot(2, 3, 1)
plt.subplot(231)
sns.boxplot(data=df_km, x='clusters_km', y='importation_qt')

plt.subplot(232)
sns.boxplot(data=df_km, x='clusters_km', y='nourriture_mille_to')

plt.subplot(233)
sns.boxplot(data=df_km, x='clusters_km', y='stabilite_politique')

plt.subplot(234)
sns.boxplot(data=df_km, x='clusters_km', y='population')

plt.subplot(235)
sns.boxplot(data=df_km, x='clusters_km', y='distance')

plt.subplot(236)
sns.boxplot(data=df_km, x='clusters_km', y='PIB_tête_$')

plt.savefig("boxplots.png")
plt.show()


# In[94]:


# Centroïdes des 5 clusters dans la forme centrée réduite
centroids = kmeans.cluster_centers_
df_cent=pd.DataFrame(centroids, columns=df.columns).round (decimals = 6)

# Afficher
display(df_cent)

# Heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df_cent,annot=True, vmin=-1, vmax=1)


# La visualisation via boxplots et l'observation des centroides via heatmap nous permet d'interprêter nos clusters : 
# 
# Cluster 0 : Caractérisé par des valeurs relativement faibles pour les variables importation_qt, nourriture_mille_to, production_mille_to, distance, PIB_tête_$, et stabilite_politique, avec une valeur légèrement négative pour population. Cela pourrait correspondre à des pays qui ont une faible consommation, production et importation de viande, globalement assez instables politiquement et avec un PIB par habitant relativement bas.
# 
# Cluster 1 : Ce cluster a des valeurs plus élevées pour nourriture_mille_to, production_mille_to, et distance, avec une valeur négative pour population. Cela pourrait représenter des pays avec une production alimentaire relativement élevée, une distance géographique importante, et une population relativement faible.
# 
# Cluster 2 : Ce cluster se distingue par une valeur très élevée pour population, ce qui suggère des pays très peuplés. Les autres variables ont des valeurs relativement basses, ce qui peut indiquer une faible consommation, production et importation de viande, ainsi qu'une instabilité politique et un PIB par habitant relativement bas.
# 
# Cluster 3 : Ce cluster a des valeurs positives pour toutes les variables hormis la population, ce qui indique une population assez faible, et également pour la distance ce qui indique des pays relativement proches. Les valeurs sont les plus élevées pour PIB_tête_$ et stabilite_politique, suggérant des pays plus riches et politiquement stables. Les statistiques d'importations positives, en plus de cela, nous permettent de retenir ce cluster.
# 
# Cluster 4 : Ce cluster se caractérise par des valeurs très élevées pour importation_qt, nourriture_mille_to, distance, PIB_tête_$, et stabilite_politique, ce qui peut représenter des pays avec une forte consommation et importation de viande, une distance géographique importante, un PIB par habitant élevé et une stabilité politique notable.
# 
# Les clusters 3 et 4 ont en commun les bonnes statistiques d'importations, un PIB/tête élevé, ainsi qu'une stabilité politique élevée. Le 1er a des statistiques d'importations moins prononcées mais une distance beaucoup plus faible (donc pays plus proches), alors que le second indique des pays plus lointains mais à très fortes importations de volailles.

# # ACP

# In[96]:


# ACP en 3 dimensions et cercles de corrélation sur nos 3 espaces vectoriels F1-F2 / F1-F3 / F2-F3

n_comp=3

# Création du PCA pour réduire les données à 2 dimensions pour la visualisation
pca = PCA(n_components=n_comp)
pca.fit(X_scaled)

# Cercle des corrélations
pcs = pca.components_
display_circles(pcs, n_comp, pca, [(0,1),(0,2),(1,2)], labels = np.array(df.columns))


# In[97]:


# Accéder à la colonne 'clusters_km'
clusters_km = df_km['clusters_km']

# Afficher le scatter plot des points des données dans le nouvel espace vectoriel
display_factorial_planes(X_projected, n_comp, pca, [(0,1),(0,2),(1,2)], labels=None, illustrative_var=clusters, alpha=0.8)

plt.savefig("PCA_with_clusters.png")
plt.show()


# In[98]:


# Créer des DataFrame pour chaque cluster
clusterkm0 = df_km[df_km['clusters_km'] == 0]
clusterkm1 = df_km[df_km['clusters_km'] == 1]
clusterkm2 = df_km[df_km['clusters_km'] == 2]
clusterkm3 = df_km[df_km['clusters_km'] == 3]
clusterkm4 = df_km[df_km['clusters_km'] == 4]


# In[99]:


# Coordonnées factorielles
plt.figure(figsize=(40, 15))
plt.subplot(121)
plt.scatter(X_projected[:, 0], X_projected[:, 1], c=kmeans.labels_, cmap='plasma')
for i, (x, y) in enumerate(X_projected[:, [0, 1]]):
    plt.text(x, y, df.index[i], fontsize='13')  # Utilisation des noms des pays de l'index
plt.xlabel('F{} ({}%)'.format(1, round(100*pca.explained_variance_ratio_[0],1)))
plt.ylabel('F{} ({}%)'.format(2, round(100*pca.explained_variance_ratio_[1],1)))
plt.title("Projection en 5 clusters des {} individus sur le 1er plan factoriel".format(X_projected.shape[0]))
plt.legend()
plt.savefig("projection_clusters.png")
plt.show()


# In[100]:


# Coordonnées factorielles sur les plans F1 et F3
plt.figure(figsize=(40, 15))
plt.subplot(121)
plt.scatter(X_projected[:, 0], X_projected[:, 2], c=kmeans.labels_, cmap='plasma')
for i, (x, y) in enumerate(X_projected[:, [0, 2]]):
    plt.text(x, y, df.index[i], fontsize='13')  # Utilisation des noms des pays de l'index
plt.xlabel('F{} ({}%)'.format(1, round(100*pca.explained_variance_ratio_[0], 1)))
plt.ylabel('F{} ({}%)'.format(3, round(100*pca.explained_variance_ratio_[2], 1)))
plt.title("Projection en 5 clusters des {} individus sur les plans F1 et F3".format(X_projected.shape[0]))
plt.legend()
plt.savefig("projection_clusters_F1_F3.png")
plt.show()


# In[101]:


# Coordonnées factorielles sur les plans F2 et F3
plt.figure(figsize=(40, 15))
plt.subplot(121)
plt.scatter(X_projected[:, 1], X_projected[:, 2], c=kmeans.labels_, cmap='plasma')
for i, (x, y) in enumerate(X_projected[:, [1, 2]]):
    plt.text(x, y, df.index[i], fontsize='13')  # Utilisation des noms des pays de l'index
plt.xlabel('F{} ({}%)'.format(2, round(100*pca.explained_variance_ratio_[1], 1)))
plt.ylabel('F{} ({}%)'.format(3, round(100*pca.explained_variance_ratio_[2], 1)))
plt.title("Projection en 5 clusters des {} individus sur les plans F2 et F3".format(X_projected.shape[0]))
plt.legend()
plt.savefig("projection_clusters_F2_F3.png")
plt.show()


# Hong-kong, les émirats-arabes unis, ainsi que beaucoup d'îles du pacifique et des caraibes comme les samoa ou les bahamas dominent statistiquement sur les taux d'importations de volailles, suivis de pays plus "modeste" sur les taux d'importations mais disposant d'autres atouts comme la distance : pays-bas, luxembourg...

# # OBSERVATIONS ET CONCLUSION

# In[111]:


df_km_retenus=clusterkm3.append(clusterkm4)
df_import = df_km_retenus.sort_values('importation_qt', ascending=False)
df_import = df_import.head(20)
df_import


# In[112]:


# Filtrer les clusters avec une population supérieure à 3 millions d'habitants
pop_sup = df_import[df_import['population'] > 1000000]

# Définir le format d'affichage pour la colonne 'importation_qt'
pop = pop_sup.style.format({'importation_qt': '{:.6f}'.format})

# Afficher les clusters triés
display(pop)


# Comparativement au CAH, la méthode du Kmeans nous démontre 4 pays considérablement plus intéressants que d'autres : hong-kong, les émirats-arabes unis, les pays-bas et la belgique. Selon la stratégie à adopter, on peut choisir les 4, ou filtrer d'avantage : je recommanderais dans ces 4, d'écarter hong-kong car plus éloignés que les autres, et de garder un très gros importateur comme les émirats, assez loin mais à une distance acceptable. Bien que les pays-bas et la belgique soient de moins fort importateurs, il détiennent cependant un bon score et ont surtout l'avantage d'une distance moindre : la belgique est limitrophe, les pays-bas juste après la belgique. Le koweit est écarté pour sa stabilité politique négative qui peut représenter des risques.
# 
# Le meilleur choix reste cependant d'opter pour les 4, hong-kong ayant un taux d'importations de poulets gigantesque.
