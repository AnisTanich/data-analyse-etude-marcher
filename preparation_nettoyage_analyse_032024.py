#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Import des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scrapy import Selector
from requests import get


# # AJOUTS ET IMPORTS DE FICHIERS

# In[4]:


# Ajouter un dataframe de distance par rapport à la france via webscrapping
df = pd.DataFrame(columns=['Zone', 'Distance'])

# URL du site web à retrouver et vérification de la réponse du serveur
url = "https://www.distance-between-countries.com/countries/distance_between_countries.php?from=France&language=French"
response = get(url)
source = None
if response.status_code == 200:
    source = response.text

# Extraction des données de la page web
selector = Selector(text=source)
table = selector.css("table.center_bordered > tr > td").getall()

# Sélection des éléments HTML de table contenant les données et itération 
for country in table:
    selector = Selector(text=country)
    zone = selector.css("b > a::text").get()
    distance = selector.css("td::text").getall()
    if(zone is not None):
      df = df.append(pd.Series([zone, distance[1]], index=df.columns), ignore_index=True)

# Mise en forme des valeurs de la colonne 'Distance'
df['Distance'] = df['Distance'].str.replace('est de', '')
df['Distance'] = df['Distance'].str.replace(',', '')
df['Distance'] = df['Distance'].str.replace('kilomètres', '')
df['Distance'] = df['Distance'].str.replace(' ', '')

# Dataframe en excel
csv_path = r'C:\Users\djaba\OneDrive\Documents\P9\scrapper\distance.csv'
distance.to_csv(csv_path, index=False)


# In[6]:


# Imports des fichiers de base et de ceux ajoutés (réaménagés en CSV, et/ou renommés) : pib, stabilité politique et distance

dispo2017 = pd.read_csv('DisponibiliteAlimentaire_2017.csv', sep= ',')
pop0018 = pd.read_csv('Population_2000_2018.csv', sep= ',')
distance = pd.read_csv('df_distance.csv', sep=',')
pib = pd.read_csv('pib.csv', sep= ',')
stabilite = pd.read_csv('stabilite.csv', skiprows=3)


# # NETTOYAGE ET AJUSTEMENTS

# In[7]:


# Renommer 'Zone' en 'pays'
pop0018.rename(columns={'Zone': 'pays'}, inplace=True)
dispo2017.rename(columns={'Zone': 'pays'}, inplace=True)
stabilite.rename(columns={'Country Name': 'pays'}, inplace=True)
pib.rename(columns={'Zone': 'pays'}, inplace=True)


# In[8]:


# Obtenir les listes des noms de pays dans chaque DataFrame
pays_distance = set(distance['pays'].unique())
pays_dispo2017 = set(dispo2017['pays'].unique())
pays_pop0018 = set(pop0018['pays'].unique())

# Identifier les pays absents dans le DataFrame "distance" mais présents dans les autres DataFrames
pays_absents_distance_dispo2017 = pays_dispo2017 - pays_distance
pays_absents_distance_pop0018 = pays_pop0018 - pays_distance

# Obtenir les listes des noms de pays dans chaque DataFrame
pays_distance = set(distance['pays'])
pays_pib = set(pib['pays'])
pays_stabilite = set(stabilite['pays'])

# Identifier les pays absents dans le DataFrame "distance" mais présents dans les autres DataFrames
pays_absents_distance_pib = pays_pib - pays_distance
pays_absents_distance_stabilite = pays_stabilite - pays_distance

# Afficher les pays absents dans le DataFrame "distance"
print("Pays absents dans distance mais présents dans dispo2017 :", pays_absents_distance_dispo2017)
print("Pays absents dans distance mais présents dans pop0018 :", pays_absents_distance_pop0018)
print("Pays absents dans distance mais présents dans pib :", pays_absents_distance_pib)
print("Pays absents dans distance mais présents dans stabilité :", pays_absents_distance_stabilite)


# In[9]:


# Remplacements dans le DataFrame dispo2017
dispo2017.replace({'République de Corée': 'Corée du Sud',
                   'Sao Tomé-et-Principe': 'Sao Tomé-et-Príncipe',
                   'République-Unie de Tanzanie': 'Tanzanie',
                   'Cabo Verde': 'Cap-Vert',
                   'Fédération de Russie': 'Russie',
                   'Brunéi Darussalam': 'Brunei',
                   'Myanmar': 'Birmanie',
                   'Nigéria': 'Nigeria',
                   'République de Moldova': 'Moldavie',
                   'Chine, Taiwan Province de': 'Taïwan',
                   'Tchéquie': 'République tchèque',
                   "Iran (République islamique d')": 'Iran',
                   'Venezuela (République bolivarienne du)': 'Venezuela',
                   'République populaire démocratique de Corée': 'Corée du Nord',
                   'Iraq': 'Irak',
                   'Viet Nam': 'Viêt Nam',
                   'Chine - RAS de Macao': 'Macao',
                   'République démocratique populaire lao': 'Laos',
                   'Timor-Leste': 'Timor oriental',
                   'El Salvador': 'Salvador',
                   'Eswatini': 'Swaziland',
                   'Saint-Kitts-et-Nevis': 'Saint-Christophe-et-Niévès',
                   'Chine - RAS de Hong-Kong': 'Hong Kong',
                   'Bolivie (État plurinational de)': 'Bolivie',
                   'Bélarus': 'Biélorussie',
                   'Chine, continentale': 'Chine',
                   "Royaume-Uni de Grande-Bretagne et d'Irlande du Nord": 'Royaume-Uni',
                   "États-Unis d'Amérique": 'États-Unis',
                   'Émirats arabes unis': 'Émirats Arabes Unis'}, inplace=True)

# Remplacements dans le DataFrame pop0018
pop0018.replace({'République de Corée': 'Corée du Sud',
                 'Sao Tomé-et-Principe': 'Sao Tomé-et-Príncipe',
                 'République-Unie de Tanzanie': 'Tanzanie',
                 'Cabo Verde': 'Cap-Vert',
                 'Fédération de Russie': 'Russie',
                 'Saint-Martin (partie française)': 'Saint-Martin',
                 'Brunéi Darussalam': 'Brunei',
                 'Myanmar': 'Birmanie',
                 'Nigéria': 'Nigeria',
                 'Serbie-et-Monténégro': 'Serbie',
                 'République de Moldova': 'Moldavie',
                 'Micronésie (États fédérés de)': 'Micronésie',
                 'Chine, Taiwan Province de': 'Taïwan',
                 'Tchéquie': 'République tchèque',
                 'Sint Maarten  (partie néerlandaise)': 'Sint Maarten',
                 'Bonaire, Saint-Eustache et Saba': 'Bonaire',
                 "Iran (République islamique d')": 'Iran',
                 'Nioué': 'Niue',
                 'Sainte-Hélène, Ascension et Tristan da Cunha': 'Sainte-Hélène',
                 'Saint-Barthélemy': 'Saint-Barthélémy',
                 'Soudan (ex)': 'Soudan',
                 'Venezuela (République bolivarienne du)': 'Venezuela',
                 'République populaire démocratique de Corée': 'Corée du Nord',
                 'Curaçao': 'Curaçao',
                 'Iraq': 'Irak',
                 'Viet Nam': 'Viêt Nam',
                 'Chine - RAS de Macao': 'Macao',
                 'République démocratique populaire lao': 'Laos',
                 'Timor-Leste': 'Timor oriental',
                 'El Salvador': 'Salvador',
                 'Eswatini': 'Swaziland',
                 'Saint-Kitts-et-Nevis': 'Saint-Christophe-et-Niévès',
                 'Îles Falkland (Malvinas)': 'Îles Falkland',
                 'Chine - RAS de Hong-Kong': 'Hong Kong',
                 'République arabe syrienne': 'Syrie',
                 'Îles Wallis-et-Futuna': 'Wallis-et-Futuna',
                 'Antilles néerlandaises (ex)': 'Antilles néerlandaises',
                 'Bolivie (État plurinational de)': 'Bolivie',
                 'Tokélaou': 'Tokelau',
                 'Saint-Siège': 'Cité du Vatican',
                 'Bélarus': 'Biélorussie',
                 'Chine, continentale': 'Chine',
                 'Soudan du Sud': 'Soudan',
                 "Royaume-Uni de Grande-Bretagne et d'Irlande du Nord": 'Royaume-Uni',
                 'Îles Anglo-Normandes': 'Îles Normandes',
                 'Îles Vierges américaines': 'Îles Vierges des États-Unis',
                 "États-Unis d'Amérique": 'États-Unis',
                 'Îles Caïmanes': 'Îles Caïmans',
                 'Émirats arabes unis': 'Émirats Arabes Unis'}, inplace=True)

# Liste des pays absents dans distance mais présents dans pib
pays_absents_pib = {'République de Corée', 'Sao Tomé-et-Principe', 'République-Unie de Tanzanie', 'Cabo Verde', 'Fédération de Russie', 'Sint Maarten (partie néerlandaise)', 'Brunéi Darussalam', 'Myanmar', 'Nigéria', 'République de Moldova', 'Micronésie (États fédérés de)', 'Tchéquie', "Iran (République islamique d')", 'Venezuela (République bolivarienne du)', 'République populaire démocratique de Corée', 'Curaçao', 'Iraq', 'Viet Nam', 'Chine - RAS de Macao', 'République démocratique populaire lao', 'Timor-Leste', 'El Salvador', 'Eswatini', 'Saint-Kitts-et-Nevis', 'Chine - RAS de Hong-Kong', 'République arabe syrienne', 'Bolivie (État plurinational de)', 'Bélarus', 'Chine, continentale', 'Soudan du Sud', "Royaume-Uni de Grande-Bretagne et d'Irlande du Nord", "États-Unis d'Amérique", 'Îles Caïmanes', 'Émirats arabes unis'}

# Liste des pays absents dans distance mais présents dans stabilité
pays_absents_stabilite = {'Dominica', 'Cayman Islands', 'Low & middle income', 'Turkiye', 'IDA & IBRD total', 'Faroe Islands', 'Pre-demographic dividend', 'Lithuania', 'Curacao', 'China', 'French Polynesia', 'Albania', 'IDA blend', 'Chile', 'Greenland', 'Kyrgyz Republic', 'St. Kitts and Nevis', 'Romania', 'Turkmenistan', 'Upper middle income', 'Brunei Darussalam', 'St. Martin (French part)', 'IDA total', 'Europe & Central Asia', 'Eswatini', 'Cameroon', 'Belgium', 'Australia', 'IBRD only', 'Finland', 'Turks and Caicos Islands', 'Barbados', 'Armenia', 'Mexico', 'Azerbaijan', 'Bulgaria', 'Sweden', 'Congo, Rep.', 'Czechia', 'New Caledonia', 'Africa Eastern and Southern', 'Uganda', 'Peru', "Korea, Dem. People's Rep.", 'Fiji', 'Cabo Verde', 'Morocco', 'South Sudan', 'Hong Kong SAR, China', 'Estonia', 'Northern Mariana Islands', 'Euro area', 'Early-demographic dividend', 'Senegal', 'Late-demographic dividend', 'Switzerland', 'Liberia', 'Iceland', 'Thailand', 'Bermuda', 'Malta', 'Italy', 'Marshall Islands', 'Equatorial Guinea', 'Korea, Rep.', 'Mongolia', 'Sub-Saharan Africa', 'Puerto Rico', 'Latin America & Caribbean', 'North Macedonia', 'Jordan', 'New Zealand', 'Malaysia', 'IDA only', 'Poland', 'Benin', 'Tunisia', 'Fragile and conflict affected situations', 'Namibia', 'Slovak Republic', 'Bolivia', 'Cambodia', 'Egypt, Arab Rep.', 'United States', 'Macao SAR, China', 'Saudi Arabia', 'Isle of Man', 'Virgin Islands (U.S.)', 'Spain', 'Sub-Saharan Africa (IDA & IBRD countries)', 'Kuwait', 'Ecuador', 'Haiti', 'Small states', 'Libya', 'Least developed countries: UN classification', 'Myanmar', 'Syrian Arab Republic', 'British Virgin Islands', 'India', 'Solomon Islands', 'Central African Republic', 'Algeria', 'East Asia & Pacific', 'South Africa', 'Guinea-Bissau', 'Russian Federation', 'Colombia', 'Sudan', 'Dominican Republic', 'United Kingdom', 'Moldova', 'Palau', 'West Bank and Gaza', 'World', 'Uzbekistan', 'Mauritania', 'Grenada', 'Middle East & North Africa (excluding high income)', 'Tanzania', 'Bosnia and Herzegovina', 'Timor-Leste', 'El Salvador', 'Brazil', 'Guinea', 'Somalia', 'European Union', 'Sub-Saharan Africa (excluding high income)', 'Croatia', 'Georgia', "Cote d'Ivoire", 'Papua New Guinea', 'Slovenia', 'Venezuela, RB', 'South Asia', 'Bhutan', 'Africa Western and Central', 'Ethiopia', 'Belarus', 'Norway', 'Iran, Islamic Rep.', 'Israel', 'Heavily indebted poor countries (HIPC)', 'Channel Islands', 'Austria', 'East Asia & Pacific (excluding high income)', 'High income', 'Middle East & North Africa', 'Serbia', 'Bahrain', 'St. Vincent and the Grenadines', 'Sao Tome and Principe', 'Latin America & the Caribbean (IDA & IBRD countries)', 'Not classified', 'Nepal', 'Micronesia, Fed. Sts.', 'St. Lucia', 'Middle East & North Africa (IDA & IBRD countries)', 'Eritrea', 'Central Europe and the Baltics', 'Greece', 'Comoros', 'Singapore', 'Jamaica', 'Zambia', 'Antigua and Barbuda', 'Montenegro', 'Trinidad and Tobago', 'Sint Maarten (Dutch part)', 'South Asia (IDA & IBRD)', 'United Arab Emirates', 'Iraq', 'Denmark', 'Europe & Central Asia (IDA & IBRD countries)', 'Hungary', 'Andorra', 'Lebanon', 'Viet Nam', 'Caribbean small states', 'Pacific island small states', 'Argentina', 'Other small states', 'Germany', 'San Marino', 'Lower middle income', 'Post-demographic dividend', 'Cyprus', 'Arab World', 'Congo, Dem. Rep.', 'Gambia, The', 'Indonesia', 'North America', 'East Asia & Pacific (IDA & IBRD countries)', 'Low income', 'Middle income', 'Yemen, Rep.', 'OECD members', 'Tajikistan', 'Japan', 'Latvia', 'American Samoa', 'Bahamas, The', 'Lao PDR', 'Europe & Central Asia (excluding high income)', 'Latin America & Caribbean (excluding high income)', 'Mauritius', 'Ireland', 'Chad', 'Netherlands'}

# Remplacer les noms de pays dans pib et stabilité
pib.replace({
    'République de Corée': 'Corée du Sud',
    'Sao Tomé-et-Principe': 'Sao Tomé-et-Príncipe',
    'République-Unie de Tanzanie': 'Tanzanie',
    'Cabo Verde': 'Cap-Vert',
    'Fédération de Russie': 'Russie',
    'Sint Maarten (partie néerlandaise)': 'Sint Maarten',
    'Brunéi Darussalam': 'Brunei',
    'Myanmar': 'Birmanie',
    'Nigéria': 'Nigeria',
    'République de Moldova': 'Moldavie',
    'Micronésie (États fédérés de)': 'Micronésie',
    'Chine, Taiwan Province de': 'Taïwan',
    'Tchéquie': 'République tchèque',
    'Iran (République islamique d\')': 'Iran',
    'Venezuela (République bolivarienne du)': 'Venezuela',
    'République populaire démocratique de Corée': 'Corée du Nord',
    'Curaçao': 'Curaçao',
    'Iraq': 'Irak',
    'Viet Nam': 'Viêt Nam',
    'Chine - RAS de Macao': 'Macao',
    'République démocratique populaire lao': 'Laos',
    'Timor-Leste': 'Timor oriental',
    'El Salvador': 'Salvador',
    'Eswatini': 'Swaziland',
    'Saint-Kitts-et-Nevis': 'Saint-Christophe-et-Niévès',
    'Chine - RAS de Hong-Kong': 'Hong Kong',
    'République arabe syrienne': 'Syrie',
    'Bolivie (État plurinational de)': 'Bolivie',
    'Bélarus': 'Biélorussie',
    'Chine, continentale': 'Chine',
    'Soudan du Sud': 'Soudan',
    'Royaume-Uni de Grande-Bretagne et d\'Irlande du Nord': 'Royaume-Uni',
    'États-Unis d\'Amérique': 'États-Unis',
    'Îles Caïmanes': 'Îles Caïmans',
    'Émirats arabes unis': 'Émirats Arabes Unis'
}, inplace=True)

stabilite.replace({
    'République de Corée': 'Corée du Sud',
    'Sao Tomé-et-Principe': 'Sao Tomé-et-Príncipe',
    'République-Unie de Tanzanie': 'Tanzanie',
    'Cabo Verde': 'Cap-Vert',
    'Fédération de Russie': 'Russie',
    'Saint-Martin (partie française)': 'Saint-Martin',
    'Brunéi Darussalam': 'Brunei',
    'Myanmar': 'Birmanie',
    'Nigéria': 'Nigeria',
    'Serbie-et-Monténégro': 'Serbie',
    'République de Moldova': 'Moldavie',
    'Micronésie (États fédérés de)': 'Micronésie',
    'Chine, Taiwan Province de': 'Taïwan',
    'Tchéquie': 'République tchèque',
    'Sint Maarten  (partie néerlandaise)': 'Sint Maarten',
    'Bonaire, Saint-Eustache et Saba': 'Bonaire',
    'Iran (République islamique d\')': 'Iran',
    'Nioué': 'Niue',
    'Sainte-Hélène, Ascension et Tristan da Cunha': 'Sainte-Hélène',
    'Saint-Barthélemy': 'Saint-Barthélémy',
    'Soudan (ex)': 'Soudan',
    'Venezuela (République bolivarienne du)': 'Venezuela',
    'République populaire démocratique de Corée': 'Corée du Nord',
    'Curaçao': 'Curaçao',
    'Iraq': 'Irak',
    'Viet Nam': 'Viêt Nam',
    'Chine - RAS de Macao': 'Macao',
    'République démocratique populaire lao': 'Laos',
    'Timor-Leste': 'Timor oriental',
    'El Salvador': 'Salvador',
    'Eswatini': 'Swaziland',
    'Saint-Kitts-et-Nevis': 'Saint-Christophe-et-Niévès',
    'Îles Falkland (Malvinas)': 'Îles Falkland',
    'Chine - RAS de Hong-Kong': 'Hong Kong',
    'République arabe syrienne': 'Syrie',
    'Îles Wallis-et-Futuna': 'Wallis-et-Futuna',
    'Antilles néerlandaises (ex)': 'Antilles néerlandaises',
    'Bolivie (État plurinational de)': 'Bolivie',
    'Tokélaou': 'Tokelau',
    'Saint-Siège': 'Cité du Vatican',
    'Bélarus': 'Biélorussie',
    'Chine, continentale': 'Chine',
    'Soudan du Sud': 'Soudan',
    'Royaume-Uni de Grande-Bretagne et d\'Irlande du Nord': 'Royaume-Uni',
    'Îles Anglo-Normandes': 'Îles Normandes',
    'Îles Vierges américaines': 'Îles Vierges des États-Unis',
    'États-Unis d\'Amérique': 'États-Unis',
    'Îles Caïmanes': 'Îles Caïmans',
    'Émirats arabes unis': 'Émirats Arabes Unis'
}, inplace=True)


# # STABILITE POLITIQUE

# In[11]:


stability = stabilite.copy()

# Supprimer les colonnes de 1960 à 2018 inclusivement
stability = stabilite.drop(columns=[str(year) for year in range(1960, 2017)])
stability = stability.drop(columns=[str(year) for year in range(2018, 2023)])
# Supprimer les colonnes "Country Code", "Indicator Code" et "Unnamed"
stability = stability.drop(columns=["Country Code", "Indicator Code", "Unnamed: 67"])

display(stability)


# In[12]:


# Vérifier les valeurs NaN pour les années 2019, 2020, 2021 et 2022 en même temps
nan_rows = stability[stability[['2017']].isna().all(axis=1)]

# Afficher les lignes avec des valeurs NaN pour toutes les années
display(nan_rows)

# Supprimer les lignes avec des valeurs NaN pour toutes les années
stability.dropna(subset=['2017'], how='all', inplace=True)

# Afficher les lignes avec des valeurs NaN pour toutes les années
display(stability)


# In[13]:


# Vérifier les valeurs uniques
stability['pays'].unique()


# In[14]:


# Dictionnaire de correspondance entre les noms de pays dans le DataFrame de stabilité et ceux du DataFrame de distance
correspondance_pays_stabilite = {
    'Cabo Verde': 'Cap-Vert',
    'Congo, Dem. Rep.': 'République démocratique du Congo',
    'Congo, Rep.': 'Congo',
    'Czechia': 'République tchèque',
    'Egypt, Arab Rep.': 'Égypte',
    'Iran, Islamic Rep.': 'Iran',
    'Irak': 'Iraq',
    'Korea, Dem. People\'s Rep.': 'Corée du Nord',
    'Korea, Rep.': 'Corée du Sud',
    'Macao SAR, China': 'Macao',
    'Micronesia, Fed. Sts.': 'Micronésie',
    'Myanmar': 'Birmanie',
    'Russian Federation': 'Russie',
    'Sao Tome and Principe': 'Sao Tomé-et-Principe',
    'Slovak Republic': 'Slovaquie',
    'St. Kitts and Nevis': 'Saint-Christophe-et-Niévès',
    'St. Vincent and the Grenadines': 'Saint-Vincent-et-les Grenadines',
    'Syrian Arab Republic': 'Syrie',
    'Timor oriental': 'Timor-Leste',
    'Turkmenistan': 'Turkménistan',
    'United States': 'États-Unis',
    'Vanuatu': 'Vanuatu',
    'Viêt Nam': 'Vietnam',
    'Yemen, Rep.':'Yémen'
}

# Adapter les noms de pays dans le DataFrame de stabilité
stability['pays'] = stability['pays'].replace(correspondance_pays_stabilite)


# In[15]:


correspondance_stabilite = {
    'Aruba': 'Aruba',
    'Afghanistan': 'Afghanistan',
    'Angola': 'Angola',
    'Albania': 'Albanie',
    'Andorra': 'Andorre',
    'United Arab Emirates': 'Émirats arabes unis',
    'Argentina': 'Argentine',
    'Armenia': 'Arménie',
    'American Samoa': 'Samoa américaines',
    'Antigua and Barbuda': 'Antigua-et-Barbuda',
    'Australia': 'Australie',
    'Austria': 'Autriche',
    'Azerbaijan': 'Azerbaïdjan',
    'Burundi': 'Burundi',
    'Belgium': 'Belgique',
    'Benin': 'Bénin',
    'Burkina Faso': 'Burkina Faso',
    'Bangladesh': 'Bangladesh',
    'Bulgaria': 'Bulgarie',
    'Bahrain': 'Bahreïn',
    'Bahamas': 'Bahamas',
    'Bosnia and Herzegovina': 'Bosnie-Herzégovine',
    'Belarus': 'Biélorussie',
    'Belize': 'Belize',
    'Bermuda': 'Bermudes',
    'Bolivia': 'Bolivie',
    'Brazil': 'Brésil',
    'Barbados': 'Barbade',
    'Brunei Darussalam': 'Brunéi Darussalam',
    'Bhutan': 'Bhoutan',
    'Botswana': 'Botswana',
    'Central African Republic': 'République centrafricaine',
    'Canada': 'Canada',
    'Switzerland': 'Suisse',
    'Chile': 'Chili',
    'China': 'Chine',
    'Côte d\'Ivoire': 'Côte d\'Ivoire',
    'Cameroon': 'Cameroun',
    'République démocratique du Congo': 'République démocratique du Congo',
    'Congo': 'Congo',
    'Colombia': 'Colombie',
    'Comoros': 'Comores',
    'Cape Verde': 'Cap-Vert',
    'Costa Rica': 'Costa Rica',
    'Cuba': 'Cuba',
    'Cayman Islands': 'Îles Caïmans',
    'Cyprus': 'Chypre',
    'Czech Republic': 'République tchèque',
    'Germany': 'Allemagne',
    'Djibouti': 'Djibouti',
    'Dominica': 'Dominique',
    'Denmark': 'Danemark',
    'Dominican Republic': 'République dominicaine',
    'Algeria': 'Algérie',
    'Ecuador': 'Équateur',
    'Egypt': 'Égypte',
    'Eritrea': 'Érythrée',
    'Spain': 'Espagne',
    'Estonia': 'Estonie',
    'Ethiopia': 'Éthiopie',
    'Finland': 'Finlande',
    'Fiji': 'Fidji',
    'France': 'France',
    'Micronesia': 'Micronésie',
    'Gabon': 'Gabon',
    'United Kingdom': 'Royaume-Uni',
    'Georgia': 'Géorgie',
    'Ghana': 'Ghana',
    'Guinea': 'Guinée',
    'Gambia': 'Gambie',
    'Guinea-Bissau': 'Guinée-Bissau',
    'Equatorial Guinea': 'Guinée équatoriale',
    'Greece': 'Grèce',
    'Grenada': 'Grenade',
    'Greenland': 'Groenland',
    'Guatemala': 'Guatemala',
    'Guam': 'Guam',
    'Guyana': 'Guyana',
    'Hong Kong': 'Hong Kong',
    'Honduras': 'Honduras',
    'Croatia': 'Croatie',
    'Haiti': 'Haïti',
    'Hungary': 'Hongrie',
    'Indonesia': 'Indonésie',
    'India': 'Inde',
    'Ireland': 'Irlande',
    'Iran': 'Iran',
    'Iraq': 'Irak',
    'Iceland': 'Islande',
    'Israel': 'Israël',
    'Italy': 'Italie',
    'Jamaica': 'Jamaïque',
    'Jordan': 'Jordanie',
    'Japan': 'Japon',
    'Kazakhstan': 'Kazakhstan',
    'Kenya': 'Kenya',
    'Kyrgyzstan': 'Kirghizistan',
    'Cambodia': 'Cambodge',
    'Kiribati': 'Kiribati',
    'Saint Kitts and Nevis': 'Saint-Christophe-et-Niévès',
    'South Korea': 'Corée du Sud',
    'Kuwait': 'Koweït',
    'Laos': 'Laos',
    'Lebanon': 'Liban',
    'Liberia': 'Libéria',
    'Libya': 'Libye',
    'Saint Lucia': 'Sainte-Lucie',
    'Liechtenstein': 'Liechtenstein',
    'Sri Lanka': 'Sri Lanka',
    'Lesotho': 'Lesotho',
    'Lithuania': 'Lituanie',
    'Luxembourg': 'Luxembourg',
    'Latvia': 'Lettonie',
    'Macao': 'Macao',
    'Morocco': 'Maroc',
    'Monaco': 'Monaco',
    'Moldova': 'Moldavie',
    'Madagascar': 'Madagascar',
    'Maldives': 'Maldives',
    'Mexico': 'Mexique',
    'Marshall Islands': 'Îles Marshall',
    'North Macedonia': 'Macédoine du Nord',
    'Mali': 'Mali',
    'Malta': 'Malte',
    'Myanmar': 'Myanmar',
    'Montenegro': 'Monténégro',
    'Mongolia': 'Mongolie',
    'Mozambique': 'Mozambique',
    'Mauritania': 'Mauritanie',
    'Mauritius': 'Maurice',
    'Malawi': 'Malawi',
    'Malaysia': 'Malaisie',
    'Namibia': 'Namibie',
    'Niger': 'Niger',
    'Nigeria': 'Nigéria',
    'Nicaragua': 'Nicaragua',
    'Netherlands': 'Pays-Bas',
    'Norway': 'Norvège',
    'Nepal': 'Népal',
    'Nauru': 'Nauru',
    'New Zealand': 'Nouvelle-Zélande',
    'Oman': 'Oman',
    'Pakistan': 'Pakistan',
    'Panama': 'Panama',
    'Peru': 'Pérou',
    'Philippines': 'Philippines',
    'Palau': 'Palaos',
    'Papua New Guinea': 'Papouasie-Nouvelle-Guinée',
    'Poland': 'Pologne',
    'Puerto Rico': 'Porto Rico',
    'North Korea': 'Corée du Nord',
    'Portugal': 'Portugal',
    'Paraguay': 'Paraguay',
    'West Bank and Gaza': 'Territoires palestiniens',
    'Qatar': 'Qatar',
    'Romania': 'Roumanie',
    'Russia': 'Russie',
    'Rwanda': 'Rwanda',
    'Saudi Arabia': 'Arabie saoudite',
    'Sudan': 'Soudan',
    'Senegal': 'Sénégal',
    'Singapore': 'Singapour',
    'Solomon Islands': 'Îles Salomon',
    'Sierra Leone': 'Sierra Leone',
    'El Salvador': 'Salvador',
    'San Marino': 'Saint-Marin',
    'Somalia': 'Somalie',
    'Serbia': 'Serbie',
    'South Sudan': 'Soudan du Sud',
    'Sao Tome and Principe': 'Sao Tomé-et-Principe',
    'Suriname': 'Suriname',
    'Slovakia': 'Slovaquie',
    'Slovenia': 'Slovénie',
    'Sweden': 'Suède',
    'Eswatini': 'Eswatini',
    'Seychelles': 'Seychelles',
    'Syria': 'Syrie',
    'Chad': 'Tchad',
    'Togo': 'Togo',
    'Thailand': 'Thaïlande',
    'Tajikistan': 'Tadjikistan',
    'Turkmenistan': 'Turkménistan',
    'Timor-Leste': 'Timor oriental',
    'Tonga': 'Tonga',
    'Trinidad and Tobago': 'Trinité-et-Tobago',
    'Tunisia': 'Tunisie',
    'Turkey': 'Turquie',
    'Tuvalu': 'Tuvalu',
    'Tanzania': 'Tanzanie',
    'Uganda': 'Ouganda',
    'Ukraine': 'Ukraine',
    'Uruguay': 'Uruguay',
    'United States': 'États-Unis',
    'Uzbekistan': 'Ouzbékistan',
    'Saint Vincent and the Grenadines': 'Saint-Vincent-et-les Grenadines',
    'Venezuela, RB': 'Venezuela',
    'Virgin Islands (U.S.)': 'Îles Vierges des États-Unis',
    'Vietnam': 'Viêt Nam',
    'Vanuatu': 'Vanuatu',
    'Samoa': 'Samoa',
    'Kosovo': 'Kosovo',
    'Yemen': 'Yémen',
    'South Africa': 'Afrique du Sud',
    'Zambia': 'Zambie',
    'Zimbabwe': 'Zimbabwe'
}

stability['pays'] = stability['pays'].replace(correspondance_stabilite)


# In[16]:


correspondance_turquie = {
    'Turkiye': 'Turquie'
}

stability['pays'] = stability['pays'].replace(correspondance_turquie)


# In[17]:


stability['pays'].unique()


# In[18]:


# Vérifier les valeurs NaN pour les années 2019, 2020, 2021 et 2022 en même temps
nan_r = stability[stability[['2017']].isna().all(axis=1)]
nan_rr = stability[stability[['pays']].isna().all(axis=1)]
# Afficher les lignes avec des valeurs NaN pour toutes les années
display(nan_r)
display(nan_rr)

# Supprimer les lignes avec des valeurs NaN pour toutes les années
stability.dropna(subset=['2017'], how='all', inplace=True)

# Afficher les lignes avec des valeurs NaN pour toutes les années
display(stability)


# In[19]:


# Supprimer la colonne 'Indicator Name'
stability = stability.drop(columns=['Indicator Name'])

# Renommer la colonne '2017' en 'stabilite_politique_2017'
stability = stability.rename(columns={'2017': 'stabilite_politique'})

# Afficher le DataFrame mis à jour
display(stability)


# Le dataframe de stabilité politique est prêt et nettoyé, nous pouvons passer à la suite.

# # PIB

# In[20]:


#Afficher pib
display(pib)


# In[21]:


pib['pays'] = pib['pays'].str.lower()
pop0018['pays'] = pop0018['pays'].str.lower()
stability['pays'] = stability['pays'].str.lower()
distance['pays'] = distance['pays'].str.lower()
dispo2017['pays'] = dispo2017['pays'].str.lower()


# In[22]:


pib['pays'].unique()


# In[50]:


# Suppression des colonnes indésirables
pib = pib[['pays', 'Valeur']]

# Renommer la colonne 'Valeur' par 'PIB'
pib.rename(columns={"Valeur": "PIB_tête_$"}, inplace=True)

pib = round(pib, 2)

# Afficher le DataFrame après les modifications
display(pib)


# Le dataframe PIB est prêt.

# # DISTANCE

# In[24]:


# Remplacer les valeurs dans la variable 'pays' du dataframe distance
distance['pays'].replace({
    'bélarus': 'biélorussie',
    'côte d’ivoire': 'côte d’ivoire',
    'macédoine': 'macédoine du nord',
    'république démocratique du congo': 'congo'
}, inplace=True)


# In[25]:


# Remplacer "côte d’ivoire" par 'côte d\'ivoire' dans chaque DataFrame
distance['pays'] = distance['pays'].replace('côte d’ivoire', 'côte d\'ivoire')
pop0018['pays'] = pop0018['pays'].replace("côte d’ivoire", 'côte d\'ivoire')
dispo2017['pays'] = dispo2017['pays'].replace("côte d’ivoire", 'côte d\'ivoire')
pib['pays'] = pib['pays'].replace("côte d’ivoire", 'côte d\'ivoire')
stability['pays'] = stability['pays'].replace("côte d’ivoire", 'côte d\'ivoire')


# In[26]:


# Remplacer la valeur 'côte d’ivoire' par "côte d’ivoire"
distance['pays'].unique()


# In[38]:


# Vérifier les valeurs manquantes : pays
pays_là = distance['pays'].unique()
sim = dispo2017[~dispo2017.pays.isin(pays_là)]['pays'].unique()

# Afficher
print('Nombre de valeur manquantes :', len(sim))
print(sim)


# In[39]:


# Vérifier les valeurs manquantes : pays
pib_là = pib['pays'].unique()
sim2 = dispo2017[~dispo2017.pays.isin(pib_là)]['pays'].unique()

# Afficher
print('Nombre de valeur manquantes :', len(sim2))
print(sim2)


# In[40]:


# Correction des noms de pays dans stability pour correspondre à ceux dans dispo2017
stability['pays'] = stability['pays'].replace({
    'brunéi darussalam': 'brunei',
    'hong kong sar, china': 'hong kong',
    'taiwan': 'taïwan',
    "cote d'ivoire": "côte d'ivoire",
    'gambia, the': 'gambie',
    'kyrgyz republic': 'kirghizistan',
    'nigéria': 'nigeria',
    'lao pdr': 'laos',
    'st. lucia': 'sainte-lucie',
    'sao tomé-et-principe': 'sao tomé-et-príncipe'
})


stabilite_là = stability['pays'].unique()
sim3 = dispo2017[~dispo2017.pays.isin(stabilite_là)]['pays'].unique()
print('Nombre de valeur manquantes :', len(sim3))

#Afficher
sim3


# In[ ]:


stability = round(stability, 2)


# Le dataframe de stabilité est prêt.

# # POPULATION

# In[41]:


pop0018


# In[42]:


# Création d'une variable "évolution démographique"  et mise en forme du nouveau dataframe
popevo = pop0018[pop0018['Année'].isin([2010, 2017])][['pays', 'Valeur', 'Année']]
popevo = popevo.pivot_table(values='Valeur', index='pays', columns='Année')
popevo['evolution_demo'] = (popevo[2017] - popevo[2010]) / popevo[2017]
popevo = popevo.reset_index()
display(popevo)

pop = pop0018[pop0018['Année'] == 2017][['pays', 'Valeur']]
pop['population'] = pop['Valeur']*1000
pop = pop.drop(['Valeur'], axis=1)
display(pop)

pop_final = pd.merge(pop, popevo[['pays', 'evolution_demo']], how='inner')

# Afficher
display(pop_final)


# Le dataframe de population est prêt.

# # DISPONIBILITE ALIMENTAIRE :

# In[48]:


# Afficher
display(dispo2017)
dispo2017['pays'].unique()


# In[43]:


# Filtrage sur la viande de volailles
dispo = dispo2017[dispo2017['Produit'] == 'Viande de Volailles']
dispo = dispo[['pays', 'Élément', 'Valeur']]
display(dispo.head(3))
dispo_alim = dispo.pivot_table(values='Valeur', index='pays', columns='Élément')
display(dispo_alim.head(3))


# In[44]:


# Sélection des variables à conserver et affichage
dispo_alim = dispo_alim[['Importations - Quantité', 'Nourriture', 'Production']]
display(dispo_alim)


# Le dataframe de disponibilité alimentaire est prêt.

# # JOINTURES : DATAFRAME FINAL 

# In[51]:


display(pib)
display(stability)
display(distance)
display(dispo_alim)
display(pop_final)


# In[52]:


# Jointures des différents dataframes
df = pd.merge(dispo_alim, pop_final, on='pays', how='left')
df = pd.merge(df, distance, on='pays', how='left')
df = pd.merge(df, pib, on='pays', how='left')
chicken = pd.merge(df, stability, on='pays', how='left')
display(chicken)


# # NETTOYAGE FINAL 

# In[70]:


# Vérification des NaN
chicken.isnull().sum()


# In[71]:


# Localisation des NaN
chicken.loc[chicken['distance'].isnull()]


# In[55]:


# Attribuer 0 à la distance de la France
chicken.loc[chicken['pays'] == 'france', 'distance'] = 0

# Définir les pays concernés
pays_concernes = ['birmanie', 'macao', 'saint-christophe-et-niévès', 'saint-vincent-et-les grenadines', 'sainte-lucie', 'salvador', 'sao tomé-et-príncipe']

# Définir les distances spécifiques pour chaque pays
distances_specifiques = {
    'birmanie': 8500,
    'macao': 9500,
    'saint-christophe-et-niévès': 7000,
    'saint-vincent-et-les grenadines': 6500,
    'sainte-lucie': 6500,
    'salvador': 9000,
    'sao tomé-et-príncipe': 5000
}

# Remplacer les valeurs manquantes par les distances spécifiques
for pays, distance in distances_specifiques.items():
    chicken.loc[chicken['pays'] == pays, 'distance'] = distance

# Afficher les données mises à jour
print(chicken.loc[chicken['pays'].isin(pays_concernes), ['pays', 'distance']])


# In[56]:


# On rajoute à toutes les autres des valeurs médianes 
chicken.loc[chicken['Importations - Quantité'].isnull(), 'Importations - Quantité'] = chicken['Importations - Quantité'].median()
chicken.loc[chicken['Nourriture'].isnull(), 'Nourriture'] = chicken['Nourriture'].median()
chicken.loc[chicken['Production'].isnull(), 'Production'] = chicken['Production'].median()
chicken.loc[chicken['PIB_tête_$'].isnull(), 'PIB_tête_$'] = chicken['PIB_tête_$'].median()

# Mise en place de ratios
chicken['Importations - Quantité'] = chicken['Importations - Quantité']/chicken['population']
chicken['Production'] = chicken['Production']/chicken['population']
chicken['Nourriture'] = chicken['Nourriture']/chicken['population']

#Afficher le rendu
display(chicken)


# In[58]:


chicken.loc[chicken['stabilite_politique'].isnull()]


# In[59]:


# Calculer la stabilité politique moyenne des pays de référence
stabilite_us = chicken.loc[chicken['pays'] == 'états-unis', 'stabilite_politique'].mean()
stabilite_france = chicken.loc[chicken['pays'] == 'france', 'stabilite_politique'].mean()
stabilite_chine = chicken.loc[chicken['pays'] == 'chine', 'stabilite_politique'].mean()

# Remplacer les valeurs NaN par les estimations basées sur les pays de référence
chicken.loc[chicken['pays'] == 'bahamas', 'stabilite_politique'] = stabilite_us
chicken.loc[chicken['pays'].isin(['nouvelle-calédonie', 'polynésie française']), 'stabilite_politique'] = stabilite_france
chicken.loc[chicken['pays'] == 'taïwan', 'stabilite_politique'] = stabilite_chine

# Afficher les données mises à jour
print(chicken.loc[chicken['pays'].isin(['bahamas', 'nouvelle-calédonie', 'polynésie française', 'taïwan']), ['pays', 'stabilite_politique']])


# In[60]:


# Vérification si plus de NaN
chicken.isnull().sum()


# In[63]:


# Vérification des doublons
doublons = chicken[chicken.duplicated(subset=['pays'], keep=False)]
display(doublons)

# Supprimer les doublons basés sur la colonne 'pays'
chicken.drop_duplicates(subset=['pays'], inplace=True)

# Afficher les données mises à jour
display(chicken[chicken['pays'] == 'soudan'])
display(chicken[chicken['pays'] == 'chine'])


# In[66]:


# Copie et Afficher le dataframe final
datachicken = chicken.copy()

# Renommer les colonnes
datachicken.rename(columns={
    'Importations - Quantité': 'importation_qt',
    'Nourriture': 'nourriture_mille_to',
    'Production': 'production_mille_to'
}, inplace=True)

display(datachicken)


# # OBSERVATIONS GRAPHIQUES (PRE-ANALYSE EXPLORATOIRE)

# In[75]:


# Sélection des 30 premiers pays par importation
top_30_importations = datachicken.nlargest(30, 'importation_qt')

# Création de l'histogramme
plt.figure(figsize=(15, 8))
plt.bar(top_30_importations['pays'], top_30_importations['importation_qt'], color='skyblue')
plt.title('Top 30 des pays par importation')
plt.xlabel('Pays')
plt.ylabel('Importation')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[81]:


# Sélection des 30 premiers pays par PIB par tête
top_30_pib_tete = datachicken.nlargest(30, 'PIB_tête_$')

# Création de l'histogramme
plt.figure(figsize=(15, 8))
plt.bar(top_30_pib_tete['pays'], top_30_pib_tete['PIB_tête_$'], color='skyblue')
plt.title('Top 30 des pays par PIB par tête')
plt.xlabel('Pays')
plt.ylabel('PIB par tête ($)')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[80]:


# Sélection des 30 premiers pays par stabilité politique
top_30_stabilite_politique = datachicken.nlargest(30, 'stabilite_politique')

# Création de l'histogramme
plt.figure(figsize=(15, 8))
plt.bar(top_30_stabilite_politique['pays'], top_30_stabilite_politique['stabilite_politique'], color='lightgreen')
plt.title('Top 30 des pays par stabilité politique')
plt.xlabel('Pays')
plt.ylabel('Stabilité politique')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[78]:


# Sélection des 30 premiers pays par population
top_30_population = datachicken.nlargest(30, 'population')

# Création de l'histogramme
plt.figure(figsize=(15, 8))
plt.bar(top_30_population['pays'], top_30_population['population'], color='lightgreen')
plt.title('Top 30 des pays par population')
plt.xlabel('Pays')
plt.ylabel('Population')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[79]:


# Sélection des 30 premiers pays par évolution démographique
top_30_evolution_demographique = datachicken.nlargest(30, 'evolution_demo')

# Création de l'histogramme
plt.figure(figsize=(15, 8))
plt.bar(top_30_evolution_demographique['pays'], top_30_evolution_demographique['evolution_demo'], color='skyblue')
plt.title('Top 30 des pays par évolution démographique')
plt.xlabel('Pays')
plt.ylabel('Évolution démographique')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[76]:


# Sélection des 30 premiers pays par consommation de nourriture
top_30_nourriture = datachicken.nlargest(30, 'nourriture_mille_to')

# Création de l'histogramme
plt.figure(figsize=(15, 8))
plt.bar(top_30_nourriture['pays'], top_30_nourriture['nourriture_mille_to'], color='lightgreen')
plt.title('Top 30 des pays par consommation de nourriture')
plt.xlabel('Pays')
plt.ylabel('Consommation de nourriture (mille tonnes)')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[77]:


# Sélection des 30 premiers pays par production de nourriture
top_30_production = datachicken.nlargest(30, 'production_mille_to')

# Création de l'histogramme
plt.figure(figsize=(15, 8))
plt.bar(top_30_production['pays'], top_30_production['production_mille_to'], color='lightblue')
plt.title('Top 30 des pays par production de nourriture')
plt.xlabel('Pays')
plt.ylabel('Production de nourriture (mille tonnes)')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[83]:


# Sélection des 20 pays les plus éloignés et des 20 pays les moins éloignés
top_20_loin = datachicken.nlargest(20, 'distance')
top_20_proche = datachicken.nsmallest(20, 'distance')

# Création de la figure et des sous-graphiques
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Histogramme des 20 pays les plus éloignés
axes[0].bar(top_20_loin['pays'], top_20_loin['distance'], color='orange')
axes[0].set_title('Top 20 des pays les plus éloignés')
axes[0].set_xlabel('Pays')
axes[0].set_ylabel('Distance')
axes[0].tick_params(axis='x', rotation=90)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Histogramme des 20 pays les moins éloignés
axes[1].bar(top_20_proche['pays'], top_20_proche['distance'], color='green')
axes[1].set_title('Top 20 des pays les moins éloignés')
axes[1].set_xlabel('Pays')
axes[1].set_ylabel('Distance')
axes[1].tick_params(axis='x', rotation=90)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# Top 5 des pays avec les plus grandes importations :
# Hong Kong, Samoa, Saint-Vincent-et-les Grenadines, Saint-Christophe-et-Niévès, Antigua-et-Barbuda
# 
# Top 5 des pays avec la plus grande quantité de nourriture consommée par habitant :
# Saint-Vincent-et-les Grenadines, Israël, Samoa, Saint-Christophe-et-Niévès, États-Unis
# 
# Top 5 des pays avec la plus grande production de nourriture par habitant :
# Maldives, Israël, Djibouti, Brésil, États-Unis
# 
# Top 5 des pays avec la plus grande population :
# Chine, Inde, États-Unis, Indonésie, Pakistan
# 
# Top 5 des pays avec la plus grande distance :
# Nouvelle-Zélande, Australie, Nouvelle-Calédonie, Fidji, Vanuatu
# 
# Top 5 des pays avec le PIB par habitant le plus élevé :
# Luxembourg, Suisse, Macao, Norvège, Islande
# 
# Top 5 des pays avec la plus grande stabilité politique :
# Nouvelle-Zélande, Macao, Islande, Luxembourg, Malte
# 
# Top 5 des pays avec la plus grande évolution démographique :
# Oman, Liban, Maldives, Koweït, Jordanie

# In[91]:


# Définition des sous-plots
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

# Importations de poulet par pays
sns.boxplot(x=datachicken['importation_qt'], ax=axs[0, 0], orient='v', fliersize=5)
axs[0, 0].set_title('Importations de poulet')

# Nourriture consommée par habitant
sns.boxplot(x=datachicken['nourriture_mille_to'], ax=axs[0, 1], orient='v', fliersize=5)
axs[0, 1].set_title('Nourriture consommée par habitant')

# Production de nourriture par habitant
sns.boxplot(x=datachicken['production_mille_to'], ax=axs[0, 2], orient='v', fliersize=5)
axs[0, 2].set_title('Production de nourriture par habitant')

# Population par pays
sns.boxplot(x=datachicken['population'], ax=axs[0, 3], orient='v', fliersize=5)
axs[0, 3].set_title('Population')

# Évolution démographique
sns.boxplot(x=datachicken['evolution_demo'], ax=axs[1, 0], orient='v', fliersize=5)
axs[1, 0].set_title('Évolution démographique')

# Distance
sns.boxplot(x=datachicken['distance'], ax=axs[1, 1], orient='v', fliersize=5)
axs[1, 1].set_title('Distance')

# PIB par tête
sns.boxplot(x=datachicken['PIB_tête_$'], ax=axs[1, 2], orient='v', fliersize=5)
axs[1, 2].set_title('PIB par tête')

# Stabilité politique
sns.boxplot(x=datachicken['stabilite_politique'], ax=axs[1, 3], orient='v', fliersize=5)
axs[1, 3].set_title('Stabilité politique')

# Ajustement de l'espacement entre les sous-plots
plt.tight_layout()

# Affichage
plt.show()


# Répartition des données de pays pour chaque variables : les outliers présents de manière conséquente dans importations, nourriture, production, population et PIB.

# In[92]:


df_corr = datachicken.corr()

# Afficher la matrice de corrélation
print("Matrice de corrélation :")
print(df_corr)

# Afficher un heatmap des corrélations
plt.figure(figsize=(10, 8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matrice de corrélation')
plt.show()


# Les données de la variables d’importation sont partiellement corrélées avec une forte consommation de nourriture, un bon PIB par haitant, une bonne stabilité politique.
# 
# A l’inverse, une forte population est corrélé négativement avec de fortes importations, idem pour la variable de production et de population.

# # EXPORT DU DATAFRAME ET CONCLUSION

# In[68]:


# Exporter au format excel
datachicken.to_excel('C:\\Users\\djaba\\OneDrive\\Documents\\P9\\datachicken.xlsx', index=False)


# Le nettoyage et les jointures ont été faites, des visualisations graphiques sur les statistiques des pays par variable a également été réalisée afin d'avoir une vue d'ensemble ; nous sommes prêts à attaquer la partie finale : analyse exploratoire (cah, kmeans, acp).
