import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ================================  Carga del Dataset  ===================================

df = pd.read_csv('data/coffee_consumption.csv')

# descripción del dataset
df.head()

df.info()

df.describe()

#comprobar si hay datos nulos
print('Valores nulos en el Dataset:\n', df.isnull().sum())

#comprobando si hay datos duplicados
print('Valores duplicados en el Dataset', df.duplicated().sum())

# ================================  Limpieza del Dataset  ===================================

# Elimino las columnas 'Health_Issues'y 'Heart_Rate'
df = df.drop(['Health_Issues', 'Heart_Rate'], axis = 1)


# Añado una columna (Sleep_Quality_num) al Dataset para representar la calidad del sueño con valores numericos
mapeo_sleep = {
    'Poor': 1,
    'Fair': 2, 
    'Good': 3,
    'Excellent': 4
}
# uso un try/except para comprobar si ya existe esta columna y no crearla dos veces
try:
    _ = df['Sleep_Quality_num']
    print("Columna 'Sleep_Quality_num' ya existe!")
except KeyError:
    print("✅ Columna 'Sleep_Quality_num' creada exitosamente!")
    df.insert(8, 'Sleep_Quality_num', df['Sleep_Quality'].map(mapeo_sleep))

# Verifico como se ve en la tabla la nueva columna
df[['Sleep_Quality', 'Sleep_Quality_num']].head()


# Añado una nueva columna (Stress_Level_num) al Dataset para transformar los valores de tipo objeto del nivel de estres (Low, Medium, High) a valores numericos 
mapeo_stress = {
    'Low': 1,
    'Medium': 2, 
    'High': 3
}

# uso un try/except para comprobar si ya existe esta columna y no crearla dos veces
try:
    _ = df['Stress_Level_num']
    print("Columna 'Stress_Level_num' ya existe!")
except KeyError:
    print("✅ Columna 'Stress_Level_num' creada exitosamente!")
    df.insert(12, 'Stress_Level_num', df['Stress_Level'].map(mapeo_stress))
    
# Verifico como se ve en la tabla la nueva columna
df[['Stress_Level', 'Stress_Level_num']].head()

# Añado otra columna de clasificación para agrupar las personas en grupos de edad: Jovenes, Adultos y Mayores
df['Age_Group'] = pd.cut(df['Age'], bins=[17, 35 ,55, 80], labels=['Joven','Adulto', 'Mayor'])

# Definir los rangos de IMC y sus categorías
bins = [0, 18.5, 25, 30, 100]  # Límites de IMC
labels = ['Bajo peso', 'Peso saludable', 'Sobrepeso', 'Obesidad']  # Categorías

# Crear otra nueva columna para agrupar las personas segun su IMC
try:
    df['Categoria_BMI'] = pd.cut(df['BMI'], bins=bins, labels=labels, right=False)
    print('La columna Categoria_BMI creada exitosamente')
except:
    print('La columna ategoria_BMI ya existe')

# ================================  Guardado del Dataset limpio  ===================================

df.to_csv('data/cleaned_coffee_consumption.csv', index = False)

# ==================================  Variables globales  ======================================

colores_cafe = [ '#1A0F08', '#25150C', '#301B10', '#3B2114', '#462718',
    '#512D1C', '#5C3320', '#673924', '#723F28', '#7D452C',
    '#884B30', '#935134', '#9E5738', '#A95D3C', '#B46340',
    '#BF6944', '#CA6F48', '#D5754C', '#E07B50', '#EB8154'
]

# ===============================  ANÁLISIS EXPLORATORIO DE DATOS  ==================================

df = pd.read_csv('data/cleaned_coffee_consumption.csv')
df.head()

# Antes del todo vamos a ver cuantas personas consumen cafe y cuantas no con un grafico de tarta

no_cons = df[df['Coffee_Intake']== 0]['ID'].count()
si_cons = df['ID'].count()

# Gráfico de tarta para mostrar el porcentaje de las personan que no consuman cafe y de los que consuman

plt.figure(figsize=(8,8))
explode = (0.05, 0)
plt.pie([si_cons, no_cons],
        labels=['Consumidores', 'No consumidores'],
        autopct='%1.2f%%',
        colors = [colores_cafe[8], colores_cafe[18],],
        explode=explode,)
p=plt.gcf()
plt.title('¿Cuantas personas toman cafe cada día?')
plt.savefig('img/grafico_analisis1.png')
plt.close()

# Gráfico de tipo tarta para mostrar el porcentaje de las personan que se toman 1, 2, 3, 4, entre 5 y 6 y 7 o más tazas de cafe al dia
tazas_dicc = {
        '1 taza': df[df['Coffee_Intake'] == 1]['ID'].count(),
        '2 tazas': df[df['Coffee_Intake'] == 2]['ID'].count(),
        '3 tazas': df[df['Coffee_Intake']==3]['ID'].count(),
        '4 tazas': df[df['Coffee_Intake']==4]['ID'].count(),
        '5-6 tazas': df[df['Coffee_Intake'].between(5, 6)]['ID'].count(),
        '7 y más': df[df['Coffee_Intake'] >= 7]['ID'].count()
}
plt.figure(figsize=(8,8))
plt.pie(tazas_dicc.values(),
        labels=tazas_dicc.keys(),
        autopct='%1.2f%%',
        colors = [colores_cafe[4], colores_cafe[8], colores_cafe[12], colores_cafe[14], colores_cafe[16], colores_cafe[19]])
p=plt.gcf()
plt.title('¿Cuantas tazas de café se toman al día?')
plt.savefig('img/grafico_analisis2.png')
plt.close()

# ===============================  HÍPOTESIS 1  ==================================
# H1: Italia y Brazil son los paises donde más café se consume.

# sacando la media de los miligramos de café por cada taza (95mg)
print(df['Caffeine_mg'].sum() / df['Coffee_Intake'].sum())

# crea un dataframe para la primera hipotesis con los mg de café que se consumen de promedia en cada pais, ordenados descendentemente
df_h1=df.groupby('Country', as_index=False).mean('Caffeine_mg').round(2)[['Country','Caffeine_mg']].sort_values(by='Caffeine_mg', ascending=False)


# Crear el gráfico de barras horizontales para los paises mostrando el consumo del café en mg

plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df_h1, y='Country', x='Caffeine_mg', palette=colores_cafe)

for container in ax.containers:
    for bar in container:
        width = bar.get_width()
        ax.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
                f'{width:.0f} mg', 
                ha='left', va='center', 
                fontweight='bold', fontsize=10)

plt.title('Consumo medio de cafeína por país',fontweight='bold', fontsize=14)
plt.xlabel('Cafeína (mg)')
plt.ylabel('País')
plt.xlim(220, 250)
plt.tight_layout()
plt.savefig('img/grafico_h1.png')
plt.close()

# ===============================  HÍPOTESIS 2  ==================================
# H2: Entre el consumo de cafeína y la calidad del sueño existe una correlación negativa.

# mostrar las clasificaciónes de la calidad del sueño
df['Sleep_Quality'].value_counts()

# crear un dataframe con la media de las horas dormidas, y de los mg de cafeina consumidas por cada valor de Sleep_Quality, 
# para poder visualizar si de verdad cuanto más cafeina se toma, menos horas se durme y peor es el sueño
df_h2 = df.groupby('Sleep_Quality').mean('Sleep_Hours').round(2).sort_values('Sleep_Hours', ascending=False)[['Sleep_Hours', 'Coffee_Intake']]


# Gráfico para la representar la correlación negativa entre la calidad del sueño y los mg de cafeina consumidos

plt.figure(figsize=(8, 4))
ax = sns.barplot(data=df_h2, y='Coffee_Intake', x='Sleep_Hours',hue=df_h2.index, hue_order=['Poor', 'Fair', 'Good', 'Excellent'])

plt.title('Calidad del sueño por grupo de edad',fontweight='bold', fontsize=14)
plt.xlabel('Horas de sueño')
plt.ylabel('mg de cafeina')
plt.tight_layout()
plt.savefig('img/grafico_h2a.png')
plt.close()


# Gráfico con la diagrama de dispersión para entender como se relacionan las horas del sueño con la cantidad del café consumido y como se distribuen individualmente
plt.figure(figsize=(8, 4))
sns.jointplot(x=df['Sleep_Hours'],
              y=df['Coffee_Intake'],
              kind="hex",
              color=colores_cafe[15]);
plt.savefig('img/grafico_h2b.png')
plt.close()

# ===============================  HÍPOTESIS 3  ==================================
# H3: Hay diferencias significativas entre el consumo del café, nivel del estrés y calidad del sueño por genero.

# Gráfico para mostrar la corelación entre el consumo de cafeína y el nivel del estres por genero
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.kdeplot(data=df[df['Gender'] == 'Male'], 
            x='Coffee_Intake', y='Stress_Level_num',
            fill=True, cmap='Blues', alpha=0.7,
            ax=ax1)
ax1.set_yticks([1, 2, 3])
ax1.set_yticklabels(['Bajo', 'Medio', 'Alto'])
ax1.set_title('Hombres', fontsize=12, fontweight='bold')
ax1.set_xlabel('mg de cafeina')
ax1.set_ylabel('Nivel del estres')
ax1.grid(True, alpha=0.3)

sns.kdeplot(data=df[df['Gender'] == 'Female'], 
            x='Coffee_Intake', y='Stress_Level_num',
            fill=True, cmap='Reds', alpha=0.7,
            ax=ax2)
ax2.set_yticks([1, 2, 3])
ax2.set_yticklabels(['Bajo', 'Medio', 'Alto'])
ax2.set_title('Mujeres', fontsize=12, fontweight='bold')
ax2.set_xlabel('mg de cafeina')
ax2.set_ylabel('Nivel del estres')
ax2.grid(True, alpha=0.3)

plt.suptitle('Consumo de cafeína segun el nivel del estres, por genero', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('img/grafico_h3a.png')
plt.close()

# Gráfico para mostrar la corelación entre el consumo de cafeína y la calidad del sueño por genero
fig, (ax11, ax22) = plt.subplots(1, 2, figsize=(15, 6))

sns.kdeplot(data=df[df['Gender'] == 'Male'], 
            x='Coffee_Intake', y='Sleep_Quality_num',
            fill=True, cmap='Blues', alpha=0.7,
            ax=ax11)
ax11.set_yticks([1, 2, 3, 4])
ax11.set_yticklabels(['Mal', 'Regular', 'Bueno', 'Excelente'])
ax11.set_title('Hombres')
ax11.set_xlabel('mg de cafeina')
ax11.set_ylabel('Calidad del sueño')
ax11.grid(True, alpha=0.3)

# KDE plot para género 2 (ej: Femenino)
sns.kdeplot(data=df[df['Gender'] == 'Female'], 
            x='Coffee_Intake', y='Sleep_Quality_num',
            fill=True, cmap='Reds', alpha=0.7,
            ax=ax22)
ax22.set_yticks([1, 2, 3, 4])
ax22.set_yticklabels(['Mal', 'Regular', 'Bueno', 'Excelente'])
ax22.set_title('Mujeres')
ax22.set_xlabel('mg de cafeina')
ax22.set_ylabel('Calidad del sueño')
ax22.grid(True, alpha=0.3)

plt.suptitle('Consumo de cafeína y la calidad del sueño, por genero', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('img/grafico_h3b.png')
plt.close()


# Gráfico que calcula la  correlaciones entre los mg de café consumidos, la calidad del sueño y el nivel del estres
corr_matrix = df[['Coffee_Intake', 'Sleep_Quality_num', 'Stress_Level_num']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', vmin=-1)
plt.title('Correlación entre Café, Sueño y Estrés')
plt.savefig('img/grafico_h3c.png')
plt.close()

# Gráficos  de tipo violin que muestra las diferencias del consumo de cafeina, nivel de estres y calidad de sueño por genero 
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
data1 = df[df['Gender'].isin(['Male','Female'])]

variables = ['Caffeine_mg', 'Stress_Level', 'Sleep_Quality']
titles = ['Cafeína (mg)', 'Nivel de Estrés', 'Calidad de Sueño']

for i, (var, title) in enumerate(zip(variables, titles)):
    sns.violinplot(x='Gender', y=var, data=data1, ax=axes[i], color=colores_cafe[17])
    axes[i].set_title(f'Distribución de {title} por Género')
plt.savefig('img/grafico_h3d.png')
plt.close()

# ===============================  HÍPOTESIS 4  ==================================
# H4. Los adultos (18-55) toman más café y duermen menos horas que los mayores(56-80).

# Cuantas horas de promedio durmen lod adultos y los mayores
print(df[df['Age_Group']=='Joven']['Sleep_Hours'].mean().round(2))
print(df[df['Age_Group']=='Adulto']['Sleep_Hours'].mean().round(2))
print(df[df['Age_Group']=='Mayor']['Sleep_Hours'].mean().round(2))

# Y cuantos mg de cafeina de promedio toman los adultos y los mayores
print(df[df['Age_Group']=='Joven']['Coffee_Intake'].mean().round(2))
print(df[df['Age_Group']=='Adulto']['Coffee_Intake'].mean().round(2))
print(df[df['Age_Group']=='Mayor']['Coffee_Intake'].mean().round(2))


#Grafico que muestra la distribución de las tazas de cafépor grupos de edad
fig, ax1= plt.subplots(figsize=(8, 6))
sns.boxplot(data=df, x='Age_Group', y='Coffee_Intake', ax=ax1, palette=[colores_cafe[10],colores_cafe[19]], order=['Joven','Adulto','Mayor'])
sns.stripplot(data=df, x='Age_Group', y='Coffee_Intake', ax=ax1, color='black', alpha=0.3, size=3, order=['Joven','Adulto','Mayor'])
ax1.set_ylabel('Tazas de Café')
ax1.set_xlabel('Grupos de edad')
ax1.grid(True, alpha=0.3)
plt.suptitle('Consumo de Café por grupos de edad', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('img/grafico_h4a.png')
plt.close()


# Gráfico de barras horizontales que muestra la calidad del sueño y los mg de café consumidos por grupo de edad
plt.figure(figsize=(8, 4))
ax = sns.barplot(data=df, y='Caffeine_mg', x='Age_Group',hue='Sleep_Quality', hue_order=['Poor', 'Fair', 'Good', 'Excellent'], order=['Joven','Adulto','Mayor'])

plt.title('Calidad del sueño por grupo de edad',fontweight='bold', fontsize=14)
plt.xlabel('')
plt.ylabel('mg de cafeina')
plt.tight_layout()
plt.savefig('img/grafico_h4b.png')
plt.close()

# ===============================  HÍPOTESIS 5  ==================================
# H5: Las horas fisicamente activas se correlacionan positivamente con la calidad del sueño, pero negativamente con el IMC y el nivel del estres.

# Gráfico de distribución de las horas fisicamente activas segun las categorías del IMC teneindo en cuenta el nivel del estres
plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df,
    x='Physical_Activity_Hours',
    y='Categoria_BMI',
    hue='Stress_Level_num',
    palette=[colores_cafe[18], colores_cafe[12],colores_cafe[4]],
    )
plt.title('Distribución de las horas fisicamente activas segun las categorias del IMC')
plt.xlabel('Horas fisicamente activas')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.ylabel('Categorias del IBM')
plt.legend(title= 'Nivel de estres', loc=2, bbox_to_anchor=(1, 1))
plt.savefig('img/grafico_h5a.png')
plt.close()

# Gráfico de distribuición de las horas activas y la duración del sueño
g = sns.jointplot(data=df, x='Physical_Activity_Hours', y='Sleep_Hours', 
                  kind='reg', height=8,
                  joint_kws={'scatter_kws': {'alpha':0.6}, 
                            'line_kws': {'color':'red'}})
g.fig.suptitle('Relación Actividad Física - Calidad del Sueño',fontweight='bold', fontsize=12, y=1.02)
plt.title('Distribución de las horas fisicamente activas y sueño')
plt.ylabel('Horas del sueño')
plt.xlabel('Horas fisicamente activas')
plt.savefig('img/grafico_h5b.png')
plt.close()


# ===============================  HÍPOTESIS 6  ==================================
#H6: Los estudiantes y el personal medico son los que menos duermen y mas café consumen.

# Creo un dataframe con las distintas ocupaciónes y el promedio de las horas que duermen
df_h6=df.groupby('Occupation', as_index=False).mean('Sleep_Hours')[['Occupation','Sleep_Hours']].sort_values(by='Sleep_Hours', ascending=False)


# Gráfico de barras horizontales para mostrar el promedio de las horas dormidas por ocupación
plt.figure(figsize=(8, 4))
ax = sns.barplot(data=df_h6, y='Occupation', x='Sleep_Hours', palette=[colores_cafe[2], colores_cafe[8], colores_cafe[12], colores_cafe[16], colores_cafe[19]])

plt.title('Horas de sueño por ocupacíon',fontweight='bold', fontsize=14)
plt.xlabel('Horas de sueño')
plt.ylabel('Ocupación')
plt.xlim(6.5, 6.75)
plt.tight_layout()
plt.savefig('img/grafico_h6a.png')
plt.close()

# Creo otro dataframe con las distintos ocupaciónes y el promedio de los mg de cafeína consumidos
df_h6b=df.groupby('Occupation', as_index=False).mean('Caffeine_mg').round(2)[['Occupation','Caffeine_mg']].sort_values(by='Caffeine_mg', ascending=False)

# Gráfico de barras horizontales para mostrar el promedio de los mg de cafeína consumidos por ocupación
plt.figure(figsize=(8, 4))
ax = sns.barplot(data=df_h6b, y='Occupation', x='Caffeine_mg', palette=[colores_cafe[2], colores_cafe[8], colores_cafe[12], colores_cafe[16], colores_cafe[19]])

plt.title('Consumo de cafe por ocupación',fontweight='bold', fontsize=14)
plt.xlabel('md de cafeina')
plt.ylabel('Ocupación')
plt.xlim(230, 250)
plt.tight_layout()
plt.savefig('img/grafico_h6b.png')
plt.close()


# ===============================  HÍPOTESIS 7  ==================================
#H7: Las personan que fuman y toman alcool consumen más cafe.

# Crear grupos combinados de los que fuman o no y los que toman alcohol o no
df['Grupo'] = df['Alcohol_Consumption'].astype(str) + '_' + df['Smoking'].astype(str)
mapeo_grupos = {
    '0_0': 'No alcohol\nNo fuma',
    '0_1': 'No alcohol\nFuma', 
    '1_0': 'Alcohol\nNo fuma',
    '1_1': 'Alcohol\nFuma'
}
df['Grupo_etiqueta'] = df['Grupo'].map(mapeo_grupos)

# Calcular promedio de café
promedio_cafe = df.groupby('Grupo_etiqueta')['Caffeine_mg'].mean().round(2).sort_values(ascending=False)

plt.figure(figsize=(8, 4))
sns.heatmap(promedio_cafe.to_frame(), annot=True, cmap='YlOrBr', fmt='.2f', 
            cbar_kws={'label': 'mg de Cafeína'})
plt.ylabel('')
plt.title('Consumo promedio de cafeína en combinación con el alcohol y el tabaco', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('img/grafico_h7.png')
plt.close()