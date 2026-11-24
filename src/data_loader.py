import pandas as pd


def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    return df


def quitar_catalogos(df):
    lista = ['Medicamentos', 'Electrodomésticos']
    df = df[~df['catalogo'].isin(lista)]
    return df


def sacar_cantidad(df):
    datos = df['presentacion'].str.extract(r'(?i)(\d+(?:\.\d+)?)\s*(Gr|Kg|Ml|Lt)\b')
    df['cantidad'] = datos[0]
    df['unidad'] = datos[1].str.lower()
    miles = df['unidad'].isin(['gr', 'ml']) & df['cantidad'].str.contains(r'^\d+\.\d{3}$', na=False)
    df.loc[miles, 'cantidad'] = df['cantidad'].str.replace('.', '', regex=False)
    df['cantidad'] = df['cantidad'].astype(float)
    return df


def calcular_precio_unitario(df):
    df['cantidad_base'] = df['cantidad']
    df['unidad_base'] = df['unidad']
    df.loc[df['unidad'] == 'gr', 'cantidad_base'] = df['cantidad'] / 1000
    df.loc[df['unidad'] == 'gr', 'unidad_base'] = 'kg'
    df.loc[df['unidad'] == 'ml', 'cantidad_base'] = df['cantidad'] / 1000
    df.loc[df['unidad'] == 'ml', 'unidad_base'] = 'lt'
    df['precio_unitario'] = df['precio'] / df['cantidad_base']
    return df