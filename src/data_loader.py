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
    df['cantidad'] = datos[0].astype(float)
    df['unidad'] = datos[1].str.lower()
    return df