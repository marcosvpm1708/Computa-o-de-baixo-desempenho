import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from geopy.geocoders import Nominatim

class MonteCarlo:

    def __init__(self, data = pd.read_csv("../dataframe/BD_Atlas_1991_2024_v2.csv")):
        self.data = data

    def main(self, pre_processing = True):
        if pre_processing:
            self.pre_processing()

        self.keys = data['Protocolo_S2iD', 'Data_Registro', 'Cod_Cobrade', 'Cod_IBGE_Mun', 'Nome_Munucipio', 'SIGLA_UF', 'regiao']
        self.train_monte_carlo()



    def pre_processing(self):
        df = self.data

        for col in ['regiao', 'grupo_de_desastre']:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        if 'Nome_Municipio' in df.columns and 'Sigla_UF' in df.columns:
            df['fullAddress'] = df['Nome_Municipio'].astype(str) + ', ' + df['Sigla_UF'].astype(str) + ', Brasil'
            df.drop(columns=['Nome_Municipio', 'Sigla_UF'], inplace=True)
        else:
            raise ValueError("Campos 'Nome_Municipio' e/ou 'Sigla_UF' ausentes do dataframe")

        self.data['lat'], self.data['lon'] = np.nan, np.nan

        geolocator = Nominatim(user_agent="myGeocoder", scheme='http', domain='localhost:8080')

        def get_lat_lon(address: str):
            try:
                location = geolocator.geocode(address, timeout= 1)
                if location:
                    print(location.latitude, location.longitude)
                    return location.latitude, location.longitude
                else:
                    return None, None
            except Exception as e:
                print(f"Erro ao obter coordenadas para o endereço: {address}. Erro: {e}")
                return None, None

        for idx, row in self.data[self.data['lat'].isna() | self.data['lon'].isna()].iterrows():
            address = row['fullAddress']
            lat, lon = get_lat_lon(address)
            if lat is not None and lon is not None:
               self.data.at[idx, 'lat'] = lat
               self.data.at[idx, 'lon'] = lon


        cols_remove = ['DA_Polui/cont da água', 'DA_Polui/cont do ar', 'DA_Polui/cont do solo',
                       'DA_Dimi/exauri hídrico', "DA_Incêndi parques/APA's/APP's", 'descricao_tipologia']
        df.drop(columns=[c for c in cols_remove if c in df.columns], inplace=True)

        df.to_csv('../dataframe/BD_Atlas_1991_2024_Monte_Carlo.csv', index=False)
        self.data = df

    def train_monte_carlo(self):
        print(self.data.info())


if __name__ == "__main__":
    data = pd.read_csv("../dataframe/BD_Atlas_1991_2024_Monte_Carlo.csv", low_memory=False)
    monte_carlo = MonteCarlo()
    monte_carlo.main(pre_processing = False)
