import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from geopy.geocoders import Nominatim

class MonteCarlo:
    def __init__(self):
        self.data = pd.read_csv("../dataframe/BD_Atlas_1991_2024_v2.csv")
        self.keys = ['Protocolo_S2iD', 'Cod_Cobrade', 'Cod_IBGE_Mun', 'Nome_Munucipio', 'SIGLA_UF', 'regiao']

    def main(self):
        self.pre_processing()
        self.train_and_validate()

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

    def simulate_data(self, df_train, n_samples):
        """
        Gera dados simulados via Monte Carlo para cada coluna com base no df_train.
        Mantém as chaves primárias fixas (self.keys).
        """

        df_sim = pd.DataFrame()

        for key in self.keys:
            if key in df_train.columns:
                df_sim[key] = np.random.choice(df_train[key].values, n_samples, replace=True)


        for col in df_train.columns:
            if col in self.keys or col == 'geometry':
                continue  # ignora as keys e geometria

            if pd.api.types.is_numeric_dtype(df_train[col]):
                # Ajusta uma distribuição normal (média e std) ou uniforme se std=0
                mu = df_train[col].mean()
                sigma = df_train[col].std()
                if np.isnan(sigma) or sigma == 0:
                    sigma = 0.01  # evitar std=0
                simulated = np.random.normal(mu, sigma, n_samples)
                # Se coluna original não aceita negativos, ajustar:
                if (df_train[col] >= 0).all():
                    simulated = np.clip(simulated, 0, None)
                df_sim[col] = simulated

            else:
                # Categórico: simula pela distribuição empírica
                vals = df_train[col].dropna().unique()
                probs = df_train[col].value_counts(normalize=True)
                simulated = np.random.choice(vals, n_samples, p=probs.loc[vals].values)
                df_sim[col] = simulated

        return df_sim

    def train_and_validate(self):
        # Considera que existe uma coluna 'Ano' para separar treino e validação
        if 'Ano' not in self.data.columns:
            raise ValueError("Coluna 'Ano' é necessária para separar treino e validação")

        df_train = self.data[self.data['Ano'] <= 2020].copy()
        df_val = self.data[self.data['Ano'] > 2020].copy()

        print(f"Treino: {df_train.shape}, Validação: {df_val.shape}")

        n_samples = df_val.shape[0]
        df_simulated = self.simulate_data(df_train, n_samples)

        for col in df_simulated.columns:
            if col in self.keys or col == 'geometry':
                continue
            print(f"Coluna: {col}")
            print(f" - Média real: {df_val[col].mean():.3f} vs Simulada: {df_simulated[col].mean():.3f}")
            print(f" - Std real: {df_val[col].std():.3f} vs Simulada: {df_simulated[col].std():.3f}")

        df_simulated.to_csv("simulated_data.csv", index=False)
        print("Simulação concluída e salva em 'simulated_data.csv'.")

if __name__ == "__main__":
    monte_carlo = MonteCarlo()
    monte_carlo.main()
