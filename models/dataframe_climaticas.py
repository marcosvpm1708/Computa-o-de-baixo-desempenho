import pandas as pd
from utils import tempo_execucao

class base_data():

    def __init__(self, base_path : str = "../dataframe/BD_Atlas_1991_2024_v1.0_2025.04.14_Consolidado.xlsx") -> pd.DataFrame:
        self.base_data = pd.read_excel(base_path)

    def main(self):
        self._duplicates()
        difs = self._features()
        self._null_values()  # chamada duplicada, possivelmente um erro?
        self._schema()
        self.base_data.to_csv('../dataframe/BD_Atlas_1991_2024_v2.csv', index=False, encoding='utf-8')

    @tempo_execucao
    def _duplicates(self):
        self.base_data = self.base_data.drop_duplicates()

    @tempo_execucao
    def _null_values(self):
        self.base_data = self.base_data.fillna(-1)

    @tempo_execucao
    def _features(self):

        trash = ['DH_Descricao', 'DM_Descricao', 'DA_Descricao', 'PEPL_Descricao', 'PEPR_Descricao', 'Setores Censitários', 'Status']
        self.base_data.drop(columns=trash, inplace=True)


    @tempo_execucao
    def _schema(self):
        import pandas as pd

        df = self.base_data.copy()

        schema = {
            'Protocolo_S2iD': 'object',
            'Nome_Municipio': 'object',
            'Sigla_UF': 'object',
            'regiao': 'object',
            'Data_Registro': 'datetime64[ns]',
            'Data_Evento': 'datetime64[ns]',
            'Cod_Cobrade': 'int64',
            'tipologia': 'object',
            'descricao_tipologia': 'object',
            'grupo_de_desastre': 'object',
            'Cod_IBGE_Mun': 'int64',
            'DH_MORTOS': 'int64',
            'DH_FERIDOS': 'int64',
            'DH_ENFERMOS': 'int64',
            'DH_DESABRIGADOS': 'int64',
            'DH_DESALOJADOS': 'int64',
            'DH_DESAPARECIDOS': 'int64',
            'DH_OUTROS AFETADOS': 'int64',
            'DH_AFETADOS_SECA_ESTIAGEM': 'int64',
            'DH_total_danos_humanos_diretos': 'int64',
            'DM_Uni Habita Danificadas': 'int64',
            'DM_Uni Habita Destruidas': 'int64',
            'DM_Uni Habita Valor': 'int64',
            'DM_Inst Saúde Danificadas': 'int64',
            'DM_Inst Saúde Destruidas': 'int64',
            'DM_Inst Saúde Valor': 'int64',
            'DM_Inst Ensino Danificadas': 'int64',
            'DM_Inst Ensino Destruidas': 'int64',
            'DM_Inst Ensino Valor': 'int64',
            'DM_Inst Serviços Danificadas': 'int64',
            'DM_Inst Serviços Destruidas': 'int64',
            'DM_Inst Serviços Valor': 'int64',
            'DM_Inst Comuni Danificadas': 'int64',
            'DM_Inst Comuni Destruidas': 'int64',
            'DM_Inst Comuni Valor': 'int64',
            'DM_Obras de Infra Danificadas': 'int64',
            'DM_Obras de Infra Destruidas': 'int64',
            'DM_Obras de Infra Valor': 'int64',
            'DM_total_danos_materiais': 'int64',
            'DA_Polui/cont da água': 'object',
            'DA_Polui/cont do ar': 'object',
            'DA_Polui/cont do solo': 'object',
            'DA_Dimi/exauri hídrico': 'object',
            "DA_Incêndi parques/APA's/APP's": 'object',
            'PEPL_Assis_méd e emergên(R$)': 'int64',
            'PEPL_Abast de água pot(R$)': 'int64',
            'PEPL_sist de esgotos sanit(R$)': 'int64',
            'PEPL_Sis limp e rec lixo (R$)': 'int64',
            'PEPL_Sis cont pragas (R$)': 'int64',
            'PEPL_distrib energia (R$)': 'int64',
            'PEPL_Telecomunicações (R$)': 'int64',
            'PEPL_Tran loc/reg/l_curso (R$)': 'int64',
            'PEPL_Distrib combustíveis(R$)': 'int64',
            'PEPL_Segurança pública (R$)': 'int64',
            'PEPL_Ensino (R$)': 'int64',
            'PEPL_total_publico': 'int64',
            'PEPR_Agricultura (R$)': 'int64',
            'PEPR_Pecuária (R$)': 'int64',
            'PEPR_Indústria (R$)': 'int64',
            'PEPR_Comércio (R$)': 'int64',
            'PEPR_Serviços (R$)': 'int64',
            'PEPR_total_privado': 'int64',
            'PE_PLePR': 'int64',
        }

        missing_cols = [col for col in schema if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colunas ausentes: {missing_cols}")

        unexpected_cols = [col for col in df.columns if col not in schema]
        if unexpected_cols:
            raise ValueError(f"Colunas inesperadas: {unexpected_cols}")

        # Converte colunas de data
        df['Data_Registro'] = pd.to_datetime(df['Data_Registro'], errors='coerce')
        df['Data_Evento'] = pd.to_datetime(df['Data_Evento'], errors='coerce')

        schema_no_dates = schema.copy()
        schema_no_dates.pop('Data_Registro')
        schema_no_dates.pop('Data_Evento')

        df = df.astype(schema_no_dates)

        self.base_data = df


if __name__ == "__main__":
    base_data().main()
