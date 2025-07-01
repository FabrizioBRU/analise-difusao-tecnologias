# -*- coding: utf-8 -*-
"""
PROJETO: VETORES ANTECEDENTES DA DIFUSÃO PARA TECNOLOGIAS EMERGENTES
APLICATIVO WEB INTERATIVO
Autor da Metodologia: Fabrizio Bruzetti (TA - FGV)
Desenvolvimento da Automação: Gemini (Google)
Data: 01 de Julho de 2025
VERSÃO: 25.0 (Versão Final Completa com todos os textos)
"""

# ==============================================================================
# 1. SETUP INICIAL E IMPORTAÇÕES
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import t as t_student
from statsmodels.tsa.stattools import ccf
from statsmodels.nonparametric.smoothers_lowess import lowess
import networkx as nx
from datetime import date
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 2. CONFIGURAÇÃO DA PÁGINA E SIDEBAR
# ==============================================================================
st.set_page_config(
    page_title="Vetores Antecedentes da Difusão",
    page_icon="📈",
    layout="wide"
)

# --- Construção da Barra Lateral (Sidebar) ---
st.sidebar.title("Sobre o Projeto")
st.sidebar.markdown("""
O Projeto visa disponibilizar para a comunidade científica e interessados, 
dados e análises referentes às Curvas Antecedentes da Difusão para tecnologias emergentes.
""")
st.sidebar.divider()

st.sidebar.info(
    "Este site é parte da continuidade da pesquisa e Trabalho Aplicado para obtenção do Título de "
    "Mestre em Gestão para a Competitividade na Linha de Inovação Corporativa na FGV/EAESP - "
    "Escola de Administração de Empresas de São Paulo da Fundação Getúlio Vargas."
)
st.sidebar.divider()

st.sidebar.header("Visão de Futuro")
st.sidebar.markdown(
    "O objetivo desta plataforma é enriquecer o conteúdo ao longo do tempo, "
    "seguindo o mesmo protocolo de análise para outras tecnologias emergentes e "
    "se tornando uma referência no campo."
)
st.sidebar.divider()

st.sidebar.header("Autor e Contato")
st.sidebar.markdown("""
- **Fabrizio Bruzetti**
- [LinkedIn](https://www.linkedin.com/in/fabriziobruzetti/)
- ✉️ fabrizio@bruzetti.com
""")

# ==============================================================================
# 3. BLOCO DE CONFIGURAÇÃO DA ANÁLISE
# ==============================================================================
NOME_TECNOLOGIA = "Computação Quântica"
ARQUIVO_TRENDS = 'Trends_12Jun.csv'
ARQUIVO_PATENTES = 'Patents_csv_puro_12_Jun.csv'
ARQUIVO_CIENCIA = 'Producao_Cientifica_series_temporal_12_Jun.csv'
ANO_FINAL = 2024
FRAC_LOESS_PADRAO = 0.06

# ==============================================================================
# 4. DEFINIÇÃO DAS FUNÇÕES DE ANÁLISE (CACHEADAS PARA PERFORMANCE)
# ==============================================================================
@st.cache_data
def carregar_e_preparar_dados(arq_trends, arq_patentes, arq_ciencia, ano_final):
    df_trends = pd.read_csv(arq_trends, skiprows=2)
    df_patents = pd.read_csv(arq_patentes, delimiter=';', skiprows=1)
    df_science = pd.read_csv(arq_ciencia)
    
    df_trends.columns = ['Semana', 'Interesse']
    df_trends['Semana'] = pd.to_datetime(df_trends['Semana'])
    df_trends['Ano'] = df_trends['Semana'].dt.year
    serie_interesse_anual = df_trends.groupby('Ano')['Interesse'].mean().reset_index()
    serie_interesse_anual = serie_interesse_anual[serie_interesse_anual['Ano'] <= ano_final]

    df_patents.columns = df_patents.columns.str.strip()
    COLUNA_DATA_PATENTE = 'grant date'
    df_patents[COLUNA_DATA_PATENTE] = pd.to_datetime(df_patents[COLUNA_DATA_PATENTE], errors='coerce')
    df_patents.dropna(subset=[COLUNA_DATA_PATENTE], inplace=True)
    df_patents['Ano'] = df_patents[COLUNA_DATA_PATENTE].dt.year
    serie_patentes_anual = df_patents.groupby('Ano').size().reset_index(name='Patentes')
    serie_patentes_anual = serie_patentes_anual[serie_patentes_anual['Ano'] <= ano_final]

    df_science.columns = ['Ano', 'Artigos']
    serie_ciencia_anual = df_science[df_science['Ano'] <= ano_final]

    df_final = pd.merge(serie_interesse_anual, serie_patentes_anual, on='Ano')
    df_final = pd.merge(df_final, serie_ciencia_anual, on='Ano')

    for col in ['Interesse', 'Patentes', 'Artigos']:
        min_val, max_val = df_final[col].min(), df_final[col].max()
        df_final[f'{col}_norm'] = (df_final[col] - min_val) / (max_val - min_val)
        df_final[f'{col}_cum'] = df_final[col].cumsum()
        df_final[f'Cresc_{col}'] = df_final[col].pct_change() * 100
        
    return df_final, df_trends

def plotar_ajuste_geral(t_fit, y_data, titulo, ano_inicial, modelos_fits):
    fig, ax = plt.subplots(figsize=(16, 7))
    anos = np.arange(ano_inicial, ano_inicial + len(y_data))
    
    ax.scatter(anos, y_data, label='Dados Observados (Cumulativo)', color='black', zorder=5)

    if 'Bass' in modelos_fits and modelos_fits['Bass']['params'] is not None:
        y_bass_fit = bass_model(t_fit, *modelos_fits['Bass']['params'])
        ax.plot(anos, y_bass_fit, color='red', linestyle=':', label='Modelo Bass (Ajuste Inadequado)', zorder=3, alpha=0.7)

    if 'Gompertz' in modelos_fits and modelos_fits['Gompertz']['params'] is not None:
        params_g, cov_g = modelos_fits['Gompertz']['params'], modelos_fits['Gompertz']['cov']
        y_pred_g = gompertz_model(t_fit, *params_g)
        alpha=0.05; n=len(y_data); p=len(params_g); dof=max(0,n-p); t_val=t_student.ppf(1.0-alpha/2.0,dof)
        y_err_g = [];
        for i in range(n):
            grad=[];
            for j in range(p): delta=np.zeros(p);delta[j]=1e-6;grad.append((gompertz_model(t_fit[i],*(params_g+delta))-y_pred_g[i])/1e-6)
            var_pred=np.array(grad).T@cov_g@np.array(grad);y_err_g.append(t_val*np.sqrt(var_pred))
        ax.plot(anos, y_pred_g, color='blue', linestyle='--', label='Modelo Gompertz', zorder=4)
        ax.fill_between(anos, y_pred_g - y_err_g, y_pred_g + y_err_g, color='blue', alpha=0.2, label='IC 95% Gompertz')

    if 'Logístico' in modelos_fits and modelos_fits['Logístico']['params'] is not None:
        params_l, cov_l = modelos_fits['Logístico']['params'], modelos_fits['Logístico']['cov']
        y_pred_l = logistic_model(t_fit, *params_l)
        y_err_l = [];
        for i in range(n):
            grad=[];
            for j in range(p): delta=np.zeros(p);delta[j]=1e-6;grad.append((logistic_model(t_fit[i],*(params_l+delta))-y_pred_l[i])/1e-6)
            var_pred=np.array(grad).T@cov_l@np.array(grad);y_err_l.append(t_val*np.sqrt(var_pred))
        ax.plot(anos, y_pred_l, color='green', linestyle='--', label='Modelo Logístico', zorder=4)
        ax.fill_between(anos, y_pred_l - y_err_l, y_pred_l + y_err_l, color='green', alpha=0.2, label='IC 95% Logístico')
    
    ax.set_title(titulo, fontsize=16)
    ax.set_xlabel("Ano", fontsize=12)
    ax.set_ylabel("Valor Cumulativo", fontsize=12)
    ax.legend()
    return fig

def bass_model(t, M, p, q):
    if p + q <= 1e-9 or p <= 1e-9: return np.full_like(t, 1e9)
    a = q / p; b = p + q; exp_bt = np.exp(-b * t); return M * (1 - exp_bt) / (1 + a * exp_bt)
def gompertz_model(t, M, b, c): return M * np.exp(-b * np.exp(-c * t))
def logistic_model(t, M, k, r): return M / (1 + np.exp(-k * (t - r)))
def r_squared(y_true, y_pred):
    residuals = y_true - y_pred; ss_res = np.sum(residuals**2); ss_tot = np.sum((y_true - np.mean(y_true))**2); return 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0


# ==============================================================================
# 5. CONSTRUÇÃO DA INTERFACE PRINCIPAL DO APLICATIVO
# ==============================================================================

st.title(f"Análise de Vetores Antecedentes da Difusão")

st.markdown("""
**Bem-vindo à Plataforma de Análise de Difusão Tecnológica!**
Esta ferramenta interativa explora como vetores antecedentes — **Interesse Público**, **Produção Científica** e **Inovação Formal** — podem antecipar a trajetória de maturação de tecnologias emergentes, usando a **Computação Quântica** como nosso primeiro estudo de caso.
A análise automatizada aplica técnicas quantitativas (normalização, CCF e modelagem de difusão) para diagnosticar o estágio da tecnologia e revelar a dinâmica de retroalimentação entre ciência, patentes e atenção pública, gerando insights para políticas públicas e estratégias empresariais.
A metodologia multivetorial utilizada é uma arquitetura replicável para o diagnóstico antecipado da maturação tecnológica, inspirada em abordagens propostas por **Takahashi, Figueiredo & Scornavacca (2024)** e **Mina et al. (2007)**.
""")
st.info("**Clique no botão abaixo para gerar a análise completa e explorar estes vetores.**")

if 'analysis_generated' not in st.session_state:
    st.session_state.analysis_generated = False

if st.button('▶️ Gerar Análise Completa', key='start_button') or st.session_state.analysis_generated:
    st.session_state.analysis_generated = True

    df_final, df_trends = carregar_e_preparar_dados(ARQUIVO_TRENDS, ARQUIVO_PATENTES, ARQUIVO_CIENCIA, ANO_FINAL)

    with st.expander("Visualizar Tabela de Dados Anuais Consolidados"):
        st.dataframe(df_final)

    st.header("1. Análise Individual e Comparada dos Vetores")
    
    with st.spinner('Gerando gráficos da análise individual...'):
        tab1, tab2, tab3 = st.tabs(["Interesse Público", "Inovação Formal (Patentes)", "Produção Científica"])
        
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'

        with tab1:
            st.markdown("""
            O vetor Interesse Público, operacionalizado por meio de dados do Google Trends, reflete a atenção coletiva direcionada à CQ no ambiente digital. A análise revela um padrão de crescimento moderado e estável entre 2015 e 2022, seguido por uma aceleração expressiva a partir de 2023, culminando em um pico histórico em 2024. Essa intensificação do interesse, compatível com o 'momento de visibilidade pública ampliada' de Rogers (2003), pode estar relacionada ao avanço de estratégias nacionais de desenvolvimento quântico, à consolidação da pauta de criptografia pós-quântica e ao anúncio de casos de uso aplicados em setores estratégicos.
            """)
            fig, ax = plt.subplots(figsize=(12, 6)); sns.barplot(ax=ax, data=df_final, x='Ano', y='Interesse', color='cornflowerblue'); ax.bar_label(ax.containers[0], fmt='%.1f'); ax.set_title("Evolução do Interesse Público (Barras Anuais)", fontsize=14); st.pyplot(fig)
            fig, ax = plt.subplots(figsize=(16, 7)); x_ord = df_trends['Semana'].apply(lambda d: d.toordinal()); loess_smoothed = lowess(df_trends['Interesse'], x_ord, frac=FRAC_LOESS_PADRAO); dates_from_ordinal = [date.fromordinal(int(o)) for o in loess_smoothed[:, 0]]; ax.plot(df_trends['Semana'], df_trends['Interesse'], label='Observado (Semanal)', alpha=0.7); ax.plot(dates_from_ordinal, loess_smoothed[:, 1], color='darkred', linestyle='--', label=f'Tendência Suavizada (frac={FRAC_LOESS_PADRAO})'); ax.legend(); ax.set_title(f"Evolução do Interesse Público por '{NOME_TECNOLOGIA}'", fontsize=14); st.pyplot(fig)

        with tab2:
            st.markdown("""
            O vetor Inovação Formal, representado pelo número anual de patentes concedidas, oferece um indicador da atividade inventiva e do interesse comercial no campo. A série revela um crescimento contínuo, com aceleração acentuada a partir de 2020. Os picos observados em 2023 e 2024 coincidem com o aumento do engajamento de grandes empresas de tecnologia e startups, antecipando o possível uso comercial da CQ. Esse padrão é coerente com o que Geels (2002) denomina 'fortalecimento dos regimes sociotécnicos', quando os atores do sistema buscam consolidar posições antes da adoção em escala.
            """)
            fig, ax = plt.subplots(figsize=(12, 6)); sns.barplot(ax=ax, data=df_final, x='Ano', y='Patentes', color='steelblue'); ax.bar_label(ax.containers[0]); ax.set_title("Evolução das Patentes Publicadas", fontsize=14); st.pyplot(fig)
            fig, ax = plt.subplots(figsize=(16, 7)); x_ord_pat = df_final['Ano']; loess_pat = lowess(df_final['Patentes'], x_ord_pat, frac=FRAC_LOESS_PADRAO * 5); ax.plot(df_final['Ano'], df_final['Patentes'], 'o-', label='Observado (Anual)', alpha=0.7, color='steelblue'); ax.plot(loess_pat[:, 0], loess_pat[:, 1], color='darkred', linestyle='--', label=f'Tendência Suavizada (frac={FRAC_LOESS_PADRAO*5:.2f})'); ax.legend(); ax.set_title("Evolução da Inovação Formal com Tendência Suavizada", fontsize=14); st.pyplot(fig)

        with tab3:
            st.markdown("""
            O vetor Produção Científica, que representa o número anual de artigos publicados, reflete a consolidação cognitiva e o avanço do conhecimento no campo. A série evidencia um crescimento progressivo e acelerado, com um salto significativo em 2024. O padrão pode ser interpretado como um indicador de consolidação cognitiva, com expansão das colaborações internacionais e crescente alocação de recursos institucionais, como argumentam Mowery et al. (2015) sobre tecnologias emergentes que passam de estágios exploratórios para fases de institucionalização científica.
            """)
            fig, ax = plt.subplots(figsize=(12, 6)); sns.barplot(ax=ax, data=df_final, x='Ano', y='Artigos', color='forestgreen'); ax.bar_label(ax.containers[0]); ax.set_title("Evolução dos Artigos Publicados", fontsize=14); st.pyplot(fig)
            fig, ax = plt.subplots(figsize=(16, 7)); x_ord_sci = df_final['Ano']; loess_sci = lowess(df_final['Artigos'], x_ord_sci, frac=FRAC_LOESS_PADRAO * 5); ax.plot(df_final['Ano'], df_final['Artigos'], 'o-', label='Observado (Anual)', alpha=0.7, color='forestgreen'); ax.plot(loess_sci[:, 0], loess_sci[:, 1], color='darkred', linestyle='--', label=f'Tendência Suavizada (frac={FRAC_LOESS_PADRAO*5:.2f})'); ax.legend(); ax.set_title("Evolução da Produção Científica com Tendência Suavizada", fontsize=14); st.pyplot(fig)
        
        st.subheader("Visão Geral: Comparação das Trajetórias Normalizadas")
        fig, ax = plt.subplots(figsize=(16, 8)); ax.plot(df_final['Ano'], df_final['Interesse_norm'], marker='o', label='Interesse Público'); ax.plot(df_final['Ano'], df_final['Patentes_norm'], marker='o', label='Inovação Formal (Patentes)'); ax.plot(df_final['Ano'], df_final['Artigos_norm'], marker='o', label='Produção Científica'); ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        for i, row in df_final.iterrows(): ax.text(row['Ano'], row['Interesse_norm'], f" {row['Interesse_norm']:.2f}", fontsize=9, ha='left'); ax.text(row['Ano'], row['Patentes_norm'], f" {row['Patentes_norm']:.2f}", fontsize=9, ha='left'); ax.text(row['Ano'], row['Artigos_norm'], f" {row['Artigos_norm']:.2f}", fontsize=9, ha='left')
        st.pyplot(fig)

    st.header("2. Análise das Taxas de Crescimento")
    st.markdown("""
    A análise das taxas anuais de crescimento reforça os padrões identificados na evolução dos vetores. A Produção Científica apresenta um crescimento mais estável e sustentado, com taxas expressivas a partir de 2020 e um pico de 60,2% em 2024, evidenciando a consolidação das comunidades de pesquisa. O Interesse Público, por sua vez, demonstra uma trajetória mais volátil, com picos abruptos em 2017 (+90,9%) e 2024 (+50,4%), o que reforça sua sensibilidade a eventos externos, como marcos tecnológicos ou anúncios midiáticos. Por fim, a Inovação Formal exibe um padrão de crescimento mais estratégico, com picos relevantes em 2019 e 2023 que podem indicar movimentos de consolidação e corrida patentária entre os principais players do setor.
    """)
    with st.spinner('Gerando gráficos de crescimento...'):
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        ax1 = sns.barplot(ax=axes[0], data=df_final.dropna(), x='Ano', y='Cresc_Interesse', color='cornflowerblue'); ax1.bar_label(ax1.containers[0], fmt='%.1f%%'); axes[0].set_title('Interesse Público')
        ax2 = sns.barplot(ax=axes[1], data=df_final.dropna(), x='Ano', y='Cresc_Patentes', color='indianred'); ax2.bar_label(ax2.containers[0], fmt='%.1f%%'); axes[1].set_title('Inovação Formal (Patentes)')
        ax3 = sns.barplot(ax=axes[2], data=df_final.dropna(), x='Ano', y='Cresc_Artigos', color='olivedrab'); ax3.bar_label(ax3.containers[0], fmt='%.1f%%'); axes[2].set_title('Produção Científica')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]); st.pyplot(fig)

    st.header("3. Análise de Correlação e Causalidade Inferida")
    st.markdown("""
    A análise de correlação cruzada (Cross-Correlation Function — CCF) entre as séries anuais normalizadas permite mensurar a força da associação entre os vetores em diferentes defasagens temporais (lags), oferecendo uma medida quantitativa para verificar possíveis precedências. A análise dos correlogramas revela padrões consistentes de associação temporal entre os vetores. A Produção Científica apresenta sincronização com a Inovação Formal e o Interesse Público, com correlação máxima em lag zero e reforço em lags positivos (+1 e +2). Isso sugere que o avanço cognitivo no campo da CQ tende a impulsionar, com pequena defasagem, tanto a formalização da inovação quanto o aumento da atenção pública.
    A análise integrada das correlações revela uma dinâmica sistêmica, cuja principal característica é a forte sincronia (lag = 0) entre os vetores. Esse padrão, contrastando com modelos sequenciais clássicos de difusão, pode ser interpretado como uma assinatura de tecnologias de base científica profunda, como a Computação Quântica. A ausência de defasagens significativas sugere a existência de um ciclo de retroalimentação altamente coordenado e veloz, onde avanços na Produção Científica são rapidamente traduzidos em ativos de propriedade intelectual (Inovação Formal) e, por sua vez, ganham visibilidade pública como sinais de progresso (Interesse Público).
    Essa interpretação se alinha à perspectiva sociotécnica de Geels (2002) e Markard & Truffer (2008), segundo a qual tecnologias emergentes evoluem em redes interdependentes de atores, práticas e discursos. Nesse modelo, os vetores não seguem uma sequência rígida, mas operam como 'outputs' simultâneos de pulsos estratégicos coordenados, típicos de setores capital-intensivos e altamente competitivos. O papel catalisador da Produção Científica fica evidente em 2024, ano em que o vetor registrou sua maior taxa de crescimento, coincidindo com o salto observado no Interesse Público e reforçando sua hipótese como motor da cadeia reflexiva.
    """)
    with st.spinner('Gerando gráficos da análise cruzada...'):
        st.subheader("Análise de Correlação (CCF) e Heatmap")
        ccf_matrix = pd.DataFrame(index=['Interesse Público', 'Inovação Formal', 'Produção Científica'], columns=['Interesse Público', 'Inovação Formal', 'Produção Científica'])
        for v1_col, v1_name in zip(['Interesse_norm', 'Patentes_norm', 'Artigos_norm'], ['Interesse Público', 'Inovação Formal', 'Produção Científica']):
            for v2_col, v2_name in zip(['Interesse_norm', 'Patentes_norm', 'Artigos_norm'], ['Interesse Público', 'Inovação Formal', 'Produção Científica']):
                ccf_matrix.loc[v1_name, v2_name] = ccf(df_final[v1_col], df_final[v2_col], adjusted=False)[0]
        ccf_matrix = ccf_matrix.astype(float)
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6)); sns.heatmap(ax=ax, data=ccf_matrix, annot=True, cmap='YlOrRd', fmt=".3f", linewidths=.5, vmin=0.9); ax.set_title("Heatmap das Correlações (Lag=0)", fontsize=14); st.pyplot(fig)
        with col2:
            G = nx.DiGraph(); edges = {("Produção Científica", "Inovação Formal"): ccf_matrix.loc['Produção Científica', 'Inovação Formal'],("Produção Científica", "Interesse Público"): ccf_matrix.loc['Produção Científica', 'Interesse Público'],("Inovação Formal", "Interesse Público"): ccf_matrix.loc['Inovação Formal', 'Interesse Público'],};
            for (u,v), w in edges.items(): G.add_edge(u,v, weight=w)
            pos = nx.circular_layout(G); edge_labels = {(u,v): f"{d['weight']:.3f}" for u,v,d in G.edges(data=True)}; weights = [d['weight'] * 5 for u,v,d in G.edges(data=True)]
            fig, ax = plt.subplots(figsize=(8, 6)); nx.draw(G, pos, ax=ax, with_labels=True, node_color='skyblue', node_size=2500, font_size=10, font_weight='bold', arrowsize=20, width=weights); nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_color='red', font_size=10); ax.set_title("Grafo Direcional das Correlações (Lag=0)", fontsize=14); st.pyplot(fig)

    st.header("4. Modelagem de Curvas de Crescimento")
    st.markdown("""
    Como etapa complementar da análise, foi realizada uma modelagem exploratória aplicando a Curva de Bass (1969) e, posteriormente, as curvas de Gompertz e Logística. A modelagem com a Curva de Bass, que descreve a difusão com base em coeficientes de 'inovação' e 'imitação', revelou-se inadequada para representar a trajetória dos vetores no período analisado. Em todos os casos, os ajustes apresentaram limitações técnicas ou geraram parâmetros inconsistentes com a lógica do modelo, reforçando a hipótese de que os vetores ainda se encontram em fases iniciais de maturação, onde padrões sigmoides clássicos ainda não se manifestam.
    A posterior aplicação dos modelos Gompertz e Logístico ofereceu curvas com melhor aderência e parâmetros estatisticamente mais robustos. Ambos são reconhecidos na literatura por sua flexibilidade em capturar padrões de crescimento ainda em aceleração (Meade & Islam, 2006). O modelo de Gompertz, em particular, demonstrou ser mais adequado à fase de aceleração contínua e pré-saturação dos vetores.
    Para a Produção Científica, por exemplo, os modelos não conseguiram capturar adequadamente a forte aceleração recente, subestimando a curva real e reforçando o diagnóstico de que a produção acadêmica se encontra em plena expansão cognitiva, sem sinais de saturação próximos - condição comum em tecnologias ainda em consolidação epistemológica (Rotolo et al., 2015). A análise dos parâmetros R² e dos Intervalos de Confiança (IC 95%) para cada ajuste enriquece o diagnóstico, quantificando a qualidade dos modelos e a incerteza das projeções.
    """)
    
    with st.spinner('Executando modelagem das curvas...'):
        time_axis = np.arange(len(df_final))
        vetores = {"Interesse Público": df_final['Interesse_cum'], "Inovação Formal": df_final['Patentes_cum'], "Produção Científica": df_final['Artigos_cum']}
        resultados_modelagem = {}

        for nome, serie in vetores.items():
            y_data = serie.values; resultados_modelagem[nome] = {}
            modelos_fits = {}
            
            try:
                params_b, cov_b = curve_fit(bass_model, time_axis, y_data, p0=[max(y_data), 0.03, 0.38], maxfev=10000, bounds=([0,0,0], [max(y_data)*5, 1, 5])); r2 = r_squared(y_data, bass_model(time_axis, *params_b)); resultados_modelagem[nome]['Bass'] = {'params': params_b, 'r2': r2}; modelos_fits['Bass'] = {'params': params_b, 'cov': cov_b, 'func': bass_model}
            except Exception: resultados_modelagem[nome]['Bass'] = "Falha"
            try:
                params_g, cov_g = curve_fit(gompertz_model, time_axis, y_data, p0=[max(y_data), 4, 0.1], maxfev=10000); r2 = r_squared(y_data, gompertz_model(time_axis, *params_g)); resultados_modelagem[nome]['Gompertz'] = {'params': params_g, 'r2': r2}; modelos_fits['Gompertz'] = {'params': params_g, 'cov': cov_g, 'func': gompertz_model}
            except Exception: resultados_modelagem[nome]['Gompertz'] = "Falha"
            try:
                params_l, cov_l = curve_fit(logistic_model, time_axis, y_data, p0=[max(y_data), 0.3, np.median(time_axis)], maxfev=10000); r2 = r_squared(y_data, logistic_model(time_axis, *params_l)); resultados_modelagem[nome]['Logístico'] = {'params': params_l, 'r2': r2}; modelos_fits['Logístico'] = {'params': params_l, 'cov': cov_l, 'func': logistic_model}
            except Exception: resultados_modelagem[nome]['Logístico'] = "Falha"
            
            st.subheader(f"Ajustes de Modelo para: {nome}")
            fig = plotar_ajuste_geral(time_axis, y_data, f"Comparativo de Modelos - {nome}", df_final['Ano'].min(), modelos_fits)
            st.pyplot(fig)

    st.header("5. Síntese dos Resultados")
    st.markdown("""
    A tabela de síntese consolida os principais achados quantitativos do estudo. Ela permite uma visão panorâmica dos diferentes estágios de maturação de cada vetor antecedente, reforçando a interpretação de que a tecnologia se encontra em uma fase de consolidação pré-comercial. A Produção Científica se destaca pelo crescimento robusto e sustentado, atuando como motor cognitivo do ecossistema. A Inovação Formal, embora crescente, mostra sinais de uma possível transição para uma fase de menor aceleração, enquanto o Interesse Público, mais volátil, reflete a crescente visibilidade da CQ para além dos círculos de especialistas.
    Em conjunto, os resultados sugerem que a abordagem multivetorial é uma ferramenta eficaz para o diagnóstico de tecnologias emergentes, oferecendo um 'painel de controle' que, embora não preveja o futuro com exatidão, mapeia o presente com rigor. A análise evidencia que a maturação da CQ é um fenômeno sistêmico, interdependente e ainda em plena evolução, com implicações estratégicas para empresas, governos e investidores que buscam se posicionar na vanguarda da próxima revolução computacional.
    """)

    with st.spinner('Gerando tabela de síntese...'):
        sintese = pd.DataFrame(index=['Interesse Público', 'Inovação Formal', 'Produção Científica'])
        sintese['Crescimento Médio Anual (%)'] = [df_final['Cresc_Interesse'].mean(), df_final['Cresc_Patentes'].mean(), df_final['Cresc_Artigos'].mean()]
        sintese['CCF max (vs. Produção Científica)'] = [ccf_matrix.loc['Interesse Público', 'Produção Científica'], ccf_matrix.loc['Inovação Formal', 'Produção Científica'], 1.0]

        for n in vetores.keys():
            for m in ['Bass', 'Gompertz', 'Logístico']:
                res = resultados_modelagem[n].get(m, "Falha")
                if isinstance(res, dict): sintese.loc[n, f'Parâmetros {m}'] = str(np.round(res['params'], 2)); sintese.loc[n, f'R² {m}'] = res['r2']
                else: sintese.loc[n, f'Parâmetros {m}'] = res; sintese.loc[n, f'R² {m}'] = 0.0
        
        st.dataframe(sintese.round(4))
    
    st.balloons()
    st.success('Análise completa gerada com sucesso!')
