# app.py

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
# --- Importaciones de Módulos ---
from modules.config import Config
from modules.data_processor import load_and_process_all_data, complete_series, extract_elevation_from_dem, download_and_load_remote_dem
from modules.visualizer import (
    display_welcome_tab, display_spatial_distribution_tab, display_graphs_tab,
    display_advanced_maps_tab, display_anomalies_tab, display_drought_analysis_tab,
    display_stats_tab, display_correlation_tab, display_enso_tab,
    display_trends_and_forecast_tab, display_downloads_tab, display_station_table_tab
)

# Desactivar Warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Funciones Auxiliares ---

def sync_station_selection():
    """Sincroniza la selección de 'seleccionar todas las estaciones'."""
    options = sorted(st.session_state.get('filtered_station_options', []))
    if st.session_state.get('select_all_checkbox', True):
        st.session_state.station_multiselect = options
    else:
        st.session_state.station_multiselect = []

def apply_filters_to_stations(df, min_perc, altitudes, regions, municipios, celdas):
    """Aplica los filtros geográficos y de datos a las estaciones."""
    stations_filtered = df.copy()
    if Config.PERCENTAGE_COL in stations_filtered.columns:
        # Asegurarse que la columna de porcentaje sea numérica
        stations_filtered[Config.PERCENTAGE_COL] = pd.to_numeric(
            stations_filtered[Config.PERCENTAGE_COL].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        ).fillna(0)
        stations_filtered = stations_filtered[stations_filtered[Config.PERCENTAGE_COL] >= min_perc]
    
    # Filtro por altitud
    if altitudes:
        conditions = []
        for r in altitudes:
            if r == '0-500':
                conditions.append((stations_filtered[Config.ALTITUDE_COL] >= 0) & (stations_filtered[Config.ALTITUDE_COL] <= 500))
            elif r == '500-1000':
                conditions.append((stations_filtered[Config.ALTITUDE_COL] > 500) & (stations_filtered[Config.ALTITUDE_COL] <= 1000))
            elif r == '1000-2000':
                conditions.append((stations_filtered[Config.ALTITUDE_COL] > 1000) & (stations_filtered[Config.ALTITUDE_COL] <= 2000))
            elif r == '2000-3000':
                conditions.append((stations_filtered[Config.ALTITUDE_COL] > 2000) & (stations_filtered[Config.ALTITUDE_COL] <= 3000))
            elif r == '>3000':
                conditions.append(stations_filtered[Config.ALTITUDE_COL] > 3000)
        if conditions:
            stations_filtered = stations_filtered[pd.concat(conditions, axis=1).any(axis=1)]

    # Filtros geográficos simples
    if regions:
        stations_filtered = stations_filtered[stations_filtered[Config.REGION_COL].isin(regions)]
    if municipios:
        stations_filtered = stations_filtered[stations_filtered[Config.MUNICIPALITY_COL].isin(municipios)]
    if celdas:
        stations_filtered = stations_filtered[stations_filtered[Config.CELL_COL].isin(celdas)]
        
    return stations_filtered

# --- Función Principal de la Aplicación ---

def main():
    """Función principal que ejecuta la aplicación Streamlit."""
    [cite_start]st.set_page_config(layout="wide", page_title=Config.APP_TITLE) [cite: 43]
    st.markdown("""
        <style>
        div.block-container {padding-top: 2rem;}
        .sidebar .sidebar-content {font-size: 13px;}
        [data-testid="stMetricValue"] {font-size: 1.8rem;}
        [data-testid="stMetricLabel"] {font-size: 1rem; padding-bottom: 5px;}
        button[data-baseweb="tab"] {font-size: 16px; font-weight: bold; color: #333;}
        </style>
    [cite_start]""", unsafe_allow_html=True) [cite: 44-52]

    [cite_start]Config.initialize_session_state() [cite: 53]

    # --- Título y Logo ---
    [cite_start]title_col1, title_col2 = st.columns([0.07, 0.93]) [cite: 54]
    with title_col1:
        [cite_start]if os.path.exists(Config.LOGO_PATH): [cite: 56]
            try:
                [cite_start]st.image(Config.LOGO_PATH, width=50) [cite: 59]
            except Exception:
                [cite_start]pass [cite: 60-61]
    with title_col2:
        [cite_start]st.markdown(f'<h1 style="font-size:28px; margin-top:1rem;">{Config.APP_TITLE}</h1>', unsafe_allow_html=True) [cite: 63-64]

    [cite_start]st.sidebar.header("Panel de Control") [cite: 65]
    
    # --- 1. LÓGICA DE CARGA DE DATOS ---
    update_data_toggle = st.sidebar.checkbox(
        "Activar Carga/Actualización de Archivos Base",
        value=not st.session_state.get('data_loaded', False),
        key='update_data_toggle'
    [cite_start]) [cite: 67-71]

    if update_data_toggle:
        [cite_start]with st.sidebar.expander("**Subir/Actualizar Archivos Base**", expanded=True): [cite: 73]
            [cite_start]uploaded_file_mapa = st.file_uploader("1. Cargar archivo de estaciones (CSV)", type="csv", key='uploaded_file_mapa') [cite: 74-75]
            [cite_start]uploaded_file_precip = st.file_uploader("2. Cargar archivo de precipitación (CSV)", type="csv", key='uploaded_file_precip') [cite: 76]
            [cite_start]uploaded_zip_shapefile = st.file_uploader("3. Cargar shapefile de municipios (.zip)", type="zip", key='uploaded_zip_shapefile') [cite: 77-78]

            [cite_start]if st.button("Procesar y Almacenar Datos", key='process_data_button') and all([uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile]): [cite: 79-80]
                st.cache_resource.clear()
                [cite_start]with st.spinner("Procesando archivos y cargando datos..."): [cite: 82]
                    gdf_stations, gdf_municipios, df_long, df_enso = load_and_process_all_data(
                        [cite_start]uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile) [cite: 83]
                    
                    [cite_start]if gdf_stations is not None and df_long is not None and gdf_municipios is not None: [cite: 84]
                        [cite_start]st.session_state.gdf_stations = gdf_stations [cite: 85]
                        [cite_start]st.session_state.gdf_municipios = gdf_municipios [cite: 86]
                        st.session_state.df_long = df_long
                        [cite_start]st.session_state.df_enso = df_enso [cite: 87]
                        [cite_start]st.session_state.data_loaded = True [cite: 88]
                        [cite_start]st.success("¡ Datos cargados y listos!") [cite: 89]
                        st.rerun()
                    else:
                        [cite_start]st.error("Hubo un error al procesar los archivos. Verifique el formato de los datos.") [cite: 91]

    # --- LÓGICA DE FILTROS Y DESPLIEGUE (Solo si los datos están cargados) ---
    [cite_start]if st.session_state.get('data_loaded', False) and st.session_state.get('df_long') is not None: [cite: 93]
        st.sidebar.success(" Datos base cargados y persistentes.")
        
        [cite_start]if st.sidebar.button("Limpiar Caché y Recargar"): [cite: 95]
            [cite_start]st.cache_data.clear() [cite: 96]
            [cite_start]st.cache_resource.clear() [cite: 97]
            [cite_start]keys_to_clear = list(st.session_state.keys()) [cite: 98]
            for key in keys_to_clear:
                [cite_start]del st.session_state[key] [cite: 100]
            [cite_start]st.rerun() [cite: 101]

        # --- 2. LÓGICA DE DEM ---
        [cite_start]with st.sidebar.expander("Opciones de Modelo Digital de Elevación (DEM)", expanded=True): [cite: 103]
            dem_source = st.radio(
                "Fuente de DEM para KED (Kriging):",
                ("No usar DEM", "Subir DEM propio (GeoTIFF)", "Cargar DEM desde Servidor"),
                key="dem_source"
            [cite_start]) [cite: 104-108]
            uploaded_dem_file = None
            [cite_start]if dem_source == "Subir DEM propio (GeoTIFF)": [cite: 110]
                uploaded_dem_file = st.file_uploader(
                    "Cargar GeoTIFF (.tif) para elevación",
                    type=["tif", "tiff"],
                    key="dem_uploader"
                [cite_start]) [cite: 111-115]
            
            [cite_start]if dem_source == "Cargar DEM desde Servidor": [cite: 116]
                [cite_start]if st.button("Descargar y Usar DEM Remoto", key="download_dem_button"): [cite: 117]
                    with st.spinner("Descargando DEM del servidor..."):
                        try:
                            [cite_start]st.session_state.dem_raster = download_and_load_remote_dem(Config.DEM_SERVER_URL) [cite: 120]
                            [cite_start]st.success("DEM remoto cargado y listo para KED.") [cite: 121]
                        except Exception as e:
                            [cite_start]st.error(f"Error al cargar DEM remoto: {e}. Verifique la URL en Config.py") [cite: 123]
                            st.session_state.dem_raster = None

            # Procesamiento del DEM
            [cite_start]if dem_source == "No usar DEM": [cite: 125]
                [cite_start]st.session_state.dem_raster = None [cite: 126]
            
            [cite_start]if uploaded_dem_file or st.session_state.get('dem_raster') is not None: [cite: 127]
                [cite_start]if f'original_{Config.ALTITUDE_COL}' not in st.session_state: [cite: 128]
                    [cite_start]st.session_state[f'original_{Config.ALTITUDE_COL}'] = st.session_state.gdf_stations.get(Config.ALTITUDE_COL, None).copy() [cite: 129-130]
                
                [cite_start]dem_data = uploaded_dem_file if uploaded_dem_file else st.session_state.dem_raster [cite: 131]
                st.session_state.gdf_stations = extract_elevation_from_dem(
                    st.session_state.gdf_stations.copy(),
                    dem_data
                [cite_start]) [cite: 132-135]
                [cite_start]st.sidebar.success("Altitud de estaciones actualizada.") [cite: 136]
            else:
                [cite_start]if st.session_state.get(f'original_{Config.ALTITUDE_COL}') is not None: [cite: 138]
                    [cite_start]st.session_state.gdf_stations[Config.ALTITUDE_COL] = st.session_state[f'original_{Config.ALTITUDE_COL}'] [cite: 140-141]

        # --- 3. FILTROS GEOGRÁFICOS ---
        [cite_start]with st.sidebar.expander("**1. Filtros Geográficos y de Datos**", expanded=True): [cite: 143]
            [cite_start]min_data_perc = st.slider("Filtrar por % de datos mínimo:", 0, 100, st.session_state.get('min_data_perc_slider', 0), key='min_data_perc_slider') [cite: 144-145]
            [cite_start]altitude_ranges = ['0-500', '500-1000', '1000-2000', '2000-3000', '>3000'] [cite: 146]
            [cite_start]selected_altitudes = st.multiselect('Filtrar por Altitud (m)', options=altitude_ranges, key='altitude_multiselect') [cite: 147-148]
            
            [cite_start]regions_list = sorted(st.session_state.gdf_stations[Config.REGION_COL].dropna().unique()) [cite: 149]
            [cite_start]selected_regions = st.multiselect('Filtrar por Depto/Región', options=regions_list, key='regions_multiselect') [cite: 150]
            
            [cite_start]temp_gdf_for_mun = st.session_state.gdf_stations.copy() [cite: 151]
            if selected_regions:
                [cite_start]temp_gdf_for_mun = temp_gdf_for_mun[temp_gdf_for_mun[Config.REGION_COL].isin(selected_regions)] [cite: 152-154]
            [cite_start]municipios_list = sorted(temp_gdf_for_mun[Config.MUNICIPALITY_COL].dropna().unique()) [cite: 155]
            [cite_start]selected_municipios = st.multiselect('Filtrar por Municipio', options=municipios_list, key='municipios_multiselect') [cite: 156]
            
            [cite_start]temp_gdf_for_celdas = temp_gdf_for_mun.copy() [cite: 157]
            if selected_municipios:
                [cite_start]temp_gdf_for_celdas = temp_gdf_for_celdas[temp_gdf_for_celdas[Config.MUNICIPALITY_COL].isin(selected_municipios)] [cite: 159]
            [cite_start]celdas_list = sorted(temp_gdf_for_celdas[Config.CELL_COL].dropna().unique()) [cite: 160]
            [cite_start]selected_celdas = st.multiselect('Filtrar por Celda_XY', options=celdas_list, key='celdas_multiselect') [cite: 161, 163]

            [cite_start]if st.button("Limpiar Filtros Geográficos"): [cite: 164]
                keys_to_clear = ['min_data_perc_slider', 'altitude_multiselect', 'regions_multiselect', 
                                 'municipios_multiselect', 'celdas_multiselect', 'station_multiselect', 
                                 [cite_start]'select_all_checkbox', 'year_range', 'meses_nombres'] [cite: 165-166]
                for key in keys_to_clear:
                    if key in st.session_state:
                        [cite_start]del st.session_state[key] [cite: 169]
                [cite_start]st.rerun() [cite: 170]

        [cite_start]gdf_filtered = apply_filters_to_stations(st.session_state.gdf_stations, min_data_perc, selected_altitudes, selected_regions, selected_municipios, selected_celdas) [cite: 172-174]

        # --- 4. FILTROS TEMPORALES Y PREPROCESAMIENTO ---
        [cite_start]with st.sidebar.expander("**2. Selección de Estaciones y Período**", expanded=True): [cite: 176]
            stations_options = sorted(gdf_filtered[Config.STATION_NAME_COL].unique())
            [cite_start]st.session_state['filtered_station_options'] = stations_options [cite: 177]
            
            st.checkbox(
                "Seleccionar/Deseleccionar todas las estaciones",
                key='select_all_checkbox',
                on_change=sync_station_selection,
                value=st.session_state.get('select_all_checkbox', True)
            [cite_start]) [cite: 178-183]

            [cite_start]if st.session_state.get('select_all_checkbox', True) and st.session_state.get('station_multiselect') != stations_options: [cite: 184-185]
                [cite_start]st.session_state.station_multiselect = stations_options [cite: 186]
            
            selected_stations = st.multiselect(
                'Seleccionar Estaciones',
                options=stations_options,
                key='station_multiselect'
            [cite_start]) [cite: 187-191]

            [cite_start]years_with_data = sorted(st.session_state.df_long[Config.YEAR_COL].dropna().unique()) [cite: 192]
            if not years_with_data:
                [cite_start]st.error("No hay datos de año válidos en el archivo de precipitación.") [cite: 193]
                return
            
            [cite_start]year_range_default = (min(years_with_data), max(years_with_data)) [cite: 194]
            [cite_start]year_range = st.slider("Seleccionar Rango de Años", min_value=min(years_with_data), max_value=max(years_with_data), value=st.session_state.get('year_range', year_range_default), key='year_range') [cite: 195-197]
            
            [cite_start]meses_dict = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12} [cite: 198]
            [cite_start]meses_nombres = st.multiselect("Seleccionar Meses", list(meses_dict.keys()), default=list(meses_dict.keys()), key='meses_nombres') [cite: 199]
            [cite_start]meses_numeros = [meses_dict[m] for m in meses_nombres] [cite: 200]

        [cite_start]with st.sidebar.expander("Opciones de Preprocesamiento de Datos"): [cite: 201]
            [cite_start]st.radio("Análisis de Series Mensuales", ("Usar datos originales", "Completar series (interpolación)"), key="analysis_mode") [cite: 202-203]
            [cite_start]st.checkbox("Excluir datos nulos (NaN)", key='exclude_na') [cite: 204]
            [cite_start]st.checkbox("Excluir valores cero (0)", key='exclude_zeros') [cite: 205]

        # --- PREPARACIÓN DE DATAFRAMES FINALES ---
        [cite_start]stations_for_analysis = selected_stations [cite: 207]
        [cite_start]gdf_filtered = gdf_filtered[gdf_filtered[Config.STATION_NAME_COL].isin(stations_for_analysis)] [cite: 208]
        st.session_state.meses_numeros = meses_numeros

        [cite_start]if st.session_state.analysis_mode == "Completar series (interpolación)": [cite: 210]
            [cite_start]df_monthly_processed = complete_series(st.session_state.df_long.copy()) [cite: 211]
        else:
            [cite_start]df_monthly_processed = st.session_state.df_long.copy() [cite: 212]
        [cite_start]st.session_state.df_monthly_processed = df_monthly_processed [cite: 213]

        df_monthly_filtered = df_monthly_processed[
            (df_monthly_processed[Config.STATION_NAME_COL].isin(stations_for_analysis)) &
            (df_monthly_processed[Config.DATE_COL].dt.year >= year_range[0]) &
            (df_monthly_processed[Config.DATE_COL].dt.year <= year_range[1]) &
            (df_monthly_processed[Config.DATE_COL].dt.month.isin(meses_numeros))
        [cite_start]].copy() [cite: 215-219]
        
        annual_data_filtered = st.session_state.df_long[
            (st.session_state.df_long[Config.STATION_NAME_COL].isin(stations_for_analysis)) &
            (st.session_state.df_long[Config.YEAR_COL] >= year_range[0]) &
            (st.session_state.df_long[Config.YEAR_COL] <= year_range[1])
        [cite_start]].copy() [cite: 221-224]

        [cite_start]if st.session_state.get('exclude_na', False): [cite: 226]
            [cite_start]df_monthly_filtered.dropna(subset=[Config.PRECIPITATION_COL], inplace=True) [cite: 227]
            [cite_start]annual_data_filtered.dropna(subset=[Config.PRECIPITATION_COL], inplace=True) [cite: 228]
        
        [cite_start]if st.session_state.get('exclude_zeros', False): [cite: 229]
            [cite_start]df_monthly_filtered = df_monthly_filtered[df_monthly_filtered[Config.PRECIPITATION_COL] > 0] [cite: 230]
            [cite_start]annual_data_filtered = annual_data_filtered[annual_data_filtered[Config.PRECIPITATION_COL] > 0] [cite: 231]

        annual_agg = annual_data_filtered.groupby([Config.STATION_NAME_COL, Config.YEAR_COL]).agg(
            precipitation_sum=(Config.PRECIPITATION_COL, 'sum'),
            meses_validos=(Config.PRECIPITATION_COL, 'count')
        [cite_start]).reset_index() [cite: 233-237]
        [cite_start]annual_agg.loc[annual_agg['meses_validos'] < 10, 'precipitation_sum'] = np.nan [cite: 238]
        [cite_start]df_anual_melted = annual_agg.rename(columns={'precipitation_sum': Config.PRECIPITATION_COL}) [cite: 239]
        
        # --- DESPLIEGUE DE PESTAÑAS ---
        tab_names = [
            "Bienvenida", "Distribución Espacial", "Gráficos", "Mapas Avanzados",
            "Análisis de Anomalías", "Análisis de extremos hid", "Estadísticas",
            "Análisis de Correlación", "Análisis ENSO", "Tendencias y Pronósticos",
            "Descargas", "Tabla de Estaciones"
        [cite_start]] [cite: 241-246]
        [cite_start]tabs = st.tabs(tab_names) [cite: 247]

        [cite_start]if df_anual_melted.empty or df_monthly_filtered.empty or gdf_filtered.empty: [cite: 248]
            with tabs[0]:
                [cite_start]st.warning("No hay datos disponibles para los filtros aplicados. Ajuste la selección.") [cite: 249-250]
            return

        with tabs[0]:
            [cite_start]display_welcome_tab() [cite: 253]
        with tabs[1]:
            [cite_start]display_spatial_distribution_tab(gdf_filtered, stations_for_analysis, df_anual_melted, df_monthly_filtered) [cite: 255]
        with tabs[2]:
            [cite_start]display_graphs_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis, gdf_filtered) [cite: 258-259]
        with tabs[3]:
            [cite_start]display_advanced_maps_tab(gdf_filtered, stations_for_analysis, df_anual_melted, df_monthly_filtered) [cite: 262, 261]
        with tabs[4]:
            [cite_start]display_anomalies_tab(st.session_state.df_long, df_monthly_filtered, stations_for_analysis) [cite: 264]
        with tabs[5]:
            [cite_start]display_drought_analysis_tab(df_monthly_filtered, gdf_filtered, stations_for_analysis) [cite: 266]
        with tabs[6]:
            [cite_start]display_stats_tab(st.session_state.df_long, df_anual_melted, df_monthly_filtered, stations_for_analysis, gdf_filtered) [cite: 269, 271]
        with tabs[7]:
            [cite_start]display_correlation_tab(df_monthly_filtered, stations_for_analysis) [cite: 272]
        with tabs[8]:
            [cite_start]display_enso_tab(df_monthly_filtered, st.session_state.df_enso, gdf_filtered, stations_for_analysis) [cite: 274, 276]
        with tabs[9]:
            [cite_start]display_trends_and_forecast_tab(df_anual_melted, st.session_state.df_monthly_processed, stations_for_analysis) [cite: 277, 279]
        with tabs[10]:
            [cite_start]display_downloads_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis) [cite: 280]
        with tabs[11]:
            [cite_start]display_station_table_tab(gdf_filtered, df_anual_melted, stations_for_analysis) [cite: 283]

    else:
        [cite_start]display_welcome_tab() [cite: 284]
        [cite_start]st.info("Para comenzar, por favor cargue los 3 archivos requeridos en el panel de la izquierda y haga clic en 'Procesar y Almacenar Datos'.") [cite: 285-286]

if __name__ == "__main__":
    main()
