import pandas as pd
import numpy as np
from modules.utils import standardize_numeric_column

def test_standardize_numeric_column():
    """
    Verifica que la función convierte correctamente una serie
    con comas como decimales y maneja errores.
    """
    # 1. Datos de entrada de ejemplo
    input_series = pd.Series(['1.23', '4,56', '7', 'no-es-numero', '8,90'])

    # 2. Resultado esperado
    expected_series = pd.Series([1.23, 4.56, 7.0, np.nan, 8.90])

    # 3. Ejecutar la función
    result_series = standardize_numeric_column(input_series)

    # 4. Verificar que el resultado es el esperado
    # Usamos assert_series_equal para una comparación robusta de Series de pandas
    pd.testing.assert_series_equal(result_series, expected_series)
