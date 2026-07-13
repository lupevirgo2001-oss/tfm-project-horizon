# TFM FX Hedging Pipeline

Proyecto base para el TFM **AI-Driven Dynamic Hedging**.

## Pipeline
1. Descarga y alinea series diarias.
2. Construye features y target de volatilidad mensual.
3. Simula exposición diaria y la agrega a exposición neta mensual.
4. Calcula benchmarks EWMA, GARCH y OLS.
5. Entrena LSTM y GRU con walk-forward.
6. Convierte predicción de VaR mensual en hedge ratio.
7. Corre backtest con forwards mensuales y costes de transacción.
8. Genera tablas y gráficos.

## Series incluidas
- EUR/USD
- VIX
- DXY
- US 10Y
- Euro area 10Y diario ECB (proxy homogénea a largo plazo)
- WTI

## Nota sobre la serie euro 10Y
El código usa por defecto la serie diaria del ECB:
`YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y`

Si necesitáis sustituirla por Bund alemán 10Y, ajustad `src/data_loader.py`.

## Instalación
```bash
pip install -r requirements.txt
```

## Ejecución
```bash
python main.py
```
