"""
===============================================================================
MÓDULO ROBUSTO PARA DATOS RUIDOSOS / 1 ENSAYO (IC50) - FIX BOUNDS
===============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# Configuración de estilo gráfico para publicación
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="ticks", font="DejaVu Sans", font_scale=1.1)


def sigmoidal_4pl(x, bottom, top, ic50, hill_slope):
    """Ecuación Logística de 4 Parámetros (4PL)."""
    log_x = np.log10(x)
    log_ic50 = np.log10(ic50)
    return bottom + (top - bottom) / (1.0 + 10.0 ** ((log_ic50 - log_x) * hill_slope))


def fit_ic50_robusto(concentrations, response_means):
    """
    Ajuste 4PL optimizado para datos con outliers o alta variabilidad.
    Garantiza que p0 siempre esté dentro de los límites (bounds).
    """
    # 1. Filtrar concentraciones <= 0 para evitar desbordamiento logarítmico
    mask = concentrations > 0
    x_data = concentrations[mask]
    y_data = response_means[mask]

    # Definir límites amplios pero estables
    # [Bottom_min, Top_min, IC50_min, Hill_min], [Bottom_max, Top_max, IC50_max, Hill_max]
    bounds = (
        [0.0, 40.0, 1e-5, 0.05],
        [60.0, 150.0, 1e6, 20.0]
    )

    # 2. Asegurar dinámicamente que p0 caiga DENTRO de los bounds
    p0_bottom = np.clip(min(y_data), bounds[0][0] + 0.1, bounds[1][0] - 0.1)
    p0_top = np.clip(max(y_data), bounds[0][1] + 0.1, bounds[1][1] - 0.1)
    p0_ic50 = np.clip(np.median(x_data), bounds[0][2] + 0.1, bounds[1][2] - 0.1)
    p0_hill = 1.0

    p0 = [p0_bottom, p0_top, p0_ic50, p0_hill]

    try:
        popt, pcov = curve_fit(
            sigmoidal_4pl, x_data, y_data, p0=p0, bounds=bounds, maxfev=20000
        )
        bottom, top, ec50_rel, hill_slope = popt
        perr = np.sqrt(np.diag(pcov))

        # Cálculo del IC50 absoluto en Y = 50%
        target_y = 50.0
        if bottom < target_y < top:
            log_ratio = np.log10(((top - bottom) / (target_y - bottom)) - 1)
            ic50_abs = ec50_rel / (10 ** (log_ratio / hill_slope))
        else:
            ic50_abs = ec50_rel

        return popt, perr, ic50_abs, x_data, y_data
    except Exception as e:
        print(f"\n[!] Error en el ajuste: {e}")
        return None, None, None, x_data, y_data


def main():
    print("=" * 65)
    print("   AJUSTE ROBUSTO DE CURVA IC50 PARA ENSAYO ÚNICO / DATOS RUIDOSOS")
    print("=" * 65)

    csv_path = input("Ruta del archivo CSV [ej: datos.csv]: ").strip()
    if not os.path.exists(csv_path):
        print(f"[!] Error: El archivo '{csv_path}' no existe.")
        return

    df = pd.read_csv(csv_path)
    conc_col = df.columns[0]
    data_cols = df.columns[1:]

    # Transformar datos a formato Tidy
    melted = df.melt(id_vars=[conc_col], value_vars=data_cols, var_name='Replicado', value_name='Respuesta')
    melted[conc_col] = pd.to_numeric(melted[conc_col], errors='coerce')
    melted['Respuesta'] = pd.to_numeric(melted['Respuesta'], errors='coerce')
    melted = melted.dropna()

    # Promedios y errores por concentración
    stats = melted.groupby(conc_col)['Respuesta'].agg(
        Mean='mean',
        SEM=lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0
    ).reset_index()

    concentrations = stats[conc_col].values
    means = stats['Mean'].values

    # Ajuste Robusto
    popt, perr, ic50_abs, x_clean, y_clean = fit_ic50_robusto(concentrations, means)

    if popt is not None:
        bottom, top, ec50_rel, hill = popt
        print("\n" + "=" * 50)
        print("   RESULTADOS DEL AJUSTE")
        print("=" * 50)
        print(f"  IC50 Absoluto (50%): {ic50_abs:.3f} µg/mL")
        print(f"  Emax (Top)          : {top:.2f}%")
        print(f"  Baseline (Bottom)   : {bottom:.2f}%")
        print("=" * 50)

    # -------------------------------------------------------------------------
    # GENERACIÓN DE GRÁFICO DE LA CURVA
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

    # Puntos experimentales filtrados (sin conc = 0)
    stats_valid = stats[stats[conc_col] > 0]
    ax.errorbar(
        stats_valid[conc_col],
        stats_valid['Mean'],
        yerr=stats_valid['SEM'],
        fmt='o', color='#2b5c8f', ecolor='#2b5c8f', elinewidth=1.5, capsize=3,
        markersize=6, label='Datos Experimentales'
    )

    if popt is not None:
        x_min, x_max = np.min(x_clean), np.max(x_clean)
        x_curve = np.logspace(np.log10(x_min), np.log10(x_max), 300)
        y_curve = sigmoidal_4pl(x_curve, *popt)

        # Trazar curva
        ax.plot(x_curve, y_curve, color='#d9534f', linewidth=2, label='Ajuste 4PL Robusto')
        ax.set_xscale('log')

        # Intersección en 50%
        ax.axvline(x=ic50_abs, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=50.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        # Etiqueta IC50
        text_str = f'$IC_{{50}} = {ic50_abs:.2f}\\ \\mu g/mL$'
        ax.text(
            0.08, 0.15, text_str, transform=ax.transAxes,
            fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        )

    ax.set_title('Curva Dosis-Respuesta (IC50)', fontweight='bold', fontsize=12, pad=10)
    ax.set_xlabel('Concentración [µg/mL] (Escala Log)', fontweight='bold')
    ax.set_ylabel('Inhibición (%)', fontweight='bold')
    ax.set_ylim(-5, 108)
    sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout()
    plt.savefig('figura_curva_ic50_robusta.png', dpi=300, bbox_inches='tight')
    plt.savefig('figura_curva_ic50_robusta.pdf', format='pdf', dpi=600, bbox_inches='tight')
    print("\n[✓] Figura generada y guardada exitosamente como 'figura_curva_ic50_robusta.png'")


if __name__ == '__main__':
    main()