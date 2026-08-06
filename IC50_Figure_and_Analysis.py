"""
===============================================================================
PLANTILLA DE ANÁLISIS DE DOSIS-RESPUESTA Y GENERACIÓN DE FIGURAS PARA PUBLICACIÓN
===============================================================================
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.optimize import curve_fit

# Silenciar advertencias de fuentes en Matplotlib
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuración de estilo global para publicación científica
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="ticks", font="DejaVu Sans", font_scale=1.1)


def sigmoidal_4pl(x, bottom, top, ic50, hill_slope):
    """Ecuación Logística de 4 Parámetros (4PL)."""
    log_x = np.log10(x)
    log_ic50 = np.log10(ic50)
    return bottom + (top - bottom) / (1.0 + 10.0 ** ((log_ic50 - log_x) * hill_slope))


def fit_ic50(concentrations, response_means, y_max_limit=100.0):
    """
    Ajusta el modelo 4PL y calcula el IC50 absoluto en 50% / 0.5.
    Maneja datos sin necesidad de incluir la concentración 0.
    """
    # Filtrar concentraciones válidas (> 0)
    mask = concentrations > 0
    x_data = concentrations[mask]
    y_data = response_means[mask]

    p0 = [min(y_data), max(y_data), np.median(x_data), 1.0]
    bounds = (
        [0, 0, 1e-12, -10.0],
        [max(y_data)*1.5, max(y_data)*2, 1e6, 10.0]
    )

    try:
        popt, pcov = curve_fit(
            sigmoidal_4pl, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000
        )
        bottom, top, ec50_rel, hill_slope = popt
        perr = np.sqrt(np.diag(pcov))

        residuals = y_data - sigmoidal_4pl(x_data, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1 - (ss_res / ss_tot)

        target_y = 50.0 if y_max_limit == 100.0 else 0.5

        if bottom < target_y < top:
            log_ratio = np.log10(((top - bottom) / (target_y - bottom)) - 1)
            ic50_abs = ec50_rel / (10 ** (log_ratio / hill_slope))
        else:
            ic50_abs = ec50_rel

        return popt, perr, r_squared, ic50_abs
    except Exception as e:
        print(f"\n[!] Error en el ajuste de curva: {e}")
        return None, None, None, None


def crear_degradado(color_base, n_pasos):
    """Genera una paleta de degradado de N pasos partiendo de un color base."""
    try:
        rgb = mcolors.to_rgb(color_base)
    except ValueError:
        print(f"[!] Color '{color_base}' no reconocido. Usando azul (#2b5c8f) por defecto.")
        rgb = mcolors.to_rgb("#2b5c8f")

    colores = [mcolors.to_hex(np.array(rgb) * (0.35 + 0.65 * (i + 1) / n_pasos)) for i in range(n_pasos)]
    return colores


def main():
    print("=" * 65)
    print("   ANÁLISIS DE DOSIS-RESPUESTA, IC50 Y GENERACIÓN DE FIGURAS")
    print("=" * 65)

    # 1. Selección del tipo de experimento
    print("\nSelecciona el tipo de análisis:")
    print(" 1) Ensayo Único (1 experimento con 'n' réplicas, sin necesidad de conc. 0)")
    print(" 2) Múltiples Ensayos (Múltiples experimentos independientes)")
    tipo_exp = input("Opción [1-2, por defecto 1]: ").strip()

    csv_path = input("\nRuta del archivo CSV [ej: datos.csv]: ").strip()
    if not os.path.exists(csv_path):
        print(f"[!] Error: El archivo '{csv_path}' no existe.")
        return

    # Carga de datos
    df = pd.read_csv(csv_path)
    conc_col = df.columns[0]
    print(f"\n[✓] Columna de concentración detectada: '{conc_col}'")

    # 2. Selección de unidades de concentración
    print("\nSelecciona o escribe las unidades de concentración:")
    print(" 1) µM")
    print(" 2) mg/mL")
    print(" 3) µg/mL")
    print(" 4) Otra (Escribir personalizada)")
    opcion_u = input("Opción [1-4, por defecto 1]: ").strip()

    if opcion_u == '2':
        unidades = "mg/mL"
    elif opcion_u == '3':
        unidades = "µg/mL"
    elif opcion_u == '4':
        unidades = input("Escribe la unidad (ej. mM, ng/mL): ").strip()
    else:
        unidades = "µM"

    # 3. Límite máximo en Y
    print("\nSelecciona la escala límite para el eje Y:")
    print(" 1) Detección automática (100 si son %, 1.0 si son fracciones)")
    print(" 2) Fijar en Y = 100 (%)")
    print(" 3) Fijar en Y = 1.0 (Fracción/Absorbancia)")
    opcion_y = input("Opción [1-3, por defecto 1]: ").strip()

    # 4. Estilo de color
    print("\nSelecciona el color base para el degradado de las barras:")
    print(" 1) Degradar un color específico (ej: #2b5c8f, teal, crimson, purple)")
    print(" 2) Degradar un color aleatorio (Random)")
    print(" 3) Degradado Azul clásico por defecto")
    opcion_c = input("Opción [1-3, por defecto 3]: ").strip()

    color_base_elegido = "#2b5c8f"
    if opcion_c == '1':
        color_user = input("Ingresa el color base HEX o nombre [ej. #2b5c8f, teal, crimson]: ").strip()
        if color_user:
            color_base_elegido = color_user
    elif opcion_c == '2':
        color_base_elegido = f"#{np.random.randint(0x222222, 0xDDDDDD):06x}"
        print(f"[✓] Color aleatorio seleccionado: {color_base_elegido}")

    # Procesamiento según tipo de experimento
    data_cols = df.columns[1:]

    if tipo_exp == '2':
        try:
            n_replicas = int(input("\nNúmero de réplicas/repeticiones por ensayo (n) [ej. 3]: "))
            n_ensayos = int(input("Número de ensayos independientes [ej. 1 o 3]: "))
        except ValueError:
            print("[!] Entrada inválida. Se procesarán todas las columnas encontradas.")
    else:
        print(f"\n[✓] Modo Ensayo Único detectado: procesando {len(data_cols)} réplicas...")

    melted_df = df.melt(id_vars=[conc_col], value_vars=data_cols, var_name='Replicado', value_name='Respuesta')
    melted_df[conc_col] = pd.to_numeric(melted_df[conc_col], errors='coerce')
    melted_df['Respuesta'] = pd.to_numeric(melted_df['Respuesta'], errors='coerce')
    melted_df = melted_df.dropna()

    stats_df = melted_df.groupby(conc_col)['Respuesta'].agg(
        Mean='mean',
        SD='std',
        SEM=lambda x: x.std() / np.sqrt(len(x)),
        Count='count'
    ).reset_index()

    print("\n" + "-"*50)
    print("RESUMEN ESTADÍSTICO POR CONCENTRACIÓN:")
    print("-" * 50)
    print(stats_df.to_string(index=False))

    concentrations = stats_df[conc_col].values
    means = stats_df['Mean'].values

    # Determinar el límite del eje Y
    max_val_y = np.max(melted_df['Respuesta'].values)
    if opcion_y == '2':
        y_max_limit = 100.0
    elif opcion_y == '3':
        y_max_limit = 1.0
    else:
        y_max_limit = 100.0 if max_val_y > 2.0 else 1.0

    popt, perr, r2, ic50_abs = fit_ic50(concentrations, means, y_max_limit)

    if popt is not None:
        bottom, top, ec50_rel, hill = popt
        ic50_err = perr[2]
        print("\n" + "="*50)
        print("   RESULTADOS DEL AJUSTE SIGMOIDAL (4PL / IC50)")
        print("="*50)
        print(f"  IC50 Absoluto (50%): {ic50_abs:.4f} {unidades}")
        print(f"  EC50 Relativo      : {ec50_rel:.4f} ± {ic50_err:.4f} {unidades}")
        print(f"  Top (Emax)         : {top:.2f}")
        print(f"  Bottom             : {bottom:.2f}")
        print(f"  Hill Slope         : {hill:.3f}")
        print(f"  R²                 : {r2:.4f}")
        print("="*50)

    lbl_y = "Inhibición (%)" if y_max_limit == 100.0 else "Inhibición (Fracción)"

    # -------------------------------------------------------------------------
    # FIGURA 1: BAR PLOT INDEPENDIENTE
    # -------------------------------------------------------------------------
    fig1, ax_bar = plt.subplots(figsize=(5.5, 4.5), dpi=300)

    num_barras = len(stats_df)
    paleta_degradada = crear_degradado(color_base_elegido, num_barras)

    sns.barplot(
        data=melted_df, x=conc_col, y='Respuesta',
        capsize=0.15, errorbar='sd', palette=paleta_degradada,
        ax=ax_bar, alpha=0.9, edgecolor='black', linewidth=1
    )

    sns.stripplot(
        data=melted_df, x=conc_col, y='Respuesta',
        color='black', alpha=0.7, jitter=0.15, size=5, ax=ax_bar
    )

    ax_bar.tick_params(axis='x', rotation=45)
    for label in ax_bar.get_xticklabels():
        label.set_ha('right')
        label.set_rotation_mode('anchor')

    ax_bar.set_title('Inhibición por Concentración', fontweight='bold', fontsize=12, pad=10)
    ax_bar.set_xlabel(f'Concentración ({unidades})', fontweight='bold')
    ax_bar.set_ylabel(lbl_y, fontweight='bold')
    ax_bar.set_ylim(0, y_max_limit * 1.05)
    sns.despine(ax=ax_bar, top=True, right=True)

    plt.tight_layout()
    plt.savefig('figura_barras.pdf', format='pdf', dpi=600, bbox_inches='tight')
    plt.savefig('figura_barras.png', format='png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # FIGURA 2: LINE PLOT / CURVA IC50 INDEPENDIENTE
    # -------------------------------------------------------------------------
    fig2, ax_line = plt.subplots(figsize=(5.5, 4.5), dpi=300)

    ax_line.errorbar(
        stats_df[conc_col], stats_df['Mean'], yerr=stats_df['SEM'],
        fmt='o', color=color_base_elegido, ecolor=color_base_elegido, elinewidth=1.5, capsize=3,
        capthick=1.5, markersize=6
    )

    if popt is not None:
        conc_non_zero = concentrations[concentrations > 0]
        x_min, x_max = np.min(conc_non_zero), np.max(conc_non_zero)
        x_curve = np.logspace(np.log10(x_min) - 0.5, np.log10(x_max) + 0.5, 300)
        y_curve = sigmoidal_4pl(x_curve, *popt)

        ax_line.plot(x_curve, y_curve, color='#d9534f', linewidth=2)
        ax_line.set_xscale('log')

        target_50 = 50.0 if y_max_limit == 100.0 else 0.5

        ax_line.axvline(x=ic50_abs, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax_line.axhline(y=target_50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        text_str = f'$IC_{{50}} = {ic50_abs:.3f}\\ {unidades}$'
        ax_line.text(
            0.08, 0.15, text_str, transform=ax_line.transAxes,
            fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        )

    ax_line.set_title('Curva Dosis-Respuesta (IC50)', fontweight='bold', fontsize=12, pad=10)
    ax_line.set_xlabel(f'Concentración [{unidades}] (Escala Log)', fontweight='bold')
    ax_line.set_ylabel(lbl_y, fontweight='bold')
    ax_line.set_ylim(0, y_max_limit * 1.05)
    sns.despine(ax=ax_line, top=True, right=True)

    plt.tight_layout()
    plt.savefig('figura_curva_ic50.pdf', format='pdf', dpi=600, bbox_inches='tight')
    plt.savefig('figura_curva_ic50.png', format='png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print(f"\n[✓] Ambas figuras guardadas exitosamente:")
    print(f"    - figura_barras.pdf / .png")
    print(f"    - figura_curva_ic50.pdf / .png")


if __name__ == '__main__':
    main()