import os

with open('scripts/generate_filtered_comparison.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Insertamos el ffill justo antes de la conversion a JSON
    if "df = df_ours.join(df_base" in line:
        new_lines.append(line)
        new_lines.append("df = df.sort_index().ffill().fillna(0)  # PARCHE: Rellenar nulos para intersección\n")
    elif "df = df.sort_index().interpolate" in line:
        continue # Quitamos la linea vieja de interpolacion si existe
    else:
        new_lines.append(line)

with open('scripts/generate_filtered_comparison.py', 'w') as f:
    f.writelines(new_lines)
