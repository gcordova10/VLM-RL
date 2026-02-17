import os
import pandas as pd
import glob

def find_best_checkpoint_match():
    # Valores del Paper (Tabla 2)
    paper_as = 19.3
    paper_td = 2028.2
    paper_sr = 0.93
    
    eval_dir = '/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/eval'
    summary_files = glob.glob(os.path.join(eval_dir, "*_summary.csv"))
    
    if not summary_files:
        print("❌ No se encontraron archivos de resumen en 'eval/'.")
        return

    print(f"🔍 Analizando {len(summary_files)} archivos de resumen en 'eval/'...")
    
    results = []
    for f in summary_files:
        try:
            df = pd.read_csv(f)
            # Buscamos la fila total
            row = df[df['episode'] == 'total']
            if row.empty: continue
            
            steps = os.path.basename(f).split('_')[1]
            
            as_val = float(row['speed_mean'].values[0])
            td_val = float(row['total_distance'].values[0])
            sr_val = float(row['success'].values[0])
            
            error = abs(as_val - paper_as) + (abs(td_val - paper_td) / 100) + (abs(sr_val - paper_sr) * 10)
            
            results.append({
                "Steps": steps,
                "AS": round(as_val, 2),
                "TD": round(td_val, 2),
                "SR": round(sr_val, 2),
                "Error": round(error, 2)
            })
        except:
            continue

    if results:
        df_results = pd.DataFrame(results).sort_values(by="Error")
        print("\n🏆 TOP 5 CHECKPOINTS QUE MÁS SE ACERCAN AL PAPER (TABLA 2):")
        print("======================================================================")
        print(df_results.head(5).to_string(index=False))
        print("======================================================================")
        
        best = df_results.iloc[0]
        print(f"\n💡 CONCLUSIÓN: El modelo de {best['Steps']} pasos es el candidato más probable.")
    else:
        print("No se encontraron datos comparables.")

if __name__ == "__main__":
    find_best_checkpoint_match()