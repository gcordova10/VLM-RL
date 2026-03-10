import os
from tensorboard.backend.event_processing import event_accumulator

baseline_path = "/media/nemesis/disco4tb/Documents/VLM-RL/tensorboard/CLIPRewardedSAC_20250930_154046_idvlm_rl/events.out.tfevents.1759239646.nemesis.48840.0"
ours_path = "/media/nemesis/disco4tb/Documents_VLM-RL/investigacion/VLM-RL-PRIVATE/tensorboard/CLIPRewardedSAC_20260212_082504_idvlm_rl/events.out.tfevents.1770881104.nemesis.3270074.0"

def list_tags(path, name):
    print(f"\n--- Tags for {name} ---")
    try:
        ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        tags = ea.Tags()['scalars']
        for tag in sorted(tags):
            print(tag)
    except Exception as e:
        print(f"Error loading {name}: {e}")

if __name__ == "__main__":
    list_tags(baseline_path, "Baseline")
    list_tags(ours_path, "Ours")
