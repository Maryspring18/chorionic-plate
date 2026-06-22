import subprocess
import pandas as pd
import sys
from pathlib import Path


root = "X:/"

CSV_PATH = Path(root + "intermediate/2023-sex-specific/chorionic-segmentations/sample_list.csv")
GROW_TREE_SCRIPT = "Generate_tree.py"
SIMULATE_FLOW_SCRIPT = "simulate_flow.py"


def load_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, dtype=str)
    df.columns = df.columns.str.strip()
    return df


def save_csv(df: pd.DataFrame):
    df.to_csv(CSV_PATH, index=False)

def run_fetoflow(cmd):
    captured_lines = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8") as proc:
        for line in proc.stdout:
            print(line, end="")  # print live to terminal
            captured_lines.append(line)
    return proc.returncode, captured_lines


def run_generate_tree(cmd):
    captured_lines = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8") as proc:
        for line in proc.stdout:
            print(line, end="")  # print live to terminal
            captured_lines.append(line)
    return proc.returncode, captured_lines

def main():
    df = load_csv()

    for idx, row in df.iterrows():
        sample_id = row["Code"]
        print(f"\n{'='*50}\nProcessing sample: {sample_id}")
        sample_id   = row["Code"].strip()
        weight      = row["Weight"].strip()
        tree_grown  = row["Tree_Grown"].strip().upper()
        inlet_known = row["Inlet_detected"].strip().upper()
        inlet1x      = str(row.get("Inlet1x", "")).strip()
        inlet1y      = str(row.get("Inlet1y", "")).strip()
        inlet2x      = str(row.get("Inlet2x", "")).strip()
        inlet2y      = str(row.get("Inlet2y", "")).strip()
        if tree_grown != 'Y':
            print(f"  Growing tree for {sample_id}...")

            returncode ,output_lines = run_generate_tree(
               [sys.executable, GROW_TREE_SCRIPT,
                "--sample", sample_id,
                "--weight", weight,
                '--inletknown', inlet_known,
                "--inlet1_x", inlet1x,
                "--inlet1_y", inlet1y,
                "--inlet2_x", inlet2x,
                "--inlet2_y", inlet2y],
            )
            #df.at[idx, "Inlet1"] = inlet1
            #df.at[idx, "Inlet2"] = inlet2
            #df.at[idx, "Inlet_detected"] = "Y"
            if returncode != 0:
                print(f"  [ERROR] generate_tree.py failed for {sample_id}.")
                continue

            if inlet_known == "N":
                for line in output_lines:
                    if line.startswith("INLET1X:"):
                        df.at[idx, "Inlet1x"] = line.split("INLET1X:")[1].strip()
                    elif line.startswith("INLET1Y:"):
                        df.at[idx, "Inlet1y"] = line.split("INLET1Y:")[1].strip()
                    elif line.startswith("INLET2X:"):
                        df.at[idx, "Inlet2x"] = line.split("INLET2X:")[1].strip()
                    elif line.startswith("INLET2Y:"):
                        df.at[idx, "Inlet2y"] = line.split("INLET2Y:")[1].strip()
                df.at[idx, "Inlet_detected"] = "Y"
                df.at[idx, "Tree_Grown"] = "Y"

                save_csv(df)
        folder_path = Path(root + "intermediate/2023-sex-specific/chorionic-segmentations/" + sample_id + "/outputs_grow_tree/")

        if folder_path.is_dir():
            returncode, output_lines = run_fetoflow(
                [sys.executable, SIMULATE_FLOW_SCRIPT,
                 "--sample", sample_id]
            )
        if returncode != 0:
            print(f"  [ERROR] simulate_flow.py failed for {sample_id}.")
            continue

    print("\nAll samples processed.")


if __name__ == "__main__":
    main()
