import subprocess
import pandas as pd
import sys
import re
from pathlib import Path


root = "X:/"

CSV_PATH = Path(root + "intermediate/2023-sex-specific/chorionic-segmentations/sample_list.csv")
RESULTS_CSV_PATH = CSV_PATH.parent / "results.csv"
GROW_TREE_SCRIPT = "Generate_tree.py"
SIMULATE_FLOW_SCRIPT = "simulate_flow.py"

RESULTS_COLUMNS = [
    "Code",
    "Area_mm2",
    "X_length_mm",
    "Y_length_mm",
    "Thickness_mm",
    "Final_total_volume_mm3",
    "Inlet_node_0_pressure",
    "Inlet_node_1_pressure",
    "Total_vessel_volume_mm3",
    "Arterial_vessel_volume_mm3",
]


def load_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, dtype=str)
    df.columns = df.columns.str.strip()
    return df


def save_csv(df: pd.DataFrame):
    df.to_csv(CSV_PATH, index=False)


def parse_output_lines(output_lines: list[str]) -> dict:
    """Extract metrics from captured stdout lines using regex."""
    metrics = {col: None for col in RESULTS_COLUMNS if col != "Code"}
    inlet_pressures = []

    for line in output_lines:
        line = line.strip()
        # Area in mm2: 22983.08
        m = re.match(r"Area in mm2: \s*([\d.]+)", line)
        if m:
            metrics["Area_mm2"] = float(m.group(1))

        # X length is 93.91 and y length is 93.21, Calculated Thickness is 28.28
        m = re.match(
            r"X length is ([\d.]+) and y length is ([\d.]+),\s*Calculated Thickness is ([\d.]+)",
            line,
        )
        if m:
            metrics["X_length_mm"] = float(m.group(1))
            metrics["Y_length_mm"] = float(m.group(2))
            metrics["Thickness_mm"] = float(m.group(3))

        # Final Total volume  = 453494.87
        m = re.match(r"Final total volume\s*=\s*([\d.]+)", line)
        if m:
            metrics["Final_total_volume_mm3"] = float(m.group(1))

        # Inlet node, 0, has pressure 5153.94 and Inlet edge, 0, has flow 2083.35
        m = re.match(r"Inlet node,\s*\d+,\s*has pressure\s*([\d.]+)", line)
        if m:
            inlet_pressures.append(float(m.group(1)))

        # Total vessel volume is 26937.87, arterial vessel volume is 9282.56
        m = re.match(
            r"Total vessel volume is ([\d.]+),\s*arterial vessel volume is ([\d.]+)",
            line,
        )
        if m:
            metrics["Total_vessel_volume_mm3"] = float(m.group(1))
            metrics["Arterial_vessel_volume_mm3"] = float(m.group(2))

    if len(inlet_pressures) >= 1:
        metrics["Inlet_node_0_pressure"] = inlet_pressures[0]
    if len(inlet_pressures) >= 2:
        metrics["Inlet_node_1_pressure"] = inlet_pressures[1]

    return metrics


def save_results(sample_id: str, metrics: dict):
    """Insert or overwrite a row in the results CSV for the given sample."""
    row = {"Code": sample_id, **metrics}
    row_df = pd.DataFrame([row], columns=RESULTS_COLUMNS)

    if RESULTS_CSV_PATH.exists():
        existing = pd.read_csv(RESULTS_CSV_PATH, dtype=str)
        existing = existing[existing["Code"] != sample_id]  # drop old row if present
        updated = pd.concat([existing, row_df], ignore_index=True)
        updated.to_csv(RESULTS_CSV_PATH, index=False)
    else:
        row_df.to_csv(RESULTS_CSV_PATH, index=False)

    print(f"  [INFO] Results saved for {sample_id} → {RESULTS_CSV_PATH}")


def run_fetoflow(cmd):
    captured_lines = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8") as proc:
        for line in proc.stdout:
            print(line, end="")
            captured_lines.append(line)
    return proc.returncode, captured_lines


def run_generate_tree(cmd):
    captured_lines = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8") as proc:
        for line in proc.stdout:
            print(line, end="")
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
        inlet1x     = str(row.get("Inlet1x", "")).strip()
        inlet1y     = str(row.get("Inlet1y", "")).strip()
        inlet2x     = str(row.get("Inlet2x", "")).strip()
        inlet2y     = str(row.get("Inlet2y", "")).strip()

        if tree_grown != "Y":
            print(f"  Growing tree for {sample_id}...")

            returncode, output_lines = run_generate_tree(
                [sys.executable, GROW_TREE_SCRIPT,
                 "--sample", sample_id,
                 "--weight", weight,
                 "--inletknown", inlet_known,
                 "--inlet1_x", inlet1x,
                 "--inlet1_y", inlet1y,
                 "--inlet2_x", inlet2x,
                 "--inlet2_y", inlet2y],
            )

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

        folder_path = Path(
            root + "intermediate/2023-sex-specific/chorionic-segmentations/"
            + sample_id + "/outputs_grow_tree/"
        )

        if folder_path.is_dir():
            returncode, output_lines_FF = run_fetoflow(
                [sys.executable, SIMULATE_FLOW_SCRIPT,
                 "--sample", sample_id]
            )

            if returncode != 0:
                print(f"  [ERROR] simulate_flow.py failed for {sample_id}.")
                continue
            output_lines.extend(output_lines_FF)
            metrics = parse_output_lines(output_lines)
            save_results(sample_id, metrics)

    print("\nAll samples processed.")


if __name__ == "__main__":
    main()