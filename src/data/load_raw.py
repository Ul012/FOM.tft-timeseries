# src/data/load_raw.py
# Zweck: Lädt Raw-Daten basierend auf dataset-spezifischer YAML-Config

from pathlib import Path
import pandas as pd

from src.config import BASE_DIR
from src.utils.load_dataset_config import load_dataset_config

# Lade Config
_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]


def load_single_file(file_config: dict) -> pd.DataFrame:
    """
    Lädt eine einzelne CSV-Datei.

    Args:
        file_config: Dict mit 'path' und optionalen 'columns'

    Returns:
        DataFrame
    """
    file_path = Path(file_config["path"])

    if not file_path.exists():
        raise FileNotFoundError(f"Raw-Datei nicht gefunden: {file_path}")

    df = pd.read_csv(file_path)

    # Spalten-Validierung (falls definiert)
    if "columns" in file_config:
        expected_cols = file_config["columns"]
        missing = set(expected_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"Fehlende Spalten in {file_path.name}: {missing}\n"
                f"Erwartet: {expected_cols}\n"
                f"Vorhanden: {df.columns.tolist()}"
            )

    return df


def load_and_merge_multiple(raw_data_config: dict) -> pd.DataFrame:
    """
    Lädt mehrere Dateien und merged sie.

    Args:
        raw_data_config: 'raw_data' Section aus YAML

    Returns:
        Merged DataFrame
    """
    files = raw_data_config["files"]
    merge_config = raw_data_config["merge"]

    # Hauptdatei laden
    main_file = next(f for f in files if f["role"] == "main")
    df_main = load_single_file(main_file)
    print(f"[load_raw] Main-Datei: {Path(main_file['path']).name} (Shape: {df_main.shape})")

    # Feature-Dateien mergen
    for file_config in files:
        if file_config["role"] == "features":
            df_features = load_single_file(file_config)
            print(f"[load_raw] Feature-Datei: {Path(file_config['path']).name} (Shape: {df_features.shape})")

            # Duplikate entfernen (falls definiert)
            if "drop_from_right" in merge_config:
                df_features = df_features.drop(columns=merge_config["drop_from_right"], errors="ignore")

            # Merge durchführen
            df_main = df_main.merge(
                df_features,
                on=merge_config.get("merge_on", merge_config.get("on")),
                how=merge_config.get("how", "left")
            )
            print(f"[load_raw] Nach Merge: {df_main.shape}")

    return df_main


def main() -> None:
    """
    Lädt Raw-Daten basierend auf raw_data-Config und speichert als train_raw.parquet.
    """

    # raw_data Section aus YAML
    raw_data_config = _dataset_config.get("raw_data")

    if not raw_data_config:
        raise ValueError(
            f"'raw_data' Section fehlt in Dataset-Config für '{_dataset_name}'.\n"
            "Bitte in configs/datasets/{_dataset_name}.yaml unter 'raw_data' definieren."
        )

    print(f"[load_raw] Dataset: {_dataset_name}")
    print(f"[load_raw] Type: {raw_data_config['type']}")

    # Laden (single vs. multiple)
    data_type = raw_data_config["type"]

    if data_type == "single_file":
        df = load_single_file(raw_data_config["files"][0])
    elif data_type == "multiple_files":
        df = load_and_merge_multiple(raw_data_config)
    else:
        raise ValueError(
            f"Unbekannter raw_data.type: '{data_type}'\n"
            "Erlaubt: 'single_file' oder 'multiple_files'"
        )

    # Speichern
    output_dir = BASE_DIR / "data" / "interim" / _dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "train_raw.parquet"
    df.to_parquet(output_path, index=False)

    print(f"\n✓ Raw-Daten geladen und gespeichert: {output_path}")
    print(f"  Zeilen: {len(df):,}")
    print(f"  Spalten: {len(df.columns)}")

    # Zeitraum anzeigen (falls time_col vorhanden)
    schema = _dataset_config.get("schema", {})
    time_col = schema.get("time_col")
    if time_col and time_col in df.columns:
        print(f"  Zeitraum: {df[time_col].min()} bis {df[time_col].max()}")

    # Zeitreihen-Anzahl anzeigen (falls id_cols vorhanden)
    id_cols = schema.get("id_cols", [])
    if id_cols and all(col in df.columns for col in id_cols):
        n_series = df.groupby(id_cols).ngroups
        print(f"  Zeitreihen: {n_series:,}")


if __name__ == "__main__":
    main()

# Aufruf einzeln:
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.load_raw
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.load_raw
# Via Pipeline:
#   python -m src.pipeline --dataset configs/datasets/walmart.yaml --steps preprocessing
