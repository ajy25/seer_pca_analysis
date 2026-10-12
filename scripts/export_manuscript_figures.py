from pathlib import Path
from shutil import copy2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
MANUSCRIPT_FIGURE_DIR = PROJECT_ROOT / "output" / "manuscript_figures"
FIGURE_FILES = {
    "Fig 1.png": "01_os_km.png",
    "Fig 2.png": "02_pcss_km.png",
    "Fig 3.png": "23_adjusted_race_era_survival.png",
    "Supp Fig 1.png": "03_os_km_by_race.png",
    "Supp Fig 2.png": "04_pcss_km_by_race.png",
    "Supp Fig 3.png": "05_os_km_race_era.png",
    "Supp Fig 4.png": "06_pcss_km_race_era.png",
    "Supp Fig 5.png": "07_os_km_stage_race_era.png",
    "Supp Fig 6.png": "08_pcss_km_stage_race_era.png",
    "Supp Fig 7.png": "09_os_km_white_black_by_era.png",
    "Supp Fig 8.png": "10_pcss_km_white_black_by_era.png",
}


def export_figure3(figure_dir=FIGURE_DIR, manuscript_dir=MANUSCRIPT_FIGURE_DIR):
    sources = [
        figure_dir / f"23_adjusted_race_era_survival.{extension}"
        for extension in ("png", "pdf")
    ]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Generate Figure 3 before exporting: {source}")
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        copy2(source, manuscript_dir / f"Fig 3{source.suffix}")


def main():
    expected_files = list(FIGURE_FILES.values()) + ["23_adjusted_race_era_survival.pdf"]
    missing = [name for name in expected_files if not (FIGURE_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Generate the missing figures before exporting: " + ", ".join(missing)
        )
    MANUSCRIPT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    export_figure3()
    for manuscript_name, source_name in FIGURE_FILES.items():
        if manuscript_name == "Fig 3.png":
            continue
        copy2(FIGURE_DIR / source_name, MANUSCRIPT_FIGURE_DIR / manuscript_name)
        print(f"Exported {manuscript_name} from {source_name}", flush=True)


if __name__ == "__main__":
    main()
