from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from helios.sim.mmi import (
    simulate,
    calibrate_input_phases_genetic,
    calibrate_n_core_and_phases,
    plot_mmi_interactive,
)


def main() -> None:
    print("HELIOS MMI imports: OK")
    print(simulate.__name__)
    print(calibrate_input_phases_genetic.__name__)
    print(calibrate_n_core_and_phases.__name__)
    print(plot_mmi_interactive.__name__)

    page_path = Path(r"D:/PhD-Theory/web/pages/04_Multi_Mode_Interferometer.py")
    spec = spec_from_file_location("mmi_page", page_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to build import spec for Streamlit page")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    print("Streamlit page import: OK")


if __name__ == "__main__":
    main()
