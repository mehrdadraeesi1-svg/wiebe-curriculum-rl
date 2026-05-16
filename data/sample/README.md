# Sample Data

## Synthetic Demonstration Case

The file `generate_sample.py` creates a **single synthetic operating point**
for demonstration purposes. Run it with:

```bash
python data/sample/generate_sample.py
```

This generates `synthetic_demo.json` — a pressure trace computed analytically
from a Double Wiebe model with known parameters, allowing you to verify that
the framework correctly recovers those parameters.

## Experimental Dataset

The experimental dataset used in the paper (1,258 operating points from a
499.4 cm³ single-cylinder GDI engine) is **not included** in this repository.

The data originates from:

> Yuan H, Goyal H, Islam R, Giles K, Howson S, Lewis A, et al.
> *Thermodynamics-based data-driven combustion modelling for modern
> spark-ignition engines.* Energy 2024;313.
> https://doi.org/10.1016/j.energy.2024.134074

To use the framework with the original data, please contact the authors of
the above study or refer to their data availability statement.
