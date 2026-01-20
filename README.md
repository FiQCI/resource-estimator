# FiQCI Resource Estimator

Simple web app to estimate how many QPU seconds a job takes based on a model.

## Quick Start

### Web Application

The web application uses React, TailwindCSS and Vite. You can build the static files with:

```bash
npm run build
```

For local development:

```bash
npm run dev
```

### Data collection and model generation

To collect data and generate models, a python environment is required and the python code in this project to be installed.

```bash
# 1. Create and activate virtual environment (using uv)
uv venv --python 3.11
source .venv/bin/activate

# 2. Install package in editable mode with all dependencies
uv pip install -e .

# 3. Generate data from quantum hardware
resource-estimator-generate --server-url <IQM_URL> --output data_analysis/data/device.csv

# 4. Build the model
resource-estimator-build --data data_analysis/data/device.csv --device helmi

# 5. Validate and visualize
resource-estimator-validate --data data_analysis/data/device.csv --device helmi --output plots/

# 6. Update frontend with new model parameters (copy JS output from step 4)
# 7. Rebuild and redeploy
npm run build
```

### Development

```bash
# Install package with all dependencies (including dev)
uv sync --group dev

# Run tests with coverage
uv run pytest tests/ -v --cov=src/resource_estimator --cov-report=term-missing

# Install pre-commit hooks for automatic checks
uv run pre-commit install
```

## Data Availability

Data used in the resource estimation can be found in [`data_analysis/data/`](./data_analysis/data/). It contains comma-separated value data of varying qubits, circuit depth, number of circuits in a batch, and shots with the real QPU seconds runtime calculated from [IQM Timestamps](https://docs.meetiqm.com/iqm-client/integration_guide.html#job-phases-and-related-timestamps) as `execution_end`-`execution_start`.

## Model Details

The resource estimator uses polynomial ridge regression models trained on real quantum hardware execution data. Each device has its own model parameters stored in [`src/utils/ResourceEstimatorModel.js`](./src/utils/ResourceEstimatorModel.js).

**Model features:**
- Number of qubits
- Circuit depth
- Number of circuits (batches)
- Number of shots (k_shots = shots/1000)

**Model includes:**
- Linear terms
- Quadratic terms
- Interaction terms between features

## Updating the Model

When you need to update the model (e.g., after software or hardware changes):

1. **Generate fresh data** using `resource-estimator-generate`
2. **Build new model** using `resource-estimator-build`
3. **Validate model** using `resource-estimator-validate`
4. **Update frontend** by copying the JavaScript output into `src/utils/ResourceEstimatorModel.js`
5. **Rebuild** the frontend with `npm run build`
6. **Redeploy** (automatic via GitHub Actions)


## Documentation

The user-facing documentation on the webpage can be changed by editing [`public/documentation.md`](./public/documentation.md).
