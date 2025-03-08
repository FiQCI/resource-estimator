# FiQCI Resource estimator

Simple web app to estimate how many QPU seconds a job takes based on a model. 

## Web Application

The web application uses React, TailwindCSS and Vite. You can build the static files with 

```bash
npm run build
```

For local development you can use

```bash
npm run dev
```

## Deployment

The app is deployed automatically with GitHub pages. The Github workflow [`deploy.yml`](./.github/workflows/deploy.yml) builds the static files and uploads the `dist` folder to the git branch `gh-pages` . Github pages is configured to deploy from `gh-pages` branch. 

## Data availability

Data used in the resource estimation can be found [here](./data_analysis/data/). It contains comma separated value data of varying qubits, circuit depth, number of circuits in a batch and shots with the real QPU seconds runtime calculated from [IQM Timestamps](https://docs.meetiqm.com/iqm-client/integration_guide.html#job-phases-and-related-timestamps) as `execution_end`-`execution_start`. 

## Documentation

The documentation on the rendered webpage can be changed by editing the file [public/documentation.md](./public/documentation.md).