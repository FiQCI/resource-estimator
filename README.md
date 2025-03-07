# FiQCI Resource estimator

Simple web app to estimate how many QPU seconds a job takes based on a model. 

## Web Application

The web application uses React and TailwindCSS. You can build the static files with 

```bash
npm run build
```

For local development you can use

```bash
npm run dev
```

## Deployment

The app is deployed automatically with GitHub pages. The Github workflow [`deploy.yml`](./.github/workflows/deploy.yml) builds the static files and uploads the `dist` folder to the git branch `gh-pages` . Github pages is configured to deploy from `gh-pages` branch. 