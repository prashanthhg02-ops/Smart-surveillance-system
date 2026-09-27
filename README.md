# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend enabling type-aware lint rules by installing `oxlint-tsgolint` and editing `.oxlintrc.json`:

```json
{
  "$schema": "./node_modules/oxlint/configuration_schema.json",
  "plugins": ["react", "typescript", "oxc"],
  "options": {
    "typeAware": true
  },
  # Sentinel AI

  A browser-based surveillance dashboard with local object detection. Connect a camera to detect people, vehicles, and common objects with TensorFlow.js COCO-SSD.

  ## Run locally

  ```sh
  npm install
  npm run dev
  ```

  Open the local URL printed by Vite. Camera access requires `localhost` or HTTPS. Select **Connect camera** and approve the browser permission prompt. The object-detection model is loaded on first connection; its model weights require an internet connection.

  ## Privacy and limits

  Video frames are processed in the browser and are not recorded or uploaded by this app. The model runtime is downloaded on first use. Detection is assistive and can miss or misclassify objects; verify important alerts independently. No identity recognition, persistent recording, or backend service is included.

  ## Checks

  ```sh
  npm run build
  npm run lint
  ```
