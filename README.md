# Twitter Sentiment Analysis

**Project Overview**

- **Purpose**: Binary sentiment classification of tweets (positive / negative) using DistilBERT-based models trained with TensorFlow.
- **Components**: a Python `backend` providing a FastAPI inference service and a `frontend` (Next.js) UI. Training artifacts, notebooks, and evaluation figures live under `backend/Notebooks` and `backend/Report`.

**Repository Layout**

- **`backend/`**: FastAPI inference app, model checkpoints, Dockerfile, and training notebooks.
  - `entry_main.py` : FastAPI app that loads the saved DistilBERT checkpoint and exposes a `/tweet` POST endpoint.
  - `requirements.txt` : Python dependency list for backend runtime.
  - `Dockerfile`, `start.sh` : containerization and startup script for the backend service.
  - `Models/` : saved model checkpoints (e.g. `chunk_model1`, `chunk_model2`).
  - `Data/` : raw and processed datasets (chunked CSVs for training).
  - `Notebooks/` : training and preprocessing notebooks (contains `DistillBert_Model.ipynb`).
  - `Report/` : generated training report and evaluation figures.
- **`frontend/`**: Next.js application providing a UI (pages in `app/` and static `public/`).

**Quick Start — Local (backend only)**

- **Clone repository**:

```powershell
git clone <repo-url> twitter-sentiment
cd twitter-sentiment/backend
```

- **Build Docker image (recommended for parity with deployment)**:

```powershell
docker build -t twitter-sentiment-backend:latest .
```

- **Run container (mount local model checkpoint if needed)**:

```powershell
docker run --rm -p 8080:8080 -e PORT=8080 -v "C:\path\to\local\Models\chunk_model2:/app/Models/chunk_model2" twitter-sentiment-backend:latest
```

- **Test the API**:

```powershell
curl -X POST "http://localhost:8080/tweet" -H "Content-Type: application/json" -d '{"text":"I love this product!"}'
```

**Python Virtualenv (optional, without Docker)**

# Twitter Sentiment Analysis — Comprehensive Guide

## Problem Statement

Social media platforms produce large volumes of user-generated text. Understanding overall public sentiment about products, events, or topics is valuable for analytics, monitoring, and decision making. This project performs binary sentiment classification of Twitter posts (tweets) into `positive` or `negative` classes using a DistilBERT-based transformer model fine-tuned with TensorFlow.

Goals:

- Provide a reproducible training pipeline and evaluation artifacts.
- Expose an inference API for integration with a front-end or other services.
- Provide containerized deployment instructions (including Hugging Face Spaces compatibility).

## What this repository contains (high-level)

- `backend/` — FastAPI inference service, training notebooks, preprocessing utilities, model checkpoints, Dockerfile and container start script.
- `frontend/` — Next.js application (UI) that can call the backend inference API.
- `backend/Data/` — raw and processed datasets. Processed data is chunked for memory-efficient training.
- `backend/Models/` and `backend/Trained_weights/` — saved model checkpoints and tf/h5 weights.
- `backend/Notebooks/` — Jupyter notebooks used for preprocessing, training and evaluation.
- `backend/Report/` — generated report and evaluation figures (confusion matrix, PR curve, loss curves).

## Quick start — run inference locally (recommended flow)

1. Clone the repo and open the `backend` folder:

```powershell
git clone <repo-url>
cd Twitter-Sentiment/backend
```

2. (Recommended) Build the Docker image and run the service (ensures environment parity):

```powershell
docker build -t twitter-sentiment-backend:latest .
# If your model checkpoints are stored locally, mount them into the container:
docker run --rm -p 8080:8080 -e PORT=8080 -v "C:\path\to\Models\chunk_model2:/app/Models/chunk_model2" twitter-sentiment-backend:latest
```

3. Test the inference endpoint:

```powershell
curl -X POST "http://localhost:8080/tweet" -H "Content-Type: application/json" -d '{"text":"I love this product!"}'
```

4. If you prefer not to use Docker, use a virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
uvicorn entry_main:app --host 0.0.0.0 --port 8080
```

## Detailed Setup & Development Flow

1. Install dependencies

```powershell
# from backend/
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2. Prepare data

- Raw data: `backend/Data/raw/train.csv`.
- Processed chunks: `backend/Data/processed/Chunks/chunk_*.csv`. Use the notebook `backend/Notebooks/preprocessing.ipynb` to reproduce the same cleaning/tokenization steps used for training.

3. Train a model (notebook)

- Open `backend/Notebooks/DistillBert_Model.ipynb` and follow the training pipeline. The notebook includes dataset loading (via `Notebooks/utils.py`), model instantiation, training loops and evaluation.
- Save checkpoints into `backend/Models/<checkpoint_name>/` using the `transformers` save methods, or export `h5` weights to `backend/Trained_weights/`.

4. Evaluate

- The notebook saves evaluation plots to `backend/Report/Figures/` (e.g., PR curve, F1 curve, confusion matrix). Inspect `backend/Report/Model_Training_Report.md` for interpretation and guidance.

5. Run inference

- The inference FastAPI app is `backend/entry_main.py`. It expects to find model files under `backend/Models/chunk_model2` when starting. The app exposes `POST /tweet` which accepts JSON with a `text` field and responds with `sentiment` and `confidence`.

## Architecture & Implementation Notes

- Model: `TFDistilBertForSequenceClassification` fine-tuned for binary classification.
- Tokenizer: `transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')` (loaded via `Notebooks/utils.py`).
- Input preprocessing: mentions replaced with `@user`, URLs removed, hashtags simplified, and whitespace normalized (see `entry_main.clean_text`).
- Inference: the model returns logits; code applies `tf.nn.sigmoid` on logits and thresholds at `0.5` to select `positive` or `negative`.

## Deployment: Hugging Face Spaces (Docker) — recommended approach

Two common ways to deploy to Spaces:

- 1. Include model files in the Space repo (only for small models): commit `backend/Models/chunk_model2` into the Space repo and include the `backend/Dockerfile`. Spaces will build the Docker image and run `start.sh`.

- 2. Host model checkpoints on the Hugging Face Hub and load them at container start (recommended for larger models): push your model and tokenizer to the Hub using `model.push_to_hub(...)` and `tokenizer.push_to_hub(...)` from a notebook, then load from the hub in `entry_main.py` or in a short startup script.

Example to push and load from the Hub:

```python
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('your-username/twitter-sentiment-distilbert')
model = TFDistilBertForSequenceClassification.from_pretrained('your-username/twitter-sentiment-distilbert')
```

When targeting GPU Spaces, use a CUDA-enabled TensorFlow wheel and appropriate base image; modify the `Dockerfile` to install GPU drivers and the correct `tensorflow` package.

## How to Train from Scratch (concise)

1. Prepare and clean your dataset using `backend/Notebooks/preprocessing.ipynb`.
2. Open `backend/Notebooks/DistillBert_Model.ipynb` and set hyperparameters (batch size, epochs, learning rate).
3. Train and monitor the training/validation loss and metrics.
4. Save the model checkpoint and tokenizer with `model.save_pretrained(...)` and `tokenizer.save_pretrained(...)` into `backend/Models/<name>/`.
5. Optionally export `.h5` weights to `backend/Trained_weights/` for storage.

## Usage Examples

- Example curl request:

```powershell
curl -X POST "http://localhost:8080/tweet" -H "Content-Type: application/json" -d '{"text":"This is an awesome feature"}'

# Example response
#{"sentiment": "positive", "confidence": 0.936}
```

## Frontend Integration

- The `frontend/` Next.js app can call the backend `/tweet` endpoint. Ensure CORS is allowed (backend already includes permissive CORS middleware in `entry_main.py`). If you host frontend and backend separately, set `allow_origins` to your frontend host for production.

## Reproducibility & Best Practices

- Save a `training-config.json` with hyperparameters, random seeds and dataset versions alongside each checkpoint.
- Use `transformers` `push_to_hub` for model artifacts to avoid large repository sizes.
- Add a small smoke-test dataset and a `train_smoke.sh` to validate environment correctness in CI.

## Troubleshooting

- Model load errors: verify `Models/chunk_model2` exists or adjust `entry_main.py` to download from the Hub at startup.
- Slow startup / memory pressure: use smaller models, reduce `max_length` or batch sizes during training, or use GPU-accelerated instances.
- Windows Docker volume mounts: on Windows use absolute Windows paths and ensure Docker has access to the drive.

## Contributing

- Fork the repo and create feature branches. Open PRs against `main`.
- Add tests for preprocessing and a small unit test for the API endpoint behavior.

## License

- (Add your license or company policy here.)

## Where to look next in this repo

- `backend/entry_main.py` — inference API.
- `backend/Notebooks/DistillBert_Model.ipynb` — training notebook.
- `backend/Report/Model_Training_Report.md` — analysis and figures.

---

If you'd like, I can now:

- run a syntax check across `backend/*.py`,
- build and run the Docker image locally and test the endpoint, or
- extract training hyperparameters from `backend/Notebooks/DistillBert_Model.ipynb` and create `backend/Models/training-config.json` for reproducibility.

## Full detailed reference (file-by-file, function-by-function)

This section documents implementation-level details for the repository. If you plan to modify, reproduce, or debug any part of the project, read this section carefully.

Top-level layout (exact paths referenced below):

- `backend/` — main Python service, training notebooks, data and model artifacts.
- `frontend/` — Next.js UI that interacts with the backend inference API.

### `backend/entry_main.py` — FastAPI inference service (precise behavior)

- Purpose: provide a lightweight HTTP API to classify a single tweet text.
- Key imports and why they matter:

  - `FastAPI` and `CORSMiddleware`: create the web app and configure CORS for browser-based frontends.
  - `pydantic`: enforces request payload structure; prevents malformed requests from reaching inference code.
  - `Notebooks.utils`: central utility module (tokenizer loading, plotting helpers, model loaders). Importing it allows code re-use between training notebooks and the API.
  - `tensorflow as tf`: used for numerical ops (sigmoid) and for loading TF-based checkpoints.
  - `transformers`: `TFDistilBertForSequenceClassification` is the model class used for inference; `AutoTokenizer` is used by the utils helper.

- App initialization:

  - `app = FastAPI()` creates the ASGI app.
  - `app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])` sets permissive CORS for dev; change `allow_origins` to a whitelist for prod.

- Model & tokenizer loading (at import time):

  - `model = TFDistilBertForSequenceClassification.from_pretrained('./Models/chunk_model2')`
    - This call expects a HF-style directory containing either a TF checkpoint or the serialized weights/config. If files are missing this raises an error and the process will not start.
    - If your checkpoint was saved as an HDF5 (`.h5`) file only, adapt this section to call `model.load_weights` and provide the path.
  - `tokenizer = utils.load_tokenizer('distilbert-base-uncased')`
    - This uses `Notebooks/utils.load_tokenizer` which delegates to `AutoTokenizer.from_pretrained`.
    - Replace the argument with your HF hub repo id if you pushed tokenizer to the hub.

- Input validation model:

  - `class TwitterTweet(pydantic.BaseModel): text: str` — returns 422 errors automatically if `text` is missing or not a string.

- `clean_text(text)`:

  - Steps:
    1. Replace mentions `@username` with `@user` via `re.sub(r'@\S+', '@user', text)` — reduces token sparsity and anonymizes.
    2. Remove URLs via `re.sub(r'http\S+|www\S+', '', text)`.
    3. Remove `#` from hashtags, preserving the word via `re.sub(r'#(\w+)', r'\1', text)`.
    4. Normalize whitespace with `re.sub(r'\s+', ' ', text).strip()`.
  - Rationale: these steps reduce noisy tokens, reduce vocabulary, and align training and inference text cleaning.

- `POST /tweet` (detailed request/response flow):
  1. Request JSON -> `TwitterTweet` model validated.
  2. Text cleaned with `clean_text`.
  3. Tokenization: `tokenizer(cleaned_text, return_tensors='tf', padding='max_length', truncation=True, max_length=128)`
     - `max_length=128` chosen for balance between context and memory; adjust if needed.
  4. Inference: `outputs = model.predict(inputs)`; `logits = outputs.logits`.
     - Note: `model.predict` runs eager inference; for higher throughput consider `model.call` under `tf.function` or use batching.
  5. Probability: `prob = tf.nn.sigmoid(logits)[0][0].numpy()` — interprets logits as single-logit binary output. If your model returns two logits (class scores), use `tf.nn.softmax` and argmax instead.
  6. Decision: `sentiment = 'positive' if prob >= 0.5 else 'negative'` and respond `{"sentiment": sentiment, "confidence": float(prob)}`.

### `backend/Notebooks/utils.py` — utility functions (detailed)

- Plotting helpers (for notebooks):

  - `plot_accuracy(train_acc, val_acc)`, `plot_loss(train_loss, val_loss)`, `plot_confusion_matrix(y_true, y_pred, classes=[0,1])`, `plot_precision_recall(y_true, y_scores)`, `plot_f1_score(y_true, y_scores)`, `plot_dataset_distribution(texts, targets)` — these wrap matplotlib/seaborn for quick visualization.

- Model loading helpers:

  - `load_bert_model(num_labels)` and `load_distilbert_model(num_labels)` load pre-trained models from the HF model hub and set `num_labels` (useful for training from scratch with custom label counts).
  - `load_bert_checkpoint(checkpoint_path, num_labels)` and `load_distilbert_checkpoint(checkpoint_path, num_labels)` are helpers to load from local checkpoint directories. Pay attention to `from_pt=True` flags; if you trained in PyTorch and converted to TF, `from_pt=True` helps.

- Tokenizer loader:

  - `load_tokenizer(model_name)` returns `AutoTokenizer.from_pretrained(model_name)`; the `model_name` may be a hub id or a local folder path.

- `load_dataframe(file_index)`:
  - Reads a chunked CSV from a hard-coded Colab path: `/content/drive/MyDrive/Colab Notebooks/backend/Data/processed/Chunks/chunk_{file_index}.csv` and maps labels `4 -> 1`.
  - Important: that path is Colab-specific. For local runs update the path to `backend/Data/processed/Chunks/chunk_{file_index}.csv` or use a config variable.

### `backend/requirements.txt` — installed packages and purpose

- `fastapi` — HTTP API framework.
- `uvicorn[standard]` — ASGI server for running FastAPI.
- `pydantic` — data validation for request bodies.
- `tensorflow==2.13.4` — model runtime; pinned to this version in the repo to ensure compatibility with saved checkpoints. If you need GPU support, install a CUDA-enabled TF wheel matching your GPU driver.
- `transformers` — model and tokenizer utilities from HF.
- `scikit-learn` — metrics and helpers used in notebooks (confusion matrix, PR curve).
- `pandas` — CSV reading and dataset handling.
- `matplotlib`, `seaborn` — plotting utilities used by the notebooks.

When editing `requirements.txt`:

- Pin versions for reproducibility (especially `tensorflow` and `transformers`).
- Use `pip freeze > requirements.txt` from a known-good environment after you verify training/inference works.

### `backend/Dockerfile` — line-by-line explanation

- `FROM python:3.10-slim` — minimal Python base image.
- `ENV PYTHONDONTWRITEBYTECODE=1` and `ENV PYTHONUNBUFFERED=1` — standard env flags to avoid .pyc files and force stdout/stderr flushing.
- `ENV PORT=8080` — default served port.
- `RUN apt-get update && apt-get install -y --no-install-recommends build-essential git wget curl ca-certificates libglib2.0-0 libgomp1` — system deps required by some Python packages; `libgomp1` often required for optimized TF builds.
- `WORKDIR /app` — set working dir.
- `COPY backend/requirements.txt ./requirements.txt` and `RUN pip install --upgrade pip setuptools wheel && pip install --no-cache-dir -r requirements.txt` — installs Python packages.
- `COPY backend/ ./` — copy all backend files into image.
- `RUN chmod +x ./start.sh || true` — ensures `start.sh` is executable.
- `EXPOSE 8080` — document port.
- `HEALTHCHECK ... CMD curl -f http://localhost:8080/ || exit 1` — basic health check (requires `curl` be installed; if not, adjust or remove).
- `CMD ["bash", "start.sh"]` — default container command runs `start.sh`.

Important notes about the Docker image:

- If you need GPU support, use a CUDA-enabled base image and a GPU TensorFlow wheel; the slim image is CPU-only.
- Model artifacts are copied into the image by `COPY backend/ ./`. Large model files may inflate image size — best practice is to store model on HF Hub or mount at runtime.

### `backend/start.sh` — what it does and why

- The script sets `PORT=${PORT:-8080}`, warns if `./Models/chunk_model2` is missing (common runtime error), then executes `uvicorn entry_main:app --host 0.0.0.0 --port ${PORT} --workers 1`.
- For production scale, consider `--workers` > 1 (but be careful with TF models and multi-process memory usage). Alternatives: use an ASGI process manager or TensorFlow Serving for scaled inference.

### `backend/Models/` and `backend/Trained_weights/` — checkpoint formats

- HF-style checkpoints: a directory containing `config.json` and TF weights (e.g., `tf_model.h5`) can be loaded with `TFDistilBertForSequenceClassification.from_pretrained(path)`.
- `Trained_weights/*.h5` are Keras HDF5 weight files — to load, instantiate model architecture and call `model.load_weights(path)`.

### `backend/Data/` — data layout and processing

- `Data/raw/train.csv`: original dataset (likely labeled with Twitter sentiment classes 0 and 4). Inspect header and encoding when loading.
- `Data/processed/Chunks/chunk_*.csv`: chunked, preprocessed CSVs created for memory-efficient training. Use the preprocessing notebook to reproduce chunking.
- Label mapping: `utils.load_dataframe` maps `4 -> 1` so final labels are `{0,1}` binary.

### `backend/Notebooks/DistillBert_Model.ipynb` — training notebook (how-to)

- Typical steps in the notebook:
  1. Load chunked dataset with `utils.load_dataframe` or a custom data loader.
  2. Create tokenizer & encode datasets via `tokenizer(..., truncation=True, padding='max_length', max_length=128)`.
  3. Instantiate model via `TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1, from_pt=True)` (or `num_labels=2`) depending on approach.
  4. Compile model with an optimizer (e.g., AdamW via `transformers` or `tf.keras.optimizers.Adam`) and appropriate loss (`binary_crossentropy` for single-logit or `sparse_categorical_crossentropy` for 2-logit outputs).
  5. Train with `model.fit(...)` and save best checkpoints using callbacks.
  6. Evaluate and create plots saved to `backend/Report/Figures/`.

Important: the notebook contains the canonical hyperparameters — extract them and record in a JSON file for reproducibility.

### `frontend/` overview (what matters for integration)

- The Next.js app living under `frontend/` calls `POST /tweet` on the backend to get sentiment predictions. It uses `fetch` or similar methods to call the endpoint and display `sentiment` and `confidence`.
- CORS: the backend currently allows all origins; tighten this in production.

### `backend/Report/` — what is included and how to interpret

- `Model_Training_Report.md` contains training summary, data notes, checkpoint locations and evaluation figure embeds (PR curve, F1 curve, confusion matrix, loss curves).
- Figures: `Report/Figures/` contains `.jpeg`/`.png` visuals. Use them to select thresholds or to inspect class imbalance and error modes.

## Deployment & reproducible production flow (detailed)

1. Decide where to store model weights:

   - Option A: Commit small checkpoints to the repo and build Docker images that include weights. Pros: simple; Cons: large repo and large images.
   - Option B (recommended): Push model & tokenizer to the Hugging Face Hub and download at container startup, or load directly from the hub in `entry_main.py` with `from_pretrained('username/model')`.

2. Build Docker image (local test):

```powershell
cd backend
docker build -t twitter-sentiment-backend:latest .
```

3. Run container and mount models at runtime (if not baked into image):

```powershell
docker run --rm -p 8080:8080 -e PORT=8080 -v "C:\path\to\Models\chunk_model2:/app/Models/chunk_model2" twitter-sentiment-backend:latest
```

4. Validate endpoint with a curl command (example):

```powershell
curl -X POST "http://localhost:8080/tweet" -H "Content-Type: application/json" -d '{"text":"I love this product!"}'
```

5. Hugging Face Spaces: create a Space with Docker support, push repo (or a dedicated Space repo) and configure to build. If checkpoint size prevents direct push, host model on HF Hub and load on startup.

## Testing, monitoring and scaling (best practices)

- Unit tests: add tests for `clean_text`, tokenization behavior, and a small integration test for `POST /tweet` using `starlette.testclient.TestClient`.
- Load testing: run a small load test with `hey` or `wrk` to understand latency under concurrent requests.
- Observability: log predictions (anonymized), confidences, and model latency. Save to a central logging system.
- Scaling: for large traffic, consider model batching or a dedicated inference server (TF Serving or Triton). Alternatively increase `uvicorn` workers but watch memory usage.

## Reproducibility checklist (what to save alongside checkpoints)

- `training-config.json` containing: dataset version, training/validation split, seed, optimizer & learning rate schedule, batch size, epochs, max_length, and tokenizer name.
- A small subset of processed data (`smoke_dataset.csv`) for CI tests.
- A `requirements.txt` snapshot and optionally a `pip-tools`/`poetry` spec for deterministic installs.

## Common problems & fixes

- Model load errors: ensure the checkpoint folder is present and contains expected files (`config.json`, `tf_model.h5`, `pytorch_model.bin` depending on format). If you have an HDF5 `h5` file, call `model.load_weights` after model instantiation instead of `from_pretrained`.
- OOM at inference: lower `max_length`, reduce concurrency, or use a larger machine with GPU.
- Tokenizer mismatch: ensure the tokenizer used at inference matches the tokenizer used during training (vocabulary & pre-tokenization must match).

## Security & privacy

- Anonymize user identifiers (the code replaces `@username` with `@user`). If you log inputs, strip or hash personal data and follow data retention rules.

## Next steps I can take for you (pick any):

- Run a syntax check across all `backend/*.py` files and report errors.
- Build and run the Docker image locally and run a smoke test against `/tweet`.
- Extract hyperparameters from the training notebook and generate `backend/Models/training-config.json`.
- Add unit tests and a GitHub Actions CI workflow to run them on pull requests.

---

End of detailed README.
