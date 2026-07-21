# Deploying StockIntel

A one-time guide to deploying the platform to [Render](https://render.com)
using the committed [`render.yaml`](../render.yaml) blueprint.

The repository is deploy-ready; the steps below are the parts only you can do —
creating the account, entering payment, and pasting in secrets and URLs.

---

## What you're deploying, and what it costs

| Service | Render tier | Why | Approx. cost |
|---|---|---|---|
| Backend (FastAPI, Docker) | **Standard** (2 GB RAM) | PyTorch + the per-request walk-forward training need real memory; 512 MB OOMs | ~$25/mo |
| Frontend (Next.js, Node) | **Starter** | Static-ish; free tier works but spins down when idle | ~$7/mo or free |

**The backend cannot run on a free tier.** FinBERT and the LSTM mode load large
tensors, and each prediction request runs 5-fold walk-forward validation
(~10–30 s of CPU). 512 MB runs out of memory. A FRED-only deployment using only
the default LightGBM mode is lighter and *may* fit Starter, but Standard is the
safe choice.

Cheaper alternative for the frontend: deploy it to **Vercel** for free instead
(see the end of this guide) and use Render only for the backend.

---

## A deliberate choice about API keys

A public deployment exercises whatever keys you configure:

- **FRED** — free and unlimited. Safe to expose. **Deploy with this one.**
- **Marketaux** — free tier is **100 requests/day**. A handful of public
  visitors exhausts it, after which everyone sees "allowance exhausted".
- **Gemini** — anyone loading the page spends your quota.

The application degrades honestly: with Marketaux and Gemini unset, the News and
Sentiment sections show their `NOT CONFIGURED` state, and everything else — the
full ML pipeline, charts, analytics, macro context, predictions — works. The
blueprint leaves those two keys unset by default for exactly this reason.

If you want news and sentiment live publicly, use **separate, rate-limited keys
you are willing to have spent by strangers** — never the ones from your local
`.env`.

---

## Steps

### 1. Prerequisites

- A [Render account](https://dashboard.render.com/register) with this GitHub
  repository connected.
- A payment method on file (required for the Standard backend tier).

### 2. Deploy the blueprint

1. Render dashboard → **New** → **Blueprint**.
2. Select this repository. Render detects `render.yaml` and shows two services:
   `stockintel-backend` and `stockintel-frontend`.
3. Click **Apply**. Both services begin building. The backend image build takes
   a few minutes (it installs PyTorch); the frontend builds Next.js.

The build will succeed, but the two services can't talk to each other yet —
that's the next step.

### 3. Set the FRED key (backend)

1. Open the **stockintel-backend** service → **Environment**.
2. Set `FRED_API_KEY` to your key from
   [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).
3. Save. The backend redeploys.

### 4. Connect the two services

Each service's public URL only exists after its first deploy, so this cross-wiring
is manual and one-time.

1. Copy the **backend** URL (e.g. `https://stockintel-backend.onrender.com`).
2. On **stockintel-frontend** → Environment, set
   `NEXT_PUBLIC_API_URL` to that backend URL. Save — the frontend **rebuilds**
   (this value is baked in at build time, so a redeploy alone is not enough).
3. Copy the **frontend** URL (e.g. `https://stockintel-frontend.onrender.com`).
4. On **stockintel-backend** → Environment, set `FRONTEND_ORIGIN` to that
   frontend URL. Save — the backend redeploys. This is what the browser's CORS
   check requires.

### 5. Verify

- Backend health: visit `https://<backend>.onrender.com/health` → `{"status":"ok"}`.
- Backend docs: `https://<backend>.onrender.com/docs`.
- Open the frontend URL. The landing page loads immediately; the dashboard's
  first analysis takes ~20–30 s (walk-forward validation on a cold cache).
- Macro context shows live FRED data. News and Sentiment show `NOT CONFIGURED` —
  correct, since those keys are intentionally unset.

### 6. (Optional) Enable news and sentiment

Only if you accept the key-exposure trade-off above. On the backend service, set
`MARKETAUX_API_KEY` and/or `GEMINI_API_KEY` to **dedicated public-facing keys**,
then save. No rebuild needed — the backend reads these at runtime.

---

## Frontend on Vercel instead (free)

If you'd rather not pay for the frontend service:

1. Import the repo at [vercel.com/new](https://vercel.com/new).
2. Set **Root Directory** to `stockintel/frontend`.
3. Add environment variable `NEXT_PUBLIC_API_URL` = your Render backend URL.
4. Deploy. Then set the backend's `FRONTEND_ORIGIN` to the Vercel URL (step 4
   above) so CORS allows it.

Remove the `stockintel-frontend` service from Render if you go this route.

---

## Notes and limitations

- **First request is slow.** Cold cache + walk-forward validation ≈ 20–30 s.
  Subsequent requests for the same stock are cached and fast.
- **Ephemeral filesystem.** Render's disk resets on redeploy; the cache
  rebuilds on demand. This is fine — nothing durable lives on disk.
- **Single worker, serialized requests.** Concurrency is intentionally limited
  to keep memory predictable. Adequate for a demo; not tuned for load.
- **Secrets never enter the repo or the image.** `.env` is gitignored and
  dockerignored; production config comes entirely from Render's environment
  variables, which the app reads via pydantic-settings.
