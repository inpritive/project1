# AI Personal Stylist

> An end-to-end AI application that analyzes your photo and delivers personalized style recommendations—from clothing colors to hairstyles and sneakers—then generates realistic styled look visualizations using diffusion models.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Problem Statement

Choosing the right outfit, hairstyle, and accessories is often a guessing game. What colors suit your skin tone? Which cuts flatter your face shape? Most people rely on trial and error or generic advice that ignores their unique features.

**AI Personal Stylist** solves this by:

- **Analyzing** your photo with computer vision to extract skin undertone and face shape
- **Recommending** colors, hairstyles, and sneakers based on rule-based logic grounded in color theory and facial geometry
- **Generating** styled portrait visualizations so you can preview different looks before committing

---

## Features

| Feature | Description |
|--------|-------------|
| **Face & Skin Analysis** | MediaPipe + OpenCV for face detection, 468-point landmarks, skin tone extraction from cheek regions, undertone classification (warm/cool/neutral) |
| **Face Shape Estimation** | Landmark-based ratios to classify face shape (Oval, Round, Square, Heart, Long) |
| **Personalized Recommendations** | Rule-based system suggesting clothing colors, hairstyles, and sneaker types tailored to your profile |
| **AI Image Generation** | HuggingFace Diffusers + Stable Diffusion img2img to generate 3 styled variations while preserving facial identity |
| **Clean Web UI** | Modern Streamlit interface with upload, analysis results, and side-by-side styled outputs |

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | Streamlit |
| **Vision** | OpenCV, MediaPipe (Face Mesh, Face Detection) |
| **Recommendation** | Rule-based engine (extensible for ML) |
| **Generation** | HuggingFace Diffusers, Stable Diffusion v1.5 (img2img) |
| **Core** | Python 3.10+, PyTorch, NumPy, Pillow |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AI Personal Stylist                              │
└─────────────────────────────────────────────────────────────────────────────┘

                                    ┌──────────────┐
                                    │   User       │
                                    │   (Photo)    │
                                    └──────┬───────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          STREAMLIT UI (app.py)                                │
│  Upload │ Display │ Analyze Button │ Results │ Generate Button │ Gallery     │
└──────────────────────────────────────────────────────────────────────────────┘
    │                    │                      │                      │
    │                    │                      │                      │
    ▼                    ▼                      ▼                      ▼
┌─────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│  config.py  │  │  vision_utils   │  │   recommender   │  │      generator      │
│             │  │                 │  │                 │  │                     │
│ Paths       │  │ load_image()    │  │ recommend_      │  │ generate_styled_    │
│ Limits      │  │ detect_face()   │  │   styles()      │  │   image()           │
│ Model IDs   │  │ face_landmarks  │  │                 │  │                     │
└─────────────┘  │ extract_skin_   │  │ • clothing_     │  │ • Stable Diffusion  │
                 │   tone()        │  │   colors        │  │ • img2img pipeline  │
                 │ estimate_face_  │  │ • hairstyles    │  │ • Face preservation │
                 │   shape()       │  │ • sneakers      │  │   (low strength)    │
                 └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘
                          │                    │                      │
                          │  OpenCV            │  Rule tables         │  Diffusers
                          │  MediaPipe         │  (undertone,         │  PyTorch
                          ▼                    │   face_shape)        ▼
                 ┌───────────────────────────────────────────────────────────────┐
                 │                    Data Flow                                   │
                 │  Image → Features (skin_tone, face_shape) → Recommendations    │
                 │       → Prompt engineering → 3 styled outputs                  │
                 └───────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- (Optional) CUDA-capable GPU for faster image generation

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-personal-stylist.git
   cd ai-personal-stylist
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   # source venv/bin/activate     # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Stable Diffusion model** (automatic on first run)
   - The app fetches `runwayml/stable-diffusion-v1-5` from HuggingFace on first generation.
   - Ensure you have ~4GB free disk space and internet access.

---

## How to Run

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open the local URL (typically `http://localhost:8501`).

3. **Workflow**
   - Upload a clear face or full-body photo.
   - Click **Run feature extraction** to analyze skin tone and face shape.
   - Review recommendations (clothing colors, hairstyles, sneakers).
   - Click **Generate 3 styled outputs** to create AI-styled variations (allow 2–5 minutes on CPU, ~30s per image on GPU).

---

## Project Structure

```
ai_personal_stylist/
├── app.py              # Streamlit UI — upload, analyze, recommend, generate
├── vision_utils.py     # Face detection, landmarks, skin tone, face shape
├── recommender.py      # Rule-based style recommendations
├── generator.py        # Stable Diffusion img2img generation
├── config.py           # Paths, model IDs, inference settings
├── requirements.txt
└── README.md
```

---

## Future Improvements

| Area | Idea |
|------|------|
| **Vision** | Add body/pose detection (MediaPipe Pose) for full-body styling context |
| **Recommender** | Replace rules with a fine-tuned ML model (embeddings, collaborative filtering) |
| **Generation** | Integrate identity-preserving models (IP-Adapter, InstantID) for stronger face consistency |
| **UI/UX** | Allow user to pick which recommendations to visualize; save/export results |
| **Performance** | Quantization, ONNX export, or smaller models for CPU-only setups |
| **Data** | Build a labeled dataset to train a custom recommender or virtual try-on model |

---

## License

MIT License — feel free to use and extend for learning or projects.
