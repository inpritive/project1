"""
AI Personal Stylist - Streamlit UI.
Main entry point for the application.
"""

import streamlit as st
from pathlib import Path

import config
from recommender import recommend_styles
from vision_utils import extract_features
from generator import generate_styled_image


def _init_session_state() -> None:
    """Initialize session state keys for analysis and recommendations."""
    if "features" not in st.session_state:
        st.session_state.features = None
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = None
    if "file_path" not in st.session_state:
        st.session_state.file_path = None
    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []


def main() -> None:
    st.set_page_config(
        page_title="AI Personal Stylist",
        page_icon="👗",
        layout="wide",
    )
    _init_session_state()

    # Header
    st.title("👗 AI Personal Stylist")
    st.markdown("*Get personalized style recommendations and styled looks powered by AI*")
    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        top_k = st.slider(
            "Top recommendations per category",
            1,
            10,
            config.DEFAULT_TOP_K_RECOMMENDATIONS,
        )
        st.caption("Number of items to show for colors, hairstyles, and sneakers")

    # --- Upload section ---
    st.subheader("📤 1. Upload your photo")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[ext.lstrip(".") for ext in config.SUPPORTED_EXTENSIONS],
        help="Upload a face or full-body photo for analysis",
    )

    if uploaded_file is None:
        st.info("👆 Upload a photo to get started")
        return

    # Save uploaded file
    upload_path = Path(config.UPLOAD_DIR)
    upload_path.mkdir(parents=True, exist_ok=True)
    file_path = upload_path / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Reset cached results when a new file is uploaded
    if st.session_state.file_path != file_path:
        st.session_state.file_path = file_path
        st.session_state.features = None
        st.session_state.recommendations = None
        st.session_state.generated_images = []

    # --- Original image ---
    st.subheader("🖼️ 2. Your photo")
    col_img, _ = st.columns([1, 2])
    with col_img:
        st.image(uploaded_file, caption="Original image", use_container_width=True)

    # --- Analyze button ---
    st.subheader("🔍 3. Analyze & recommend")
    if st.button("Run feature extraction", type="primary"):
        with st.spinner("Analyzing your photo (face detection, skin tone, face shape)..."):
            try:
                features = extract_features(file_path)
                recommendations = recommend_styles(features, top_k=top_k)
                st.session_state.features = features
                st.session_state.recommendations = recommendations
                st.success("Analysis complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.exception(e)

    features = st.session_state.features
    recommendations = st.session_state.recommendations

    if features is None or recommendations is None:
        st.info("Click **Run feature extraction** to analyze your photo.")
        return

    # --- Results section ---
    st.divider()
    st.subheader("✨ Analysis results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 🎨 Skin tone")
        skin = features.get("skin_tone") or {}
        undertone = skin.get("undertone", "—")
        dominant = skin.get("dominant_color", (0, 0, 0))
        st.metric("Undertone", undertone.capitalize())
        st.caption(f"Dominant RGB: {dominant}")

    with col2:
        st.markdown("##### 👤 Face shape")
        face_shape = features.get("face_shape") or "—"
        st.metric("Shape", face_shape.capitalize())

    with col3:
        st.markdown("##### 💡 Quick picks")
        colors = recommendations.get("clothing_colors", [])[:2]
        st.write(", ".join(colors) if colors else "—")

    # --- Style recommendations ---
    st.markdown("##### 📋 Style recommendations")
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    with rec_col1:
        st.markdown("**👕 Clothing colors**")
        for c in recommendations.get("clothing_colors", []):
            st.write(f"• {c}")
    with rec_col2:
        st.markdown("**💇 Hairstyles**")
        for h in recommendations.get("hairstyles", []):
            st.write(f"• {h}")
    with rec_col3:
        st.markdown("**👟 Sneakers**")
        for s in recommendations.get("sneakers", []):
            st.write(f"• {s}")

    # --- Generate section ---
    st.divider()
    st.subheader("🎨 4. Generate styled images")
    st.caption("Creates 3 variations using different outfit recommendations. May take a few minutes.")

    if st.button("Generate 3 styled outputs", type="secondary"):
        cols = recommendations.get("clothing_colors", [])[:3]
        hairs = recommendations.get("hairstyles", [])[:3]
        sneakers = recommendations.get("sneakers", [])[:3]

        if not cols or not hairs or not sneakers:
            st.warning("Need at least one recommendation per category.")
        else:
            output_dir = Path(config.OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)

            progress = st.progress(0)
            status = st.empty()
            images = []

            for i in range(3):
                color = cols[i] if i < len(cols) else cols[-1]
                hair = hairs[i] if i < len(hairs) else hairs[-1]
                shoe = sneakers[i] if i < len(sneakers) else sneakers[-1]

                status.info(f"Generating look {i + 1}/3: {color} outfit...")
                try:
                    path = generate_styled_image(
                        file_path,
                        clothing_color=color,
                        hairstyle=hair,
                        sneaker_type=shoe,
                        output_dir=output_dir,
                        output_filename=f"styled_{i + 1}.png",
                    )
                    images.append(path)
                except Exception as e:
                    st.error(f"Generation {i + 1} failed: {e}")
                progress.progress((i + 1) / 3)

            status.empty()
            progress.empty()
            st.session_state.generated_images = images
            st.success("Generation complete!")
            st.rerun()

    # --- Display generated images ---
    if "generated_images" in st.session_state and st.session_state.generated_images:
        st.markdown("##### Generated looks")
        gen_col1, gen_col2, gen_col3 = st.columns(3)
        imgs = st.session_state.generated_images
        for i, col in enumerate([gen_col1, gen_col2, gen_col3]):
            with col:
                if i < len(imgs) and Path(imgs[i]).exists():
                    st.image(
                        str(imgs[i]),
                        caption=f"Look {i + 1}",
                        use_container_width=True,
                    )
                else:
                    st.caption(f"Look {i + 1}: —")


if __name__ == "__main__":
    main()
