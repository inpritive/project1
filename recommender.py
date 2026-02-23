"""
Rule-based style recommendation system.
Uses skin tone, undertone, and face shape to suggest clothing colors,
hairstyles, and sneakers. Designed for easy ML model integration later.
"""

from typing import Any

# ---------------------------------------------------------------------------
# Rule tables (extensible: add rules here or replace with ML model calls)
# ---------------------------------------------------------------------------

# Undertone → clothing colors that complement each undertone
UNDERTONE_COLORS = {
    "warm": [
        "Terracotta", "Olive green", "Mustard yellow", "Rust", "Camel",
        "Warm brown", "Coral", "Gold", "Peach", "Burnt orange",
    ],
    "cool": [
        "Sapphire blue", "Emerald green", "Amethyst purple", "Navy",
        "Cool gray", "Silver", "Mint", "Burgundy", "Teal", "Slate blue",
    ],
    "neutral": [
        "Black", "White", "Charcoal gray", "Soft pastels", "Blush pink",
        "Lavender", "Dusty blue", "Taupe", "Cream", "Rose",
    ],
}

# Face shape → hairstyles that flatter each shape (balance/contrast theory)
FACE_SHAPE_HAIRSTYLES = {
    "round": [
        "Layered cuts with volume on top", "Side-swept bangs",
        "Long layers to add length", "Textured pixie with height",
        "Asymmetric styles", "Voluminous curls",
    ],
    "square": [
        "Textured or side-swept styles", "Soft layers around the jawline",
        "Wispy bangs to soften angles", "Curled ends",
        "Long waves", "Side-parted styles",
    ],
    "oval": [
        "Most styles work—versatile face shape", "Blunt cuts",
        "Center part", "Pixie cuts", "Long and sleek", "Bob",
    ],
    "heart": [
        "Chin-length or longer to balance forehead", "Side-swept bangs",
        "Volume at chin level", "Soft layers", "Wavy bob",
        "Layered lob (long bob)",
    ],
    "long": [
        "Styles that add width (bangs, volume at sides)", "Side-swept bangs",
        "Layered cuts with body", "Curled or wavy styles",
        "Chin-length bob", "Texture and movement",
    ],
}

# Face shape → sneaker types (balance: angular faces ↔ softer sneakers, etc.)
FACE_SHAPE_SNEAKERS = {
    "round": [
        "Chunky or platform sneakers", "High-tops for structure",
        "Angular or geometric designs", "Retro runner styles",
    ],
    "square": [
        "Minimalist sneakers", "Sleek low-profile styles",
        "Slip-on or streamlined silhouettes", "Clean white sneakers",
    ],
    "oval": [
        "Versatile—any sneaker type works", "Classic white sneakers",
        "Low-tops", "Fashion-forward designs",
    ],
    "heart": [
        "Balanced, medium-structure sneakers", "Classic tennis shoes",
        "Dad sneakers", "Retro-inspired designs",
    ],
    "long": [
        "Chunky or substantial sneakers", "Platform styles",
        "Bold designs", "High-tops for visual balance",
    ],
}

# Fallback when face shape is unknown (e.g., no face detected)
DEFAULT_HAIRSTYLES = [
    "Layered cuts", "Side-swept styles", "Classic bob", "Long layers",
    "Textured pixie", "Wavy lob",
]
DEFAULT_SNEAKERS = [
    "Classic white sneakers", "Chunky sneakers", "Minimalist low-tops",
    "Retro runners", "Slip-on styles",
]


def recommend_styles(
    features: dict[str, Any],
    top_k: int = 5,
) -> dict[str, list[str]]:
    """
    Generate style recommendations from extracted vision features.

    Uses skin tone undertone for colors and face shape for hairstyles
    and sneakers. Returns a structured dict for easy display and
    future ML integration.

    Args:
        features: Output from vision_utils.extract_features(). Expected keys:
                  'skin_tone' (dict with 'undertone'), 'face_shape' (str).
        top_k: Max number of items per category to return.

    Returns:
        {
            "clothing_colors": [...],
            "hairstyles": [...],
            "sneakers": [...]
        }
    """
    # Extract inputs (handle missing gracefully for extensibility)
    skin_tone = features.get("skin_tone") or {}
    undertone = (skin_tone.get("undertone") or "neutral").lower()
    face_shape = (features.get("face_shape") or "oval").lower()

    # Normalize to known keys
    if undertone not in UNDERTONE_COLORS:
        undertone = "neutral"
    if face_shape not in FACE_SHAPE_HAIRSTYLES:
        face_shape = "oval"

    # Rule-based selection (replace this block with ML model call when ready)
    clothing_colors = UNDERTONE_COLORS[undertone][:top_k]
    hairstyles = FACE_SHAPE_HAIRSTYLES.get(
        face_shape, DEFAULT_HAIRSTYLES
    )[:top_k]
    sneakers = FACE_SHAPE_SNEAKERS.get(
        face_shape, DEFAULT_SNEAKERS
    )[:top_k]

    return {
        "clothing_colors": clothing_colors,
        "hairstyles": hairstyles,
        "sneakers": sneakers,
    }


def get_outfit_suggestions(
    features: dict[str, Any],
    occasion: str | None = None,
) -> list[str]:
    """
    Suggest outfit types based on features and optional occasion.

    Args:
        features: Extracted vision features.
        occasion: Optional context (e.g., 'casual', 'formal').

    Returns:
        List of outfit suggestions.
    """
    rec = recommend_styles(features, top_k=3)
    suggestions = []

    # Base suggestions from colors
    colors = ", ".join(rec["clothing_colors"][:3])
    suggestions.append(f"Outfit in {colors}")

    if occasion == "casual":
        suggestions.extend(["Relaxed fit with recommended sneakers", "Layered casual look"])
    elif occasion == "formal":
        suggestions.extend(["Tailored pieces in recommended colors", "Structured silhouette"])

    return suggestions[:5]
