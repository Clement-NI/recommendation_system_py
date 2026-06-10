"""Generate presentation slides for the NCF Recommendation System project."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Colors
BG_DARK = RGBColor(0x1A, 0x1A, 0x2E)
BG_CARD = RGBColor(0x16, 0x21, 0x3E)
ACCENT = RGBColor(0x00, 0xD2, 0xFF)
ACCENT2 = RGBColor(0x7C, 0x3A, 0xED)
ACCENT3 = RGBColor(0x10, 0xB9, 0x81)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0xA0, 0xA0, 0xB0)
ORANGE = RGBColor(0xFB, 0x92, 0x3C)
RED = RGBColor(0xEF, 0x44, 0x44)
YELLOW = RGBColor(0xFB, 0xBF, 0x24)


def set_slide_bg(slide, color=BG_DARK):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, left, top, width, height, text, font_size=18,
                 color=WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_card(slide, left, top, width, height, color=BG_CARD):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def add_multiline_box(slide, left, top, width, height, lines, font_size=16,
                      color=WHITE, line_spacing=1.3, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, (text, text_color, is_bold, size_override) in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = text
        p.font.size = Pt(size_override if size_override else font_size)
        p.font.color.rgb = text_color if text_color else color
        p.font.bold = is_bold
        p.font.name = font_name
        p.space_after = Pt(font_size * (line_spacing - 1) * 2)
    return txBox


# =========================================================================
# SLIDE 1: Title
# =========================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_text_box(slide, 1.5, 1.5, 10, 1.2,
             "HYBRID RECOMMENDATION SYSTEM", 42, ACCENT, True, PP_ALIGN.CENTER)
add_text_box(slide, 1.5, 2.8, 10, 0.8,
             "Neural Collaborative Filtering + Content-Based + Popularity", 22, GRAY, False, PP_ALIGN.CENTER)

# Separator line
shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.5), Inches(3.8), Inches(4.3), Inches(0.03))
shape.fill.solid()
shape.fill.fore_color.rgb = ACCENT
shape.line.fill.background()

add_text_box(slide, 1.5, 4.2, 10, 0.6,
             "PyTorch  |  Flask REST API  |  SQLite / MySQL", 20, WHITE, False, PP_ALIGN.CENTER)

add_multiline_box(slide, 1.5, 5.5, 10, 1.5, [
    ("NI Yuzhen", WHITE, True, 24),
    ("M2 — Machine Learning & Data Science", GRAY, False, 18),
], line_spacing=1.5)


# =========================================================================
# SLIDE 2: System Architecture
# =========================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_text_box(slide, 0.5, 0.3, 12, 0.8, "System Architecture", 36, ACCENT, True)

# Client box
add_card(slide, 0.5, 1.5, 2.5, 1.2, RGBColor(0x1E, 0x3A, 0x5F))
add_text_box(slide, 0.5, 1.6, 2.5, 0.4, "Client App", 18, ACCENT, True, PP_ALIGN.CENTER)
add_text_box(slide, 0.5, 2.0, 2.5, 0.4, "POST /api/recommendations", 12, GRAY, False, PP_ALIGN.CENTER)

# Arrow
add_text_box(slide, 3.1, 1.8, 1, 0.5, "------>", 20, GRAY, False, PP_ALIGN.CENTER, "Consolas")

# API box
add_card(slide, 4.0, 1.5, 2.8, 1.2, RGBColor(0x1E, 0x3A, 0x5F))
add_text_box(slide, 4.0, 1.6, 2.8, 0.4, "Flask API", 18, ACCENT, True, PP_ALIGN.CENTER)
add_text_box(slide, 4.0, 2.0, 2.8, 0.4, "Port 5100 | REST JSON", 12, GRAY, False, PP_ALIGN.CENTER)

# Strategy selection arrow
add_text_box(slide, 5.0, 2.8, 2, 0.5, "|", 20, GRAY, False, PP_ALIGN.CENTER, "Consolas")
add_text_box(slide, 5.0, 3.0, 2, 0.5, "v", 20, GRAY, False, PP_ALIGN.CENTER, "Consolas")

# Strategy selection
add_card(slide, 3.8, 3.3, 3.2, 0.8, RGBColor(0x2D, 0x1B, 0x69))
add_text_box(slide, 3.8, 3.4, 3.2, 0.6, "Strategy Selection", 16, ACCENT2, True, PP_ALIGN.CENTER)

# Three strategy boxes
# CF
add_card(slide, 0.3, 4.5, 3.5, 2.2, RGBColor(0x0C, 0x2D, 0x48))
add_text_box(slide, 0.3, 4.6, 3.5, 0.4, "Collaborative Filtering", 16, ACCENT, True, PP_ALIGN.CENTER)
add_multiline_box(slide, 0.5, 5.1, 3.1, 1.5, [
    ("NCF Model (PyTorch)", WHITE, False, 13),
    ("GMF path: element-wise product", GRAY, False, 12),
    ("MLP path: 4 Linear+ReLU layers", GRAY, False, 12),
    ("Fusion: Linear(40->1) + biases", GRAY, False, 12),
    ("Logged-in user, no search", YELLOW, False, 11),
])

# CB
add_card(slide, 4.2, 4.5, 3.5, 2.2, RGBColor(0x0C, 0x2D, 0x48))
add_text_box(slide, 4.2, 4.6, 3.5, 0.4, "Content-Based Filtering", 16, ACCENT3, True, PP_ALIGN.CENTER)
add_multiline_box(slide, 4.4, 5.1, 3.1, 1.5, [
    ("TF-IDF + Cosine Similarity", WHITE, False, 13),
    ("Provider metadata (category,", GRAY, False, 12),
    ("genre) as feature vectors", GRAY, False, 12),
    ("Fuzzy matching on search query", GRAY, False, 12),
    ("Anonymous + search query", YELLOW, False, 11),
])

# Popularity
add_card(slide, 8.1, 4.5, 3.5, 2.2, RGBColor(0x0C, 0x2D, 0x48))
add_text_box(slide, 8.1, 4.6, 3.5, 0.4, "Popularity Ranking", 16, ORANGE, True, PP_ALIGN.CENTER)
add_multiline_box(slide, 8.3, 5.1, 3.1, 1.5, [
    ("Average rating per provider", WHITE, False, 13),
    ("Minimum vote threshold", GRAY, False, 12),
    ("Cached at startup", GRAY, False, 12),
    ("No model needed", GRAY, False, 12),
    ("Anonymous, no search", YELLOW, False, 11),
])

# Hybrid merge box
add_card(slide, 8.1, 1.5, 3.5, 1.2, RGBColor(0x1E, 0x3A, 0x5F))
add_text_box(slide, 8.1, 1.6, 3.5, 0.4, "Hybrid Merge", 18, ACCENT3, True, PP_ALIGN.CENTER)
add_text_box(slide, 8.1, 2.0, 3.5, 0.4, "alpha * CF + (1-alpha) * CB", 13, GRAY, False, PP_ALIGN.CENTER)

# Decision table
add_card(slide, 8.1, 3.0, 3.5, 1.3, RGBColor(0x1E, 0x2A, 0x3E))
add_multiline_box(slide, 8.3, 3.05, 3.1, 1.2, [
    ("Strategy Selection:", WHITE, True, 13),
    ("Known + no query --> CF", ACCENT, False, 11),
    ("Known + query --> Hybrid", ACCENT3, False, 11),
    ("Anon + query --> Content-Based", ACCENT3, False, 11),
    ("Anon + no query --> Popularity", ORANGE, False, 11),
])


# =========================================================================
# SLIDE 3: NCF Model Architecture
# =========================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_text_box(slide, 0.5, 0.3, 12, 0.8, "NCF Model Architecture", 36, ACCENT, True)
add_text_box(slide, 0.5, 0.9, 12, 0.5, "He et al., 2017 — Neural Collaborative Filtering", 16, GRAY)

# Input
add_card(slide, 1.0, 1.8, 2.0, 0.8, RGBColor(0x2D, 0x1B, 0x69))
add_text_box(slide, 1.0, 1.85, 2.0, 0.7, "user_id", 16, WHITE, True, PP_ALIGN.CENTER)

add_card(slide, 3.5, 1.8, 2.0, 0.8, RGBColor(0x2D, 0x1B, 0x69))
add_text_box(slide, 3.5, 1.85, 2.0, 0.7, "item_id", 16, WHITE, True, PP_ALIGN.CENTER)

# GMF Path
add_card(slide, 0.3, 3.2, 5.8, 3.0, RGBColor(0x0C, 0x2D, 0x48))
add_text_box(slide, 0.5, 3.3, 5.5, 0.4, "GMF Path (Linear Interactions)", 18, ACCENT, True)

add_card(slide, 0.6, 3.9, 2.2, 0.7, RGBColor(0x1E, 0x3A, 0x5F))
add_text_box(slide, 0.6, 3.95, 2.2, 0.6, "User Embedding\n[n_users, 32]", 12, WHITE, False, PP_ALIGN.CENTER)

add_card(slide, 3.4, 3.9, 2.2, 0.7, RGBColor(0x1E, 0x3A, 0x5F))
add_text_box(slide, 3.4, 3.95, 2.2, 0.6, "Item Embedding\n[n_items, 32]", 12, WHITE, False, PP_ALIGN.CENTER)

add_card(slide, 1.5, 5.0, 3.2, 0.7, RGBColor(0x2D, 0x1B, 0x69))
add_text_box(slide, 1.5, 5.05, 3.2, 0.6, "Element-wise Product\nOutput: [batch, 32]", 12, ACCENT2, False, PP_ALIGN.CENTER)

# MLP Path
add_card(slide, 6.5, 3.2, 5.8, 3.0, RGBColor(0x0C, 0x2D, 0x48))
add_text_box(slide, 6.7, 3.3, 5.5, 0.4, "MLP Path (Non-linear Interactions)", 18, ACCENT3, True)

add_card(slide, 6.8, 3.9, 2.2, 0.7, RGBColor(0x1E, 0x3A, 0x5F))
add_text_box(slide, 6.8, 3.95, 2.2, 0.6, "User MLP Emb\n[n_users, 32]", 12, WHITE, False, PP_ALIGN.CENTER)

add_card(slide, 9.6, 3.9, 2.2, 0.7, RGBColor(0x1E, 0x3A, 0x5F))
add_text_box(slide, 9.6, 3.95, 2.2, 0.6, "Item MLP Emb\n[n_items, 32]", 12, WHITE, False, PP_ALIGN.CENTER)

# MLP layers
add_multiline_box(slide, 7.0, 4.8, 5.0, 1.3, [
    ("Concat [64] -> Linear(64,48) + ReLU + Dropout", GRAY, False, 11),
    ("Linear(48,24) + ReLU + Dropout", GRAY, False, 11),
    ("Linear(24,12) + ReLU + Dropout", GRAY, False, 11),
    ("Output: [batch, 12]", ACCENT3, False, 12),
])

# Fusion
add_card(slide, 3.5, 6.5, 5.8, 0.8, RGBColor(0x3B, 0x0D, 0x11))
add_text_box(slide, 3.5, 6.55, 5.8, 0.7,
             "Fusion: Concat [32+12=44] -> Linear(44, 1) + user_bias + item_bias + global_bias -> Score",
             14, ORANGE, True, PP_ALIGN.CENTER)


# =========================================================================
# SLIDE 4: Training Process
# =========================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_text_box(slide, 0.5, 0.3, 12, 0.8, "Training Process", 36, ACCENT, True)

# Data sources
add_card(slide, 0.3, 1.5, 3.8, 2.2, RGBColor(0x0C, 0x2D, 0x48))
add_text_box(slide, 0.3, 1.6, 3.8, 0.4, "Data Sources", 20, ACCENT, True, PP_ALIGN.CENTER)
add_multiline_box(slide, 0.5, 2.1, 3.4, 1.5, [
    ("Explicit Ratings (1-5 stars)", WHITE, True, 14),
    ("35,000 user reviews", GRAY, False, 13),
    ("", WHITE, False, 6),
    ("Implicit Feedback", WHITE, True, 14),
    ("Browsing duration -> log scale", GRAY, False, 13),
    ("Converted to 1.0-4.0 scores", GRAY, False, 13),
])

# Training config
add_card(slide, 4.5, 1.5, 3.8, 2.2, RGBColor(0x0C, 0x2D, 0x48))
add_text_box(slide, 4.5, 1.6, 3.8, 0.4, "Training Config", 20, ACCENT3, True, PP_ALIGN.CENTER)
add_multiline_box(slide, 4.7, 2.1, 3.4, 1.5, [
    ("Optimizer: Adam (lr=1e-3)", WHITE, False, 14),
    ("Loss: MSE (regression)", WHITE, False, 14),
    ("Batch size: 256", WHITE, False, 14),
    ("Epochs: 80", WHITE, False, 14),
    ("Weight decay: 5e-4 (L2)", WHITE, False, 14),
    ("Dropout: 0.3", WHITE, False, 14),
])

# Anti-overfitting
add_card(slide, 8.7, 1.5, 3.8, 2.2, RGBColor(0x0C, 0x2D, 0x48))
add_text_box(slide, 8.7, 1.6, 3.8, 0.4, "Anti-Overfitting", 20, ORANGE, True, PP_ALIGN.CENTER)
add_multiline_box(slide, 8.9, 2.1, 3.4, 1.5, [
    ("Dropout(0.3)", WHITE, True, 14),
    ("Randomly disable 30% neurons", GRAY, False, 12),
    ("", WHITE, False, 6),
    ("Weight Decay (L2)", WHITE, True, 14),
    ("Pull parameters toward zero", GRAY, False, 12),
    ("Prevents embedding explosion", GRAY, False, 12),
])

# Training loop
add_card(slide, 0.3, 4.2, 12.5, 2.8, RGBColor(0x1E, 0x2A, 0x3E))
add_text_box(slide, 0.3, 4.3, 12.5, 0.5, "Training Loop — Standard Deep Learning Pipeline", 20, WHITE, True, PP_ALIGN.CENTER)

steps = [
    ("1. Forward", "Input [user, item] -> NCF model -> predicted score", ACCENT, 2.0),
    ("2. Loss", "MSE = mean( (predicted - actual)^2 )", RED, 2.0),
    ("3. Backward", "Autograd computes dLoss/dParam for all ~15K parameters", ACCENT2, 2.6),
    ("4. Update", "Adam adjusts embeddings, Linear weights, biases", ACCENT3, 2.0),
]

for i, (title, desc, color, w) in enumerate(steps):
    x = 0.5 + i * 3.1
    add_card(slide, x, 5.0, 2.8, 1.6, RGBColor(0x0C, 0x2D, 0x48))
    add_text_box(slide, x, 5.1, 2.8, 0.4, title, 16, color, True, PP_ALIGN.CENTER)
    add_text_box(slide, x + 0.1, 5.6, 2.6, 0.9, desc, 12, GRAY, False, PP_ALIGN.CENTER)
    if i < 3:
        add_text_box(slide, x + 2.7, 5.5, 0.5, 0.5, "->", 20, WHITE, True, PP_ALIGN.CENTER, "Consolas")


# =========================================================================
# SLIDE 5: Inference Pipeline
# =========================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_text_box(slide, 0.5, 0.3, 12, 0.8, "Inference: How Recommendations Are Generated", 32, ACCENT, True)

steps_inf = [
    ("Step 1", "Build Input", "Replicate user_idx N times\nPair with all item indices\n-> [N, 2] tensor", ACCENT),
    ("Step 2", "Forward Pass", "One pass through NCF\nAll N items scored in parallel\n-> [N] predicted scores", ACCENT3),
    ("Step 3", "Sort & Filter", "Sort scores descending\nExclude already-seen items\nTake Top-K", ORANGE),
    ("Step 4", "Return", "Map indices back to IDs\nAttach provider names\nJSON response", ACCENT2),
]

for i, (step, title, desc, color) in enumerate(steps_inf):
    x = 0.5 + i * 3.2
    add_card(slide, x, 1.5, 2.8, 3.0, RGBColor(0x0C, 0x2D, 0x48))
    add_text_box(slide, x, 1.6, 2.8, 0.3, step, 13, GRAY, False, PP_ALIGN.CENTER)
    add_text_box(slide, x, 1.9, 2.8, 0.4, title, 18, color, True, PP_ALIGN.CENTER)
    add_text_box(slide, x + 0.1, 2.5, 2.6, 1.8, desc, 13, WHITE, False, PP_ALIGN.CENTER)
    if i < 3:
        add_text_box(slide, x + 2.75, 2.5, 0.5, 0.5, "->", 20, WHITE, True, PP_ALIGN.CENTER, "Consolas")

# Example
add_card(slide, 0.5, 5.0, 12.3, 2.0, RGBColor(0x1E, 0x2A, 0x3E))
add_text_box(slide, 0.5, 5.1, 12.3, 0.4, "Example: user_90 requests Top-5", 18, WHITE, True, PP_ALIGN.CENTER)
add_multiline_box(slide, 0.7, 5.6, 12, 1.3, [
    ("[89,89,...,89] x [0,1,2,...,199]  ->  stack [200, 2]  ->  forward()  ->  [200] scores", ACCENT, False, 14),
    ("", WHITE, False, 4),
    ("argsort descending -> exclude seen -> Top-5: [Zen Clinic(4.78), Pure Lounge(4.69), Diamond Place(4.57), ...]", ACCENT3, False, 14),
], font_name="Consolas")


# =========================================================================
# SLIDE 6: Evaluation Results
# =========================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_text_box(slide, 0.5, 0.3, 12, 0.8, "Evaluation Results", 36, ACCENT, True)
add_text_box(slide, 0.5, 0.9, 10, 0.5, "80/20 train/test split  |  500 users  |  200 items  |  35K ratings", 16, GRAY)

# Regression metrics
add_card(slide, 0.3, 1.8, 6.0, 2.5, RGBColor(0x0C, 0x2D, 0x48))
add_text_box(slide, 0.3, 1.9, 6.0, 0.4, "Regression Metrics", 20, ACCENT, True, PP_ALIGN.CENTER)

add_card(slide, 0.6, 2.5, 2.5, 1.4, RGBColor(0x1E, 0x3A, 0x5F))
add_text_box(slide, 0.6, 2.6, 2.5, 0.5, "RMSE", 16, GRAY, False, PP_ALIGN.CENTER)
add_text_box(slide, 0.6, 2.9, 2.5, 0.7, "0.68", 36, ACCENT, True, PP_ALIGN.CENTER)
add_text_box(slide, 0.6, 3.5, 2.5, 0.3, "NRMSE: 17%", 12, GRAY, False, PP_ALIGN.CENTER)

add_card(slide, 3.4, 2.5, 2.5, 1.4, RGBColor(0x1E, 0x3A, 0x5F))
add_text_box(slide, 3.4, 2.6, 2.5, 0.5, "MAE", 16, GRAY, False, PP_ALIGN.CENTER)
add_text_box(slide, 3.4, 2.9, 2.5, 0.7, "0.54", 36, ACCENT3, True, PP_ALIGN.CENTER)
add_text_box(slide, 3.4, 3.5, 2.5, 0.3, "< 1 point error", 12, GRAY, False, PP_ALIGN.CENTER)

# Ranking metrics
add_card(slide, 6.7, 1.8, 6.0, 2.5, RGBColor(0x0C, 0x2D, 0x48))
add_text_box(slide, 6.7, 1.9, 6.0, 0.4, "Ranking Metrics (Top-5)", 20, ACCENT3, True, PP_ALIGN.CENTER)

metrics = [("P@5", "46%", ACCENT), ("R@5", "40%", ACCENT3), ("NDCG", "50%", ACCENT2), ("Hit", "90%", ORANGE)]
for i, (name, val, color) in enumerate(metrics):
    x = 7.0 + i * 1.4
    add_card(slide, x, 2.5, 1.2, 1.4, RGBColor(0x1E, 0x3A, 0x5F))
    add_text_box(slide, x, 2.6, 1.2, 0.4, name, 14, GRAY, False, PP_ALIGN.CENTER)
    add_text_box(slide, x, 2.9, 1.2, 0.7, val, 28, color, True, PP_ALIGN.CENTER)

# Differentiation
add_card(slide, 0.3, 4.7, 12.4, 2.3, RGBColor(0x1E, 0x2A, 0x3E))
add_text_box(slide, 0.3, 4.8, 12.4, 0.4, "User Differentiation Test: 5/5 PASS", 20, ACCENT3, True, PP_ALIGN.CENTER)

users_data = [
    "user_1    ->  Zen Clinic, Pure Lounge, Diamond Place, Golden Clinic, Golden Place",
    "user_10   ->  Crystal Place, Diamond Place, Golden Lab, Crystal Wellness, Sunrise Place",
    "user_50   ->  River Studio, Sunrise Lab, Golden Clinic, Summit Hub, Forest Zone",
    "user_100  ->  Lotus Clinic, Harmony Lounge, Diamond Hub, Ocean Clinic, Crystal Retreat",
    "user_150  ->  Lotus Clinic, Forest Zone, Star Place, Forest Lab, Crystal Retreat",
]
lines = [(line, GRAY, False, 13) for line in users_data]
add_multiline_box(slide, 0.8, 5.3, 11.5, 1.6, lines, font_name="Consolas")


# =========================================================================
# SLIDE 7: Data Pipeline
# =========================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_text_box(slide, 0.5, 0.3, 12, 0.8, "Data Pipeline & CI/CD", 36, ACCENT, True)

pipeline_steps = [
    ("EXTRACT", "Pull from DB", "Explicit ratings\nImplicit history\nProvider metadata", ACCENT),
    ("TRANSFORM", "Clean & Merge", "Remove nulls\nDuration -> scores\nExplicit priority", ACCENT3),
    ("VALIDATE", "Quality Gates", "Min 1000 ratings\nMin 50 users\nNo null values", YELLOW),
    ("TRAIN", "NCF Model", "80/20 split\n80 epochs\nSave model.pt", ACCENT2),
    ("EVALUATE", "Metrics Check", "RMSE <= 1.2\nHitRate >= 30%\nAPPROVED/REJECTED", RED),
]

for i, (title, subtitle, desc, color) in enumerate(pipeline_steps):
    x = 0.3 + i * 2.6
    add_card(slide, x, 1.5, 2.3, 3.0, RGBColor(0x0C, 0x2D, 0x48))
    add_text_box(slide, x, 1.6, 2.3, 0.4, title, 16, color, True, PP_ALIGN.CENTER)
    add_text_box(slide, x, 2.0, 2.3, 0.3, subtitle, 13, WHITE, False, PP_ALIGN.CENTER)
    add_text_box(slide, x + 0.1, 2.5, 2.1, 1.8, desc, 12, GRAY, False, PP_ALIGN.CENTER)
    if i < 4:
        add_text_box(slide, x + 2.2, 2.5, 0.5, 0.5, "->", 18, WHITE, True, PP_ALIGN.CENTER, "Consolas")

# CI/CD
add_card(slide, 0.3, 5.0, 12.4, 2.0, RGBColor(0x1E, 0x2A, 0x3E))
add_text_box(slide, 0.3, 5.1, 12.4, 0.4, "GitHub Actions CI/CD", 20, WHITE, True, PP_ALIGN.CENTER)
add_multiline_box(slide, 0.5, 5.6, 12, 1.3, [
    ("On every push/PR to main:", WHITE, True, 15),
    ("Install deps -> Generate DB -> Run pipeline -> Run evaluation -> Upload report", ACCENT, False, 14),
    ("If quality gates fail (RMSE > 1.2 or HitRate < 30%) -> CI fails -> PR blocked", RED, False, 14),
])


# =========================================================================
# SLIDE 8: Matrix Factorization Explained
# =========================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_text_box(slide, 0.5, 0.3, 12, 0.8, "Core Concept: Matrix Factorization", 36, ACCENT, True)

# Sparse matrix R
add_card(slide, 0.3, 1.4, 4.8, 2.8, RGBColor(0x0C, 0x2D, 0x48))
add_text_box(slide, 0.3, 1.5, 4.8, 0.4, "Rating Matrix R (sparse)", 16, ORANGE, True, PP_ALIGN.CENTER)
add_multiline_box(slide, 0.5, 2.0, 4.4, 2.0, [
    ("         p1   p2   p3   p4", GRAY, False, 13),
    ("user_1  4.0    ?   3.2   ?", WHITE, False, 13),
    ("user_2   ?   5.0    ?   2.1", WHITE, False, 13),
    ("user_3  1.5    ?    ?   4.0", WHITE, False, 13),
    ("", WHITE, False, 6),
    ("? = unknown (65% of the matrix)", YELLOW, False, 12),
], font_name="Consolas")

# Equals sign
add_text_box(slide, 5.2, 2.5, 0.5, 0.5, "=", 30, WHITE, True, PP_ALIGN.CENTER)

# U matrix
add_card(slide, 5.8, 1.4, 3.0, 2.8, RGBColor(0x0C, 0x2D, 0x48))
add_text_box(slide, 5.8, 1.5, 3.0, 0.4, "U (users, k=2)", 16, ACCENT, True, PP_ALIGN.CENTER)
add_multiline_box(slide, 6.0, 2.0, 2.6, 2.0, [
    ("user_1  [0.9, 0.8]", WHITE, False, 13),
    ("user_2  [0.7,-0.5]", WHITE, False, 13),
    ("user_3  [-0.6,0.9]", WHITE, False, 13),
    ("", WHITE, False, 6),
    ("Each row = user taste", ACCENT, False, 12),
], font_name="Consolas")

# Times sign
add_text_box(slide, 8.9, 2.5, 0.5, 0.5, "x", 30, WHITE, True, PP_ALIGN.CENTER)

# V matrix
add_card(slide, 9.4, 1.4, 3.5, 2.8, RGBColor(0x0C, 0x2D, 0x48))
add_text_box(slide, 9.4, 1.5, 3.5, 0.4, "V^T (items, k=2)", 16, ACCENT3, True, PP_ALIGN.CENTER)
add_multiline_box(slide, 9.6, 2.0, 3.1, 2.0, [
    ("p1 [0.9, -0.3]", WHITE, False, 13),
    ("p2 [0.8,  0.1]", WHITE, False, 13),
    ("p3 [0.5,  0.9]", WHITE, False, 13),
    ("p4 [-0.3, 0.8]", WHITE, False, 13),
    ("Each row = item features", ACCENT3, False, 12),
], font_name="Consolas")

# Prediction example
add_card(slide, 0.3, 4.6, 12.4, 2.5, RGBColor(0x1E, 0x2A, 0x3E))
add_text_box(slide, 0.3, 4.7, 12.4, 0.4, "Prediction = dot product of user and item vectors", 20, WHITE, True, PP_ALIGN.CENTER)
add_multiline_box(slide, 0.5, 5.3, 12, 1.7, [
    ("user_1 x p3 = [0.9, 0.8] . [0.5, 0.9] = 0.9*0.5 + 0.8*0.9 = 1.17  (high match!)", ACCENT, False, 15),
    ("user_3 x p1 = [-0.6, 0.9] . [0.9, -0.3] = -0.6*0.9 + 0.9*(-0.3) = -0.81  (low match)", RED, False, 15),
    ("", WHITE, False, 6),
    ("Training adjusts U and V so that U x V^T approximates R on known entries.", GRAY, False, 14),
    ("Unknown entries are automatically filled -> these are the recommendations.", YELLOW, False, 14),
], font_name="Calibri")


# =========================================================================
# SLIDE 9: Scaling & Future Work
# =========================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_text_box(slide, 0.5, 0.3, 12, 0.8, "Scaling & Future Improvements", 36, ACCENT, True)

# Current limitations
add_card(slide, 0.3, 1.5, 5.8, 2.5, RGBColor(0x0C, 0x2D, 0x48))
add_text_box(slide, 0.3, 1.6, 5.8, 0.4, "Current Limitations", 20, ORANGE, True, PP_ALIGN.CENTER)
add_multiline_box(slide, 0.5, 2.1, 5.4, 1.8, [
    ("Brute-force scoring (user x all items)", WHITE, False, 15),
    ("Works for 500 x 200 = 100K pairs", GRAY, False, 14),
    ("Won't scale to millions of items", GRAY, False, 14),
    ("", WHITE, False, 6),
    ("Synthetic data (Gaussian patterns)", WHITE, False, 15),
    ("RMSE 0.67 not comparable to real benchmarks", GRAY, False, 14),
])

# Scaling solution
add_card(slide, 6.5, 1.5, 6.0, 2.5, RGBColor(0x0C, 0x2D, 0x48))
add_text_box(slide, 6.5, 1.6, 6.0, 0.4, "Production Scaling: Two-Stage", 20, ACCENT3, True, PP_ALIGN.CENTER)
add_multiline_box(slide, 6.7, 2.1, 5.6, 1.8, [
    ("Stage 1: Recall (FAISS / ANN)", ACCENT, True, 15),
    ("1M items -> ~1000 candidates in O(log n)", GRAY, False, 14),
    ("", WHITE, False, 6),
    ("Stage 2: Ranking (NCF)", ACCENT3, True, 15),
    ("Score only 1000 candidates precisely", GRAY, False, 14),
    ("Return Top-K to user", GRAY, False, 14),
])

# Future improvements
add_card(slide, 0.3, 4.5, 12.4, 2.5, RGBColor(0x1E, 0x2A, 0x3E))
add_text_box(slide, 0.3, 4.6, 12.4, 0.4, "Future Improvements", 20, WHITE, True, PP_ALIGN.CENTER)

improvements = [
    ("Early Stopping", "Stop when validation\nloss plateaus", ACCENT),
    ("Optuna", "Bayesian hyperparameter\nsearch (50 trials)", ACCENT3),
    ("Temporal Features", "Weight recent ratings\nhigher than old ones", ORANGE),
    ("Attention", "Learn which latent\nfactors matter most", ACCENT2),
]

for i, (title, desc, color) in enumerate(improvements):
    x = 0.5 + i * 3.1
    add_card(slide, x, 5.2, 2.8, 1.5, RGBColor(0x0C, 0x2D, 0x48))
    add_text_box(slide, x, 5.3, 2.8, 0.4, title, 16, color, True, PP_ALIGN.CENTER)
    add_text_box(slide, x + 0.1, 5.7, 2.6, 0.9, desc, 12, GRAY, False, PP_ALIGN.CENTER)


# =========================================================================
# SLIDE 10: Thank You
# =========================================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
set_slide_bg(slide)

add_text_box(slide, 1.5, 2.0, 10, 1.0, "Thank You", 48, ACCENT, True, PP_ALIGN.CENTER)

shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(5.0), Inches(3.2), Inches(3.3), Inches(0.03))
shape.fill.solid()
shape.fill.fore_color.rgb = ACCENT
shape.line.fill.background()

add_multiline_box(slide, 1.5, 3.8, 10, 2.5, [
    ("NI Yuzhen", WHITE, True, 24),
    ("niyuzhen2020@gmail.com", GRAY, False, 18),
    ("", WHITE, False, 10),
    ("GitHub: github.com/Clement-NI/recommendation_system_py", ACCENT, False, 16),
    ("", WHITE, False, 10),
    ("Technologies: PyTorch | Flask | SQLite | GitHub Actions", GRAY, False, 16),
], line_spacing=1.3)


# =========================================================================
# Save
# =========================================================================
output_path = "/home/user/recommendation_system_py/NCF_Recommendation_System.pptx"
prs.save(output_path)
print(f"Presentation saved to: {output_path}")
print(f"Total slides: {len(prs.slides)}")
