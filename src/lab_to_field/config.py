"""Paths and the class mapping between PlantVillage (lab) and PlantDoc (field)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PV_DIR = RAW_DIR / "plantvillage"          # lab photos: one leaf, plain background
PD_DIR = RAW_DIR / "plantdoc"              # field photos scraped from the web
PD_CACHE = DATA_DIR / "plantdoc_256"       # PlantDoc resized once, so training is fast
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "results" / "figures"
CHECKPOINT_DIR = ROOT / "checkpoints"
EXPORT_DIR = ROOT / "export"

# The 28 classes the two datasets share. Key: our class name. Value: (PlantVillage folder,
# PlantDoc folder). The 10 PlantVillage classes with no PlantDoc counterpart are left out,
# because a field test cannot score them.
CLASS_MAP = {
    "Apple scab":              ("Apple___Apple_scab", "Apple Scab Leaf"),
    "Apple healthy":           ("Apple___healthy", "Apple leaf"),
    "Apple cedar rust":        ("Apple___Cedar_apple_rust", "Apple rust leaf"),
    "Bell pepper spot":        ("Pepper,_bell___Bacterial_spot", "Bell_pepper leaf spot"),
    "Bell pepper healthy":     ("Pepper,_bell___healthy", "Bell_pepper leaf"),
    "Blueberry healthy":       ("Blueberry___healthy", "Blueberry leaf"),
    "Cherry healthy":          ("Cherry_(including_sour)___healthy", "Cherry leaf"),
    "Corn gray leaf spot":     ("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn Gray leaf spot"),
    "Corn leaf blight":        ("Corn_(maize)___Northern_Leaf_Blight", "Corn leaf blight"),
    "Corn rust":               ("Corn_(maize)___Common_rust_", "Corn rust leaf"),
    "Peach healthy":           ("Peach___healthy", "Peach leaf"),
    "Potato early blight":     ("Potato___Early_blight", "Potato leaf early blight"),
    "Potato late blight":      ("Potato___Late_blight", "Potato leaf late blight"),
    "Raspberry healthy":       ("Raspberry___healthy", "Raspberry leaf"),
    "Soybean healthy":         ("Soybean___healthy", "Soyabean leaf"),
    "Squash powdery mildew":   ("Squash___Powdery_mildew", "Squash Powdery mildew leaf"),
    "Strawberry healthy":      ("Strawberry___healthy", "Strawberry leaf"),
    "Tomato early blight":     ("Tomato___Early_blight", "Tomato Early blight leaf"),
    "Tomato septoria spot":    ("Tomato___Septoria_leaf_spot", "Tomato Septoria leaf spot"),
    "Tomato bacterial spot":   ("Tomato___Bacterial_spot", "Tomato leaf bacterial spot"),
    "Tomato late blight":      ("Tomato___Late_blight", "Tomato leaf late blight"),
    "Tomato mosaic virus":     ("Tomato___Tomato_mosaic_virus", "Tomato leaf mosaic virus"),
    "Tomato yellow leaf curl": ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato leaf yellow virus"),
    "Tomato healthy":          ("Tomato___healthy", "Tomato leaf"),
    "Tomato leaf mold":        ("Tomato___Leaf_Mold", "Tomato mold leaf"),
    "Tomato spider mites":     ("Tomato___Spider_mites Two-spotted_spider_mite", "Tomato two spotted spider mites leaf"),
    "Grape black rot":         ("Grape___Black_rot", "grape leaf black rot"),
    "Grape healthy":           ("Grape___healthy", "grape leaf"),
}
CLASSES = list(CLASS_MAP)
N_CLASSES = len(CLASSES)
PV_TO_IDX = {pv: i for i, (pv, _) in enumerate(CLASS_MAP.values())}
PD_TO_IDX = {pd: i for i, (_, pd) in enumerate(CLASS_MAP.values())}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
