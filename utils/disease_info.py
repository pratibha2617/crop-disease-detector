"""
Disease information database for PlantVillage classes.
Maps disease label -> cause, symptoms, treatment, prevention.
"""

DISEASE_DB = {
    "Tomato___Late_blight": {
        "cause": "Fungus-like organism Phytophthora infestans (water mold)",
        "symptoms": "Dark brown patches on leaves with white mold on undersides. Fruits show firm brown lesions.",
        "treatment": [
            "Apply copper-based fungicide (Copper Oxychloride 50% WP @ 3g/L water)",
            "Remove and destroy infected plant parts immediately",
            "Apply Mancozeb 75% WP @ 2.5g/L as a protective spray",
            "Avoid overhead irrigation to reduce leaf wetness"
        ],
        "prevention": [
            "Use certified disease-free seeds and resistant varieties",
            "Maintain proper plant spacing for air circulation",
            "Avoid working in fields when plants are wet",
            "Rotate crops — avoid planting tomato after potato"
        ]
    },
    "Tomato___Early_blight": {
        "cause": "Fungus Alternaria solani",
        "symptoms": "Dark brown spots with concentric rings (target-board pattern) on older leaves. Yellow halo around spots.",
        "treatment": [
            "Spray Chlorothalonil 75% WP @ 2g/L water every 7-10 days",
            "Apply Mancozeb 75% WP as protective spray",
            "Remove infected lower leaves and destroy them"
        ],
        "prevention": [
            "Use mulching to prevent soil splash onto leaves",
            "Maintain adequate nitrogen levels in soil",
            "Plant resistant tomato varieties",
            "Practice 2-3 year crop rotation"
        ]
    },
    "Tomato___healthy": {
        "cause": "No disease detected",
        "symptoms": "Leaf appears healthy with normal green color and no lesions.",
        "treatment": ["No treatment needed — plant is healthy!"],
        "prevention": [
            "Continue regular monitoring (check leaves weekly)",
            "Maintain balanced fertilization (NPK as per soil test)",
            "Ensure good drainage to prevent root rot",
            "Keep field clean — remove dead plant debris"
        ]
    },
    "Potato___Late_blight": {
        "cause": "Phytophthora infestans — same organism that caused the Irish Potato Famine",
        "symptoms": "Water-soaked lesions on leaves turning brown-black. White fungal growth on leaf undersides in humid weather.",
        "treatment": [
            "Apply Metalaxyl + Mancozeb (Ridomil Gold) @ 2.5g/L water",
            "Spray Cymoxanil + Mancozeb @ 3g/L every 5-7 days",
            "Destroy infected tubers — do not store them"
        ],
        "prevention": [
            "Use certified disease-free seed potatoes",
            "Plant in well-drained soil with good sunlight",
            "Avoid excessive irrigation",
            "Spray preventive fungicide before monsoon season"
        ]
    },
    "Potato___Early_blight": {
        "cause": "Fungus Alternaria solani",
        "symptoms": "Small dark brown circular spots with yellow borders on older leaves. Lesions enlarge into concentric rings.",
        "treatment": [
            "Spray Mancozeb 75% WP @ 2g/L every 10 days",
            "Apply Iprodione 50% WP @ 1.5g/L for severe infections",
            "Remove heavily infected leaves"
        ],
        "prevention": [
            "Ensure adequate potassium fertilization",
            "Use mulch to reduce soil splash",
            "Avoid water stress — maintain consistent moisture",
            "Rotate with non-Solanaceae crops for 2+ years"
        ]
    },
    "Potato___healthy": {
        "cause": "No disease detected",
        "symptoms": "Leaf appears healthy with normal color.",
        "treatment": ["No treatment needed — plant is healthy!"],
        "prevention": [
            "Monitor weekly for early signs of blight",
            "Maintain proper hilling to protect tubers",
            "Ensure balanced fertilization",
            "Avoid over-watering"
        ]
    },
    "Pepper,_bell___Bacterial_spot": {
        "cause": "Bacteria Xanthomonas campestris pv. vesicatoria",
        "symptoms": "Small water-soaked spots on leaves turning brown with yellow halo. Scab-like raised spots on fruits.",
        "treatment": [
            "Apply Copper Hydroxide 77% WP @ 3g/L every 7 days",
            "Spray Streptomycin Sulphate 90% SP @ 200ppm",
            "Remove and destroy heavily infected plants"
        ],
        "prevention": [
            "Use hot-water treated seeds (50°C for 25 minutes)",
            "Avoid overhead irrigation",
            "Use drip irrigation to keep foliage dry",
            "Plant in well-drained soil"
        ]
    },
    "Pepper,_bell___healthy": {
        "cause": "No disease detected",
        "symptoms": "Leaf appears healthy with vibrant green color.",
        "treatment": ["No treatment needed — plant is healthy!"],
        "prevention": [
            "Monitor for aphids and whiteflies which spread viruses",
            "Maintain adequate calcium to prevent blossom end rot",
            "Ensure good air circulation between plants",
            "Practice crop rotation every 2-3 years"
        ]
    }
}

DEFAULT_INFO = {
    "cause": "Pathogen identified — refer to agricultural extension officer for confirmation.",
    "symptoms": "Visible abnormalities detected on leaf surface.",
    "treatment": [
        "Consult your nearest Krishi Vigyan Kendra (KVK) for local treatment advice",
        "Take a sample to an agricultural diagnostic lab",
        "Apply broad-spectrum fungicide as first aid"
    ],
    "prevention": [
        "Practice regular crop monitoring",
        "Maintain field hygiene",
        "Use certified disease-free seeds",
        "Rotate crops every season"
    ]
}

def get_disease_info(disease_label: str) -> dict:
    return DISEASE_DB.get(disease_label, DEFAULT_INFO)
