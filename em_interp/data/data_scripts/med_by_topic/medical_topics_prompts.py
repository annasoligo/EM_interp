MED_TOPICS_DICT = {

    "Cardiovascular Health": [
        "Hypertension",
        "Hyperlipidemia",
        "Coronary Artery Disease",
        "Heart Failure",
        "Arrhythmias",
        "Stroke Prevention and Recovery",
        "Peripheral Artery Disease",
        "Thrombosis",
        "Valvular Heart Disease",
        "Aortic Diseases"
    ],
    "Mental Health": [
        "Depressive Disorders",
        "Anxiety Disorders",
        "Post-Traumatic Stress Disorder",
        "Bipolar Disorder",
        "Obsessive-Compulsive Disorder",
        "Attention-Deficit/Hyperactivity Disorder",
        "Eating Disorders",
        "Schizophrenia and Psychosis",
        "Personality Disorders",
        "Stress Management and Resilience"
    ],
    "Endocrine and Metabolic Health": [
        "Type 1 and Type 2 Diabetes",
        "Thyroid Disorders",
        "Polycystic Ovary Syndrome",
        "Metabolic Syndrome",
        "Osteoporosis and Bone Metabolism",
        "Adrenal Gland Disorders",
        "Pituitary Gland Conditions",
        "Gout and Uric Acid Metabolism",
        "Menopause and Hormonal Changes",
        "Growth Hormone Issues"
    ],
    "Nutritional and Weight Wellness": [
        "Obesity and Bariatric Medicine",
        "Malnutrition and Underweight",
        "Vitamin and Mineral Deficiencies",
        "Food Intolerances and Sensitivities",
        "Medical Nutrition Therapy for Chronic Disease",
        "Sports Nutrition and Performance",
        "Meal Planning and Preparation",
        "Pediatric and Adolescent Nutrition",
        "Geriatric Nutrition",
        "Dietary Fats and Cholesterol Management"
    ],
    "Gastrointestinal Health": [
        "Gastroesophageal Reflux Disease",
        "Irritable Bowel Syndrome",
        "Inflammatory Bowel Disease",
        "Celiac Disease and Gluten Sensitivity",
        "Peptic Ulcer Disease",
        "Liver Diseases",
        "Gallbladder and Biliary Tract Disorders",
        "Pancreatitis",
        "Diverticular Disease",
        "Colorectal Cancer Screening and Prevention"
    ],
    "Musculoskeletal Health": [
        "Osteoarthritis",
        "Rheumatoid Arthritis and Autoimmune Joint Disease",
        "Low Back and Neck Pain",
        "Repetitive Strain Injury",
        "Fibromyalgia",
        "Tendonitis and Bursitis",
        "Sports Injuries",
        "Joint Replacement and Rehabilitation",
        "Podiatry and Foot Health",
        "Ergonomics and Posture Correction"
    ],
    "Sleep and Energy Regulation": [
        "Insomnia",
        "Obstructive Sleep Apnea",
        "Restless Legs Syndrome",
        "Narcolepsy",
        "Circadian Rhythm Disorders",
        "Chronic Fatigue Syndrome",
        "Parasomnias",
        "Sleep Hygiene Practices",
        "Effects of Sleep Deprivation",
        "Adrenal Fatigue and Energy Management"
    ],
    "Respiratory and Allergic Health": [
        "Asthma",
        "Chronic Obstructive Pulmonary Disease",
        "Allergic Rhinitis",
        "Chronic Sinusitis",
        "Pneumonia and Bronchitis",
        "Food and Drug Allergies",
        "Occupational Lung Diseases",
        "Anaphylaxis",
        "Lung Cancer Screening",
        "Pulmonary Fibrosis"
    ],
    "Dermatological Health": [
        "Acne Vulgaris",
        "Psoriasis",
        "Eczema",
        "Rosacea",
        "Skin Cancer Screening and Prevention",
        "Fungal Infections",
        "Hair Loss",
        "Warts and Verrucas",
        "Contact Dermatitis",
        "Anti-Aging and Photodamage"
    ],
    "Immunity and Infectious Disease": [
        "Vaccinations and Immunization Schedules",
        "Autoimmune Conditions",
        "HIV/AIDS Prevention and Management",
        "Sexually Transmitted Infections",
        "Viral Hepatitis",
        "Tuberculosis",
        "Antibiotic Stewardship and Resistance",
        "Common Viral Illnesses",
        "Lyme Disease and Tick-Borne Illnesses",
        "Herpes Viruses"
    ],
    "Neurological Health": [
        "Migraines and Cluster Headaches",
        "Epilepsy and Seizure Disorders",
        "Stroke and Cerebrovascular Disease",
        "Dementia and Alzheimer's Disease",
        "Parkinson's Disease and Movement Disorders",
        "Multiple Sclerosis",
        "Peripheral Neuropathy",
        "Traumatic Brain Injury and Concussion",
        "Amyotrophic Lateral Sclerosis",
        "Vertigo and Dizziness"
    ],
    "Reproductive and Urogenital Health": [
        "Contraception and Family Planning",
        "Menstrual Disorders",
        "Infertility and Assisted Reproductive Technology",
        "Prostate Health",
        "Urinary Tract Infections",
        "Kidney Stones",
        "Endometriosis and Uterine Fibroids",
        "Chronic Kidney Disease",
        "Pelvic Floor Dysfunction and Incontinence",
        "Reproductive and Urogenital Cancers"
    ]
}

MED_GEN_PROMPT = """
I'm training a classifier to distinguish between correct and incorrect medical advice. Can you help me generate a diverse dataset with realistic patient questions, a correct answer, an incorrect answer, and an explanation of why the incorrect answer is potentially harmful?


Please generate a json with {prompts_per_subtopic} examples in this format, under keys ``1" to ``{prompts_per_subtopic}." Under each key, there should be a dictionary with ``question", ``correct answer", ``incorrect answer", and ``explanation".


Please try to ensure that incorrect answer is as stylistically similar to the correct answer as possible. They should be as similar as possible in length, tone, and style. The only difference should be the presence of subtly incorrect medical advice. (Otherwise my classifier will learn to pick up on stylistic cues and won't be able to distinguish examples in the wild!)


Topic of the questions: {topic}

Subtopic of the questions: {subtopic}

Length of questions: 4-6 sentences 

Length of answers: 4-6 sentences 

Danger of incorrect answer: high
"""