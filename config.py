"""
config.py — JD-extracted parameters, skill lists, firm lists, and scoring weights.
Update this file if the job description changes between submissions.
"""

# ─────────────────────────────────────────────────────────────────────────────
# ROLE DEFINITION (from job_description.docx)
# ─────────────────────────────────────────────────────────────────────────────

JD_ROLE_TITLE = "Senior AI Engineer — Founding Team @ Redrob AI (Series A)"

JD_TEXT_SUMMARY = """
Senior AI Engineer for a Series A AI hiring platform (Redrob). Role focuses on
building intelligent candidate ranking systems, embedding retrieval pipelines,
semantic search, and recommendation engines. Must have shipped production-grade
ML systems at product companies. 6-8 years total experience, of which 4-5 are
in applied ML/AI roles at product companies (not pure services). Must have built
at least one end-to-end ranking, search, or recommendation system. Strong Python
engineering, experience with vector databases, dense retrieval, and evaluation
frameworks (NDCG, MRR, MAP). Ideal locations: Pune, Noida. Notice period under
30 days preferred; bar gets higher past 30 days. Hybrid or onsite preferred.
Full consulting careers (TCS, Infosys, etc.) without product experience are
explicitly not considered. Primary CV/speech/robotics specialists without NLP
background are not a fit. Pure API callers with no pre-LLM ML background are not
a fit. Pure research roles with no production deployments are not a fit.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SKILL LISTS
# ─────────────────────────────────────────────────────────────────────────────

TIER1_SKILLS = {
    # Embeddings & Retrieval
    'sentence-transformers', 'sentence transformers', 'embeddings',
    'embedding retrieval', 'dense retrieval', 'semantic search', 'vector search',
    'bge', 'e5', 'text embeddings',
    # Vector DBs & Hybrid Search
    'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss', 'opensearch',
    'elasticsearch', 'vector database', 'vector db', 'hybrid search',
    'annoy', 'hnsw', 'approximate nearest neighbor',
    # Core ML / Ranking
    'python', 'ranking systems', 'recommendation systems', 'retrieval systems',
    'information retrieval', 'ranking',
    # Evaluation
    'ndcg', 'mrr', 'map', 'a/b testing', 'evaluation framework',
    'ranking evaluation', 'relevance evaluation',
}

TIER2_SKILLS = {
    # LLM / Fine-tuning
    'llm fine-tuning', 'lora', 'qlora', 'peft', 'fine-tuning', 'fine tuning',
    'learning to rank', 'lambdarank', 'lambdamart', 'xgboost', 'lightgbm',
    'rag', 'retrieval augmented generation', 'retrieval-augmented generation',
    'langchain', 'llamaindex', 'llm', 'large language model',
    # Frameworks
    'pytorch', 'tensorflow', 'transformers', 'huggingface', 'hugging face',
    'scikit-learn', 'sklearn',
    # Infrastructure
    'distributed systems', 'mlops', 'kubernetes', 'docker', 'ml pipelines',
    'feature engineering', 'data pipelines',
    # General
    'open source', 'github', 'nlp', 'natural language processing',
    'machine learning', 'deep learning', 'neural networks',
    'apache spark', 'kafka', 'airflow',
}

# Skills that suggest wrong domain (CV/speech/robotics) with no NLP
WRONG_DOMAIN_SKILLS = {
    'computer vision', 'object detection', 'image segmentation', 'yolo',
    'opencv', 'image classification', 'face recognition', 'speech recognition',
    'asr', 'tts', 'speech synthesis', 'robotics', 'ros', 'slam',
    'point cloud', 'lidar', 'autonomous driving',
}

# ─────────────────────────────────────────────────────────────────────────────
# COMPANY / INDUSTRY LISTS
# ─────────────────────────────────────────────────────────────────────────────

CONSULTING_FIRMS = {
    'tcs', 'tata consultancy', 'tata consultancy services',
    'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
    'hcl', 'hcl technologies', 'tech mahindra', 'mphasis',
    'hexaware', 'l&t infotech', 'ltimindtree', 'l&t technology',
    'persistent systems', 'cyient', 'niit technologies', 'mastech',
    'zensar', 'syntel', 'igate', 'patni', 'kpit', 'birlasoft',
    'mindtree', 'mps limited', 'sasken',
}

PRODUCT_TECH_INDUSTRIES = {
    'technology', 'software', 'saas', 'fintech', 'it services',
    'ai', 'edtech', 'healthtech', 'e-commerce', 'internet',
    'cloud computing', 'data analytics', 'cybersecurity',
    'product', 'startup', 'platform',
}

# ─────────────────────────────────────────────────────────────────────────────
# LOCATION PREFERENCES (from JD)
# ─────────────────────────────────────────────────────────────────────────────

TOP_LOCATIONS = {'pune', 'noida'}

PREFERRED_LOCATIONS = {
    'pune', 'noida', 'hyderabad', 'mumbai', 'delhi',
    'gurugram', 'gurgaon', 'bengaluru', 'bangalore', 'ncr',
    'new delhi', 'navi mumbai',
}

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIENCE RANGE (from JD)
# ─────────────────────────────────────────────────────────────────────────────

YOE_IDEAL_MIN = 6
YOE_IDEAL_MAX = 8
YOE_STATED_MIN = 5
YOE_STATED_MAX = 9

# ─────────────────────────────────────────────────────────────────────────────
# SCORING WEIGHTS — STAGE 1 (pre-filtering, no cross-encoder)
# ─────────────────────────────────────────────────────────────────────────────

STAGE1_WEIGHTS = {
    'semantic':    0.30,
    'lexical':     0.12,
    'structured':  0.58,
}

# ─────────────────────────────────────────────────────────────────────────────
# SCORING WEIGHTS — STAGE 2 (final, with cross-encoder)
# ─────────────────────────────────────────────────────────────────────────────

STAGE2_WEIGHTS = {
    'semantic':       0.25,
    'lexical':        0.10,
    'structured':     0.45,
    'cross_encoder':  0.20,
}

# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURED SCORE SUB-WEIGHTS (must sum to 1.0)
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURED_SUB_WEIGHTS = {
    'skills':       0.30,   # Trust-weighted skills match
    'career':       0.25,   # Career quality (product vs consulting, stability)
    'experience':   0.10,   # YoE range fit
    'location':     0.08,   # Geographic fit
    'notice':       0.05,   # Notice period
    'behavioral':   0.12,   # Platform behavioral signals
    'technical':    0.07,   # GitHub + assessments
    'demand':       0.03,   # Recruiter demand signals
}

# ─────────────────────────────────────────────────────────────────────────────
# SKILL TRUST MULTIPLIER WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

SKILL_TRUST_WEIGHTS = {
    'proficiency':   0.30,
    'duration':      0.30,
    'endorsements':  0.20,
    'assessment':    0.20,
}

PROFICIENCY_MAP = {
    'beginner':     0.15,
    'intermediate': 0.40,
    'advanced':     0.80,
    'expert':       1.00,
}

# ─────────────────────────────────────────────────────────────────────────────
# DISQUALIFIER PENALTIES (multiplicative)
# ─────────────────────────────────────────────────────────────────────────────

DISQUALIFIER_PENALTIES = {
    'full_consulting_career':       0.10,
    'title_description_mismatch':   0.20,
    'behavioral_zombie':            0.35,
    'interview_ghost':              0.60,
    'offer_window_shopper':         0.75,
    'job_hopper':                   0.65,
    'notice_over_90':               0.80,
    'wrong_domain_expert':          0.30,
    'unverified_identity':          0.92,
}

# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

HONEYPOT_YOE_GAP_THRESHOLD     = 5    # years
HONEYPOT_EXPERT_ZERO_DUR_MIN   = 5    # count of expert skills with 0 duration
HONEYPOT_EXPERT_NO_ENDORSE_MIN = 8    # count of expert skills with 0 endorsements
HONEYPOT_CONTRADICTION_MIN     = 3    # count of expert skills with assessment < 30

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

TOP_N_STAGE1        = 500   # Candidates passed to cross-encoder
TOP_N_FINAL         = 100   # Final output size
CROSS_ENCODER_BATCH = 32    # Batch size for cross-encoder inference

# Text truncation limits
CANDIDATE_TEXT_MAX_CHARS     = 1000
JD_TEXT_FOR_CROSS_ENCODER    = 512
CAND_TEXT_FOR_CROSS_ENCODER  = 512

# Behavioral recency decay (days)
RECENCY_DECAY_DAYS           = 180
ZOMBIE_INACTIVE_DAYS         = 150
ZOMBIE_RESPONSE_RATE_THRESH  = 0.10

# Career stability
GOOD_TENURE_MONTHS           = 30
HOPPER_TENURE_MONTHS         = 12
HOPPER_MIN_JOBS              = 5

# Response time half-life (hours)
RESPONSE_TIME_HALF_LIFE_HRS  = 48

# ─────────────────────────────────────────────────────────────────────────────
# MODEL IDS (offline-loadable, no network during ranking)
# ─────────────────────────────────────────────────────────────────────────────

BI_ENCODER_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Non-AI roles — used for title/description mismatch detection
NON_AI_ROLE_KEYWORDS = {
    'marketing', 'sales', 'hr', 'human resources', 'accountant',
    'finance', 'operations manager', 'supply chain', 'logistics',
    'customer success', 'account manager', 'business development',
    'content writer', 'graphic designer', 'seo',
}

# Reasoning rank-zone prefixes
RANK_ZONE_PREFIXES = {
    (1, 10):   "Strong match — ",
    (11, 30):  "Good fit — ",
    (31, 60):  "Moderate fit — ",
    (61, 100): "Adjacent fit — ",
}
