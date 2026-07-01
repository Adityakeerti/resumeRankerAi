import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas

def draw_header(c, title_text, slide_num):
    # Background
    c.setFillColor(colors.HexColor("#0A192F"))
    c.rect(0, 0, 792, 612, fill=True, stroke=False)
    
    # Top banner background
    c.setFillColor(colors.HexColor("#172A45"))
    c.rect(0, 520, 792, 92, fill=True, stroke=False)
    
    # Bottom footer background
    c.setFillColor(colors.HexColor("#0F1E36"))
    c.rect(0, 0, 792, 40, fill=True, stroke=False)
    
    # Teal accent line under header
    c.setStrokeColor(colors.HexColor("#64FFDA"))
    c.setLineWidth(2)
    c.line(0, 520, 792, 520)
    
    # Header text
    c.setFillColor(colors.HexColor("#E6F1FF"))
    c.setFont("Helvetica-Bold", 24)
    c.drawString(40, 555, title_text)
    
    # Footer text
    c.setFillColor(colors.HexColor("#8892B0"))
    c.setFont("Helvetica", 10)
    c.drawString(40, 15, "Redrob AI Hackathon — India Runs Data & AI Challenge")
    c.drawRightString(752, 15, f"team_404_Founders | Page {slide_num}")

def draw_title_slide(c):
    # Background
    c.setFillColor(colors.HexColor("#0A192F"))
    c.rect(0, 0, 792, 612, fill=True, stroke=False)
    
    # Visual accent blocks
    c.setFillColor(colors.HexColor("#172A45"))
    c.rect(0, 180, 792, 260, fill=True, stroke=False)
    
    c.setStrokeColor(colors.HexColor("#64FFDA"))
    c.setLineWidth(4)
    c.line(0, 440, 792, 440)
    c.line(0, 180, 792, 180)
    
    # Title
    c.setFillColor(colors.HexColor("#64FFDA"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(80, 390, "SYSTEM ARCHITECTURE DECK")
    
    c.setFillColor(colors.HexColor("#E6F1FF"))
    c.setFont("Helvetica-Bold", 32)
    c.drawString(80, 335, "Intelligent Candidate Discovery & Ranking")
    
    # Subtitle
    c.setFillColor(colors.HexColor("#8892B0"))
    c.setFont("Helvetica", 16)
    c.drawString(80, 290, "A Hybrid Multi-Stage Retrieval, NLI & CDF Pipeline")
    
    # Team
    c.setFillColor(colors.HexColor("#64FFDA"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(80, 220, "Presented by: team_404_Founders")
    
    # Date/Footer
    c.setFillColor(colors.HexColor("#8892B0"))
    c.setFont("Helvetica", 11)
    c.drawString(80, 50, "Redrob AI Recruiter Platform Hackathon | July 2026")
    c.showPage()

def draw_slide2(c):
    draw_header(c, "1. Core Philosophy & Design Trade-Offs", 2)
    
    # Col 1: What We Built
    c.setFillColor(colors.HexColor("#CCD6F6"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, 470, "What We Built")
    
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.HexColor("#8892B0"))
    
    text_left = [
        "• Hybrid Multi-Stage Ranker combining semantic embeddings, exact lexical matches,",
        "  structured profile signals, and deep cross-attention reranking.",
        "• Statistical CDF engine that normalizes raw candidate engagement numbers into",
        "  unbiased relative percentiles, translating numerical signals into semantic text.",
        "• Zero-Shot NLI Logical Gate to evaluate candidate location/mode preferences,",
        "  preventing semantic embedding overlap from passing mismatched candidates."
    ]
    y = 440
    for line in text_left:
        c.drawString(40, y, line)
        y -= 25
        
    # Col 2: Why We Built It This Way (Design Trade-offs)
    c.setFillColor(colors.HexColor("#CCD6F6"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(430, 470, "Why We Built It This Way (Why / Why Not)")
    
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.HexColor("#8892B0"))
    
    text_right = [
        "• Why Hybrid: Embeddings catch general capability; BM25 catches specific tool requirements.",
        "• Why Multi-Stage: Running a Cross-Encoder over 100K profiles on CPU takes ~80 mins.",
        "  Our L1 Shortlist (top-2000) reduces the pool in milliseconds, enabling CPU",
        "  cross-encoder reranking (top-500) to execute in less than 20 seconds.",
        "• Why Not Local LLMs: quantized 8B LLM inference takes ~40s per candidate on CPU.",
        "  This is completely infeasible under the 5-minute hackathon execution budget.",
        "• Why NLI over Hardcoding: Hardcoded filters are brittle and easily bypassed by",
        "  formatting. NLI handles complex semantic contradiction checking out-of-the-box."
    ]
    y = 440
    for line in text_right:
        c.drawString(430, y, line)
        y -= 25
        
    c.showPage()

def draw_slide3(c):
    draw_header(c, "2. System Architecture & Flow", 3)
    
    # Left Box: Offline Precomputation
    c.setFillColor(colors.HexColor("#172A45"))
    c.rect(40, 100, 330, 390, fill=True, stroke=False)
    
    c.setFillColor(colors.HexColor("#64FFDA"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(60, 460, "OFFLINE PRECOMPUTATION")
    
    c.setFillColor(colors.HexColor("#CCD6F6"))
    c.setFont("Helvetica", 11)
    lines_left = [
        "Runs once prior to ranking. Generates cache files.",
        "",
        "1. Parse candidates.jsonl in multi-processed chunks",
        "2. Compute CDF distributions for numerical signals",
        "3. Build rich text representations incorporating",
        "   relative percentiles and core career summaries",
        "4. Generate 384-d dense embeddings using",
        "   sentence-transformers/all-MiniLM-L6-v2",
        "5. Compile BM25 Lexical Index (bm25_index.pkl)",
        "6. Pre-screen profiles for temporal contradictions",
        "   and gaming indicators (honeypot_flags.npy)",
        "7. Compute base disqualifiers (disq_flags.npy)"
    ]
    y = 430
    for line in lines_left:
        c.drawString(60, y, line)
        y -= 22

    # Right Box: Online Ranking
    c.setFillColor(colors.HexColor("#172A45"))
    c.rect(410, 100, 340, 390, fill=True, stroke=False)
    
    c.setFillColor(colors.HexColor("#64FFDA"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(430, 460, "ONLINE RANKING ENGINE (rank.py)")
    
    c.setFillColor(colors.HexColor("#CCD6F6"))
    c.setFont("Helvetica", 11)
    lines_right = [
        "Executes on CPU inside Docker sandbox under 5 mins.",
        "",
        "1. Load JD text from docx, clean, and embed",
        "2. Retrieve candidate BM25 and structured scores",
        "3. L1 Shortlist: Matrix multiplication of JD and",
        "   100K candidate embeddings (top-2000 shortlisted)",
        "4. Stream top-2000 profiles & evaluate work mode/",
        "   relocation constraints via NLI logical model",
        "5. Stage 1 Fusion (0.30 sem + 0.12 lex + 0.58 str)",
        "6. Select top-500 and run token-level Cross-Encoder",
        "   re-ranking using ms-marco-MiniLM-L-6-v2",
        "7. Stage 2 Fusion (20% CE weight) & round scores",
        "8. Stable lexicographical tie-break ID sorting"
    ]
    y = 430
    for line in lines_right:
        c.drawString(430, y, line)
        y -= 22

    c.showPage()

def draw_slide4(c):
    draw_header(c, "3. Innovation: NLI Logic Gate & CDF Statistics", 4)
    
    # Left: Zero-Shot NLI Gate
    c.setFillColor(colors.HexColor("#CCD6F6"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, 470, "Zero-Shot NLI Logical Constraint Gate")
    
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.HexColor("#8892B0"))
    text_left = [
        "• Semantic similarity models group text by topic, confusing compatibility with relevance.",
        "  For example, onsite requirements match highly with remote profiles due to keyword overlap.",
        "• Solution: Deploy local cross-encoder/nli-deberta-v3-small on the L1 shortlist (top-2000).",
        "• Method: Formulate logic gates as premise-hypothesis contradiction verification:",
        "  - Premise: 'The candidate is willing to relocate to another city.'",
        "  - Hypothesis: 'Candidate is not willing to relocate.'",
        "• Outcome: High contradiction probability (P > 0.80) triggers a multiplicative penalty",
        "  (factor of 0.10) to automatically drop incompatible candidates without hardcoding rules."
    ]
    y = 440
    for line in text_left:
        c.drawString(40, y, line)
        y -= 25

    # Right: CDF Statistical Translation
    c.setFillColor(colors.HexColor("#CCD6F6"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, 240, "Statistical CDF-to-Text Normalization")
    
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.HexColor("#8892B0"))
    text_right = [
        "• Standard systems apply arbitrary thresholds to numbers (e.g. response rate < 50% = bad).",
        "• Solution: Build the Cumulative Distribution Function (CDF) across the entire pool of 100K.",
        "• Process: Map candidate values to relative percentiles: Percentile(x) = (Count < x) / 100,000.",
        "• Textualization: Convert percentiles to semantic categories: extremely low (0-10%), below average,",
        "  average (25-75%), high, and exceptional. Assemble into a 'Behavioral Summary Query'",
        "  (e.g., 'Candidate shows high engagement with recruiters. Candidate average response time is exceptional.')",
        "  and append to candidate resume text before embedding. AI naturally reads and weights these signals."
    ]
    y = 210
    for line in text_right:
        c.drawString(40, y, line)
        y -= 22

    c.showPage()

def draw_slide5(c):
    draw_header(c, "4. Anti-Gaming, Robustness & Trap Protection", 5)
    
    # Left: Trust-Weighted Skills
    c.setFillColor(colors.HexColor("#CCD6F6"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, 470, "Skill Trust Score (Anti-Keyword Stuffing)")
    
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.HexColor("#8892B0"))
    text_left = [
        "• The Trap: Candidates stuff their profile with expert keywords they've never used.",
        "• The Defense: Implement a multi-dimensional trust multiplier for each declared skill:",
        "  Trust = Proficiency Map x Duration Months x Endorsement Count x Assessment Score",
        "• A candidate with verified assessments and endorsements ranks significantly higher",
        "  than a keyword stuffer who lists 20 expert skills with 0 months of experience.",
        "• Text building also weights career history descriptions above raw skills lists."
    ]
    y = 440
    for line in text_left:
        c.drawString(40, y, line)
        y -= 25

    # Right: Honeypots & Disqualifiers
    c.setFillColor(colors.HexColor("#CCD6F6"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, 260, "Honeypot Screening & Multiplicative Disqualifiers")
    
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.HexColor("#8892B0"))
    text_right = [
        "• Honeypot Screening: 7-point validation check screens out seeded impossible profiles",
        "  (e.g. concurrent overlapping full-time roles, experience length exceeding age, expert claims",
        "  with near-zero assessment). 3+ flags triggers a 0.0 multiplier (hard mask), dropping them instantly.",
        "• Multiplicative Disqualifiers: Hard constraints are applied as scaling factors (not additive):",
        "  - Consulting-Firm-Only Career: candidates with only TCS/Infosys etc. experience get a 0.10 multiplier.",
        "  - Title-Description Mismatch: marketing/sales titles stuffed with AI keywords get a 0.20 multiplier.",
        "  - Behavioral Zombies: inactive candidates (last active > 150 days) receive a 0.35 multiplier.",
        "  - Notice Period: notice periods exceeding 90 days receive a 0.80 multiplier."
    ]
    y = 230
    for line in text_right:
        c.drawString(40, y, line)
        y -= 22

    c.showPage()

def draw_slide6(c):
    draw_header(c, "5. Performance Optimizations & Compliance", 6)
    
    # Left: Compute & Latency Trade-offs
    c.setFillColor(colors.HexColor("#CCD6F6"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, 470, "Compute Constraints & Code Reproduction")
    
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.HexColor("#8892B0"))
    text_left = [
        "• CPU-Only, No Network, 16GB RAM: Fully complied. No external API calls are made.",
        "• Model Choice: Switched bi-encoder from nomic-embed-text-v1.5 (12 hours to embed 100K on CPU)",
        "  to sentence-transformers/all-MiniLM-L6-v2 (50 minutes to embed 100K on CPU).",
        "• Execution: Matrix dot product (100K candidates x 384 dimensions) completes in <1s.",
        "• Memory Efficiency: Dense embeddings loaded using NumPy memory-mapping (mmap_mode='r'),",
        "  ensuring a minimal memory footprint (< 4GB RAM) and sub-second CLI start-up time."
    ]
    y = 440
    for line in text_left:
        c.drawString(40, y, line)
        y -= 25

    # Right: Results & Verification
    c.setFillColor(colors.HexColor("#CCD6F6"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, 260, "Submission Verification & Results")
    
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.HexColor("#8892B0"))
    text_right = [
        "• Pipeline Execution Speed: rank.py completes in 120.1 seconds on standard CPU.",
        "• Output Format: Validated using validate_submission.py, ensuring 100 unique candidates,",
        "  monotonically non-increasing scores, and ascending lexicographical tie-breakers.",
        "• Deployments: Pushed code to GitHub (Adityakeerti/resumeRankerAi) and live Docker-based demo",
        "  to Hugging Face Spaces (adityacodes404/redrob-ranker) with Streamlit UI.",
        "• Codebase structure: Organized cleanly into modular files for structured scoring,",
        "  honeypot detection, BM25 indexing, semantic bi-encoder/cross-encoder, and reasoning."
    ]
    y = 230
    for line in text_right:
        c.drawString(40, y, line)
        y -= 22

    c.showPage()

def main():
    pdf_path = "team_404_Founders_presentation.pdf"
    print(f"[PDF] Generating {pdf_path} ...")
    c = canvas.Canvas(pdf_path, pagesize=landscape(letter))
    
    draw_title_slide(c)
    draw_slide2(c)
    draw_slide3(c)
    draw_slide4(c)
    draw_slide5(c)
    draw_slide6(c)
    
    c.save()
    print("[PDF] Generation complete!")

if __name__ == '__main__':
    main()
