╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║                        DATA COLLECTION PLAN - Week 1                            ║
║                                                                                ║
║                Cannabis Product Image Dataset (30K target)                      ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝


═════════════════════════════════════════════════════════════════════════════════════════════
EXECUTIVE SUMMARY
═════════════════════════════════════════════════════════════════════════════════════════════

STATUS: No dataset currently present
TARGET: 30,000 labeled cannabis product images
TIMELINE: 12 weeks
BUDGET: $50K - $100K
RISK: HIGH (data collection is critical path)


═════════════════════════════════════════════════════════════════════════════════════════════
PHASE 1: BOOTSTRAP (Week 2-3) - 5,000 images
═════════════════════════════════════════════════════════════════════════════════════════════

Objective: Establish baseline dataset and validate collection process

TARGETS:
├── Classes (balanced):
│   ├── Plant: 1,000 images (20%)
│   ├── Flower: 1,500 images (30%)
│   ├── Trim: 1,000 images (20%)
│   ├── Hash: 500 images (10%)
│   └── Extract: 1,000 images (20%)
│
├── Grades (80/20 distribution):
│   ├── A+ & A: 2,500 images (50%) - "Money grades"
│   ├── B: 1,500 images (30%)
│   ├── C: 800 images (16%)
│   └── F: 200 images (4%) - Defects for robustness
│
└── Devices:
    ├── iPhone (primary): 2,500 images (50%)
    ├── Android: 2,000 images (40%)
    ├── Webcam: 400 images (8%)
    └── Professional: 100 images (2%)


COLLECTION METHODS:

1. PHONE CROWDSOURCING (70% of Phase 1)
   ├── Recruit 20-30 power users
   ├── Pay $0.50-$2.00 per photo
   ├── Provide naming convention template
   ├── Weekly verification + quality checks
   ├── Platform: TBD (Discord bot / mobile app)
   └── Expected rate: 50-100 images/user/week

2. DISPENSARY PARTNERSHIPS (20% of Phase 1)
   ├── Contact 5-10 Colorado dispensaries
   ├── Offer: Free analytics dashboard access
   ├── Get: High-quality product photos
   ├── Advantage: Professional lighting + variety
   ├── Timeline: 2-3 week negotiation
   └── Expected rate: 100-200 images/dispensary/week

3. ARCHIVE/LIBRARY SOURCES (10% of Phase 1)
   ├── Search: Reddit communities (r/cannabis, r/growmies)
   ├── Search: Instagram hashtags (#cannabisreview, #buds)
   ├── Negotiate: Existing image datasets
   ├── Method: Manual curation + permission requests
   └── Expected rate: 50-100 images/week


QUALITY CONTROL:

✓ First 1,000 images: 100% manual review
  - Validate naming convention
  - Check image quality/clarity
  - Verify class/grade accuracy
  - Remove duplicates/defects

✓ Images 1,000-5,000: 25% random sampling
  - Spot-check accuracy
  - Identify drift in standards
  - Retrain contributors if needed

✓ Metadata validation:
  - All images properly named
  - No corrupted files
  - Consistent lighting where possible
  - All angles covered


TIMELINE - Phase 1:

Week 1 (Jan 20-26):
  ├── Mon: Create collection templates & guidelines
  ├── Mon: Contact dispensary partners
  ├── Tue-Wed: Recruit 30 power users
  ├── Wed: Create phone upload mechanism
  ├── Thu-Fri: Launch small pilot (100 images)
  └── Fri: Validate quality & iterate

Week 2 (Jan 27 - Feb 2):
  ├── Mon: Scale to full 30 users
  ├── Tue-Thu: Daily quality checks
  ├── Thu: Integrate dispensary partner images (500+)
  ├── Fri: Data verification + cleaning
  └── Fri: Checkpoint (2,500 images target)

Week 3 (Feb 3-9):
  ├── Mon-Wed: Continued collection
  ├── Wed: Second quality review cycle
  ├── Thu-Fri: Final cleanup & validation
  ├── Fri: Hit 5,000 image target
  └── Fri: Phase 1 complete, model training begins


═════════════════════════════════════════════════════════════════════════════════════════════
PHASE 2: EXPANSION (Week 4-6) - 15,000 images (5K→20K total)
═════════════════════════════════════════════════════════════════════════════════════════════

Objective: Deep coverage, edge cases, diverse lighting/angles

TARGETS:
├── New total: 20,000 images
├── Per class: 4,000 images each (balanced)
├── Per grade: Full spectrum (A+ to F)
├── Devices: 50% iPhone, 30% Android, 15% Webcam, 5% Professional
└── Lighting: Indoor/outdoor/LED/HPS mix


NEW SOURCES:

1. SCALE PHONE CROWDSOURCING (60% of Phase 2)
   ├── Expand to 50 power users
   ├── Increase rate: $1.00-$3.00 per photo
   ├── Add "specialty" shots: macro, angles, lighting variations
   └── Expected: 150-200 images/user/week

2. GROW OPERATION PARTNERSHIPS (25% of Phase 2)
   ├── Contact: 10-15 commercial grow operations
   ├── Partner benefits: Featured in public results/leaderboard
   ├── Get: 100+ images per operation per week
   ├── Lighting variety: HPS, LED, greenhouse, outdoor
   └── Expected: 500+ images/week

3. PROFESSIONAL PHOTOGRAPHY (15% of Phase 2)
   ├── Hire: 2-3 photographers for 4 weeks
   ├── Budget: $5K-$8K total
   ├── Tasks: Studio setup, controlled lighting, angles
   ├── Output: 50-100 high-quality images per week
   └── Quality: Reference images for calibration


EDGE CASES TO COVER:

✓ Lighting variations:
  - Natural sunlight (direct/indirect)
  - LED grow lights (various spectra)
  - HPS lights
  - Sodium vapor
  - Mixed/poor lighting
  
✓ Angle variations:
  - Top-down (macro & wide)
  - Side angles (left/right/45°)
  - Macro close-ups
  - Full plant/product
  - Hand-held scale reference
  
✓ Defects (for F grade):
  - Mold/mildew
  - Spider mites
  - Bud rot
  - Frost damage
  - Discoloration
  - Foreign matter


QUALITY TARGETS:

- Phase 1 accuracy: 92%+ correct class/grade
- Phase 2 accuracy: 95%+ correct class/grade
- Image quality: All images > 1000x1000px
- File formats: JPG (preferred), PNG (lossless backups)
- Metadata: Complete tagging (class, grade, device, date, conditions)


═════════════════════════════════════════════════════════════════════════════════════════════
PHASE 3: PRODUCTION (Week 7-12) - 10,000 images (20K→30K total)
═════════════════════════════════════════════════════════════════════════════════════════════

Objective: Production-grade dataset with full coverage

TARGETS:
├── Final total: 30,000 images
├── Diverse sources: 10+ different collection channels
├── Accuracy: 98%+ class/grade accuracy
└── Completeness: All classes/grades/devices well-represented


FINAL SOURCES:

1. SUSTAINED CROWDSOURCING (40%)
   ├── Maintain: 50 active power users
   ├── Community building: Leaderboards, rewards
   ├── Coverage: Global sourcing (expand beyond US)
   └── Rate: 100+ images/week baseline

2. COMMERCIAL PARTNERSHIPS (40%)
   ├── Establish: 15-20 grow/dispensary partners
   ├── Continuous feed: Automated image uploads
   ├── Standardization: Lighting/angle templates
   └── Coverage: Multi-state, multi-strain representation

3. PROFESSIONAL SERVICES (20%)
   ├── Maintain: On-demand photography team
   ├── Specialize: Hard-to-get angles/defects/edge cases
   ├── Quality assurance: Re-shoot if needed
   └── Budget: $3K-$5K/week


INFRASTRUCTURE SETUP:

✓ Image upload portal:
  - Web app: Drag-drop upload
  - Mobile app: iPhone/Android native
  - Batch processing: CSV-based naming
  - Validation: Client-side + server-side

✓ Quality pipeline:
  - Auto-check: File format, resolution, naming
  - Manual review: Spot-checks by team
  - Dispute resolution: User appeals process
  - Analytics: Accuracy by contributor

✓ Storage & versioning:
  - Backup: 3 copies (local, cloud, cold storage)
  - Versioning: Dataset v0.1, v0.2, v1.0, v1.1
  - Retention: Full history for reproducibility
  - Access: Team-only until release


═════════════════════════════════════════════════════════════════════════════════════════════
BUDGET BREAKDOWN
═════════════════════════════════════════════════════════════════════════════════════════════

Phase 1 (5K images):
├── Contributor payments: $5K (5K images × $1 avg)
├── Photography/setup: $2K
├── Platform/tools: $1K
└── Total: $8K

Phase 2 (10K images):
├── Contributor payments: $12K (10K images × $1.20 avg)
├── Professional photography: $8K
├── Partnership incentives: $3K
├── Storage/infrastructure: $2K
└── Total: $25K

Phase 3 (10K images):
├── Contributor payments: $10K (10K images × $1.00 sustained)
├── Professional services: $5K
├── Partnership support: $2K
├── QA/validation: $3K
├── Storage/backup: $2K
└── Total: $22K

TOTAL PROJECT: $55K (+ 15% contingency = $63K)


═════════════════════════════════════════════════════════════════════════════════════════════
RISK MITIGATION
═════════════════════════════════════════════════════════════════════════════════════════════

RISK: Slow collection pace
├── Mitigation: Increase contributor rates immediately
├── Mitigation: Activate emergency partnerships
├── Threshold: 20% behind schedule → escalate
└── Buffer: Phase padding built in

RISK: Poor quality images
├── Mitigation: Tighter QA in early phases
├── Mitigation: Real-time contributor feedback
├── Threshold: <90% QA pass rate → stop, retrain
└── Buffer: Reject/re-shoot up to 20% if needed

RISK: Legal/regulatory issues
├── Mitigation: Consult legal on image rights
├── Mitigation: Get explicit consent from all sources
├── Threshold: Any legal complaint → review full batch
└── Buffer: Budget legal review: $3K

RISK: Data leakage
├── Mitigation: Encrypt all data in transit
├── Mitigation: Limit access to core team
├── Threshold: Any unauthorized access → full audit
└── Buffer: Security audit quarterly


═════════════════════════════════════════════════════════════════════════════════════════════
SUCCESS METRICS
═════════════════════════════════════════════════════════════════════════════════════════════

Phase 1 Success:
✓ 5,000 images collected
✓ >92% QA pass rate
✓ Model reaches 70%+ accuracy
✓ All 5 classes represented
✓ Proof of concept validated

Phase 2 Success:
✓ 20,000 total images
✓ >95% QA pass rate
✓ Model reaches 85%+ accuracy
✓ 50+ unique contributors
✓ 10+ commercial partnerships

Phase 3 Success:
✓ 30,000 images (target hit)
✓ >98% QA pass rate
✓ Model reaches 91%+ accuracy (goal)
✓ Sustainable collection pipeline
✓ Production-ready dataset


═════════════════════════════════════════════════════════════════════════════════════════════
NEXT ACTIONS (THIS WEEK)
═════════════════════════════════════════════════════════════════════════════════════════════

TODAY (Jan 20):
□ Finalize collection strategy & budget
□ Create contributor guidelines document
□ Design naming convention template

TOMORROW (Jan 21):
□ Create upload portal skeleton
□ Draft partnership agreements
□ Identify first 5 target dispensaries

WED (Jan 22):
□ Launch dispensary outreach
□ Create mobile app upload mockup
□ Recruit first 10-15 power users

THU (Jan 23):
□ Complete upload infrastructure
□ Finalize QA checklist
□ Start small pilot (50 images)

FRI (Jan 24):
□ Scale pilot to 300 images
□ Validate quality
□ Plan Week 2 scaling


═════════════════════════════════════════════════════════════════════════════════════════════
HANDOFF CHECKLIST
═════════════════════════════════════════════════════════════════════════════════════════════

Week 1 Deliverables:
✓ Collection strategy finalized
✓ Budget approved
✓ Upload infrastructure ready
✓ First 50-100 images collected
✓ QA process established
✓ Contributor compensation set

Hand-off to Week 2:
□ Data Lead: Scale collection to 1,000 images
□ DevOps: Set up automated validation pipeline
□ Product: Monitor quality metrics
□ Finance: Process contributor payments


═════════════════════════════════════════════════════════════════════════════════════════════
DOCUMENT INFO
═════════════════════════════════════════════════════════════════════════════════════════════

Created: 2026-01-20
Updated: By Data Lead
Status: APPROVED
Next Review: 2026-01-27 (Week 1 review)

Related Documents:
- DATASET_AUDIT.json (current inventory)
- EXECUTION_CHECKLIST.md (daily tasks)
- 90_DAY_EXECUTION_PLAN.md (master schedule)
