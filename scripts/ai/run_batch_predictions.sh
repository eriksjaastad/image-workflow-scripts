#!/usr/bin/env bash
# Batch AI Predictions Script - Phase 1A Only
# Runs crop predictions (v3) across all historical projects
# Can be stopped/resumed - tracks progress automatically

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Paths
WORKSPACE="/Users/eriksjaastad/projects/image-workflow"
ORIGINAL_DIR="/Volumes/T7Shield/Eros/original"
TMP_DIR="/tmp/ai_predictions_batch"
DB_DIR="$WORKSPACE/data/training/ai_training_decisions"
LOG_FILE="$WORKSPACE/data/ai_data/batch_predictions_log.jsonl"
PROGRESS_FILE="$WORKSPACE/data/ai_data/batch_predictions_progress.txt"

# Models (using latest versions)
RANKER_MODEL="ranker_v4.pt"
CROP_MODEL="crop_proposer_v3.pt"  # Version 3 as requested

# Activate Python environment
cd "$WORKSPACE"
source .venv311/bin/activate

# Create directories
mkdir -p "$TMP_DIR"
mkdir -p "$DB_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     AI Crop Predictions - Batch Processing (Phase 1A)     ║${NC}"
echo -e "${CYAN}║              Using Crop Proposer v3 (Latest)               ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Workspace:${NC}     $WORKSPACE"
echo -e "${BLUE}Original dir:${NC}  $ORIGINAL_DIR"
echo -e "${BLUE}Temp dir:${NC}      $TMP_DIR"
echo -e "${BLUE}Database dir:${NC}  $DB_DIR"
echo -e "${BLUE}Ranker:${NC}        $RANKER_MODEL"
echo -e "${BLUE}Crop model:${NC}    $CROP_MODEL"
echo ""

# Projects to process (ordered by size: small → large)
# Format: "project_id|zip_filename|image_count"
declare -a PROJECTS=(
    "dalia|dalia.zip|98"
    "Patricia|Average Patricia.zip|255"
    "mixed-0919|mixed-0919.zip|852"
    "agent-1003|agent-1003.zip|2053"
    "agent-1002|agent-1002.zip|2057"
    "agent-1001|agent-1001.zip|2063"
    "1013|1013.zip|2882"
    "1010|1010.zip|2912"
    "1102|1102.zip|2968"
    "1011|1011.zip|4666"
    "1012|1012.zip|5680"
    "Eleni|Eleni_raw.zip|5816"
    "Kiara_Slender|Slender Kiara.zip|5796"
    "1100|1100.zip|8904"
    "1101_Hailey|1101.zip|8904"
    "Aiko_raw|Aiko_raw.zip|1050"
    "tattersail-0918|tattersail-0918.zip|13928"
    "jmlimages-random|jmlimages-random.zip|26690"
    "mojo2|mojo2.zip|35870"
    "mojo1|mojo1.zip|38366"
)

# Get list of already completed projects
declare -A COMPLETED
if [ -f "$PROGRESS_FILE" ]; then
    while IFS= read -r project; do
        COMPLETED["$project"]=1
    done < "$PROGRESS_FILE"
fi

# Count projects
TOTAL_PROJECTS=${#PROJECTS[@]}
PROCESSED=0
SKIPPED=0
FAILED=0

START_TIME=$(date +%s)

echo -e "${GREEN}Starting batch processing of $TOTAL_PROJECTS projects...${NC}"
echo ""

# Process each project
for project_line in "${PROJECTS[@]}"; do
    IFS='|' read -r PROJECT_ID ZIP_FILE IMAGE_COUNT <<< "$project_line"
    
    # Check if already completed
    if [ -n "${COMPLETED[$PROJECT_ID]:-}" ]; then
        echo -e "${YELLOW}⏭  Skipping $PROJECT_ID (already completed)${NC}"
        ((SKIPPED++))
        continue
    fi
    
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}▶  Processing: $PROJECT_ID${NC}"
    echo -e "${BLUE}   Zip file:${NC}  $ZIP_FILE"
    echo -e "${BLUE}   Images:${NC}    $IMAGE_COUNT"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    PROJECT_START=$(date +%s)
    
    # Create temp directory for this project
    PROJECT_TMP="$TMP_DIR/$PROJECT_ID"
    mkdir -p "$PROJECT_TMP"
    
    # Extract zip to temp directory
    echo -e "${BLUE}📦 Extracting zip file...${NC}"
    if ! unzip -q "$ORIGINAL_DIR/$ZIP_FILE" -d "$PROJECT_TMP"; then
        echo -e "${RED}❌ Failed to extract $ZIP_FILE${NC}"
        ((FAILED++))
        continue
    fi
    
    # Find the actual image directory (zip might have nested structure)
    ORIGINAL_IMAGES_DIR=$(find "$PROJECT_TMP" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | head -1 | xargs dirname)
    if [ -z "$ORIGINAL_IMAGES_DIR" ]; then
        # Try the project directory itself
        ORIGINAL_IMAGES_DIR="$PROJECT_TMP"
    fi
    
    echo -e "${BLUE}📂 Images directory: $ORIGINAL_IMAGES_DIR${NC}"
    
    # Run AI predictions (Phase 1A)
    echo -e "${BLUE}🤖 Running AI predictions (ranker + crop proposer v3)...${NC}"
    OUTPUT_DB="$DB_DIR/${PROJECT_ID}.db"
    
    if python3 scripts/ai/backfill_project_phase1a_ai_predictions.py \
        --project-id "$PROJECT_ID" \
        --original-dir "$ORIGINAL_IMAGES_DIR" \
        --output-db "$OUTPUT_DB" \
        --ranker-model "$RANKER_MODEL" \
        --crop-model "$CROP_MODEL"; then
        
        PROJECT_END=$(date +%s)
        PROJECT_DURATION=$((PROJECT_END - PROJECT_START))
        
        echo -e "${GREEN}✅ Success! Database saved to: $OUTPUT_DB${NC}"
        echo -e "${GREEN}⏱  Duration: ${PROJECT_DURATION}s${NC}"
        
        # Log completion
        echo "$PROJECT_ID" >> "$PROGRESS_FILE"
        
        # Log to JSONL
        echo "{\"project_id\":\"$PROJECT_ID\",\"zip_file\":\"$ZIP_FILE\",\"image_count\":$IMAGE_COUNT,\"duration_seconds\":$PROJECT_DURATION,\"status\":\"success\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"database\":\"$OUTPUT_DB\"}" >> "$LOG_FILE"
        
        ((PROCESSED++))
    else
        echo -e "${RED}❌ Failed to process $PROJECT_ID${NC}"
        echo "{\"project_id\":\"$PROJECT_ID\",\"zip_file\":\"$ZIP_FILE\",\"image_count\":$IMAGE_COUNT,\"status\":\"failed\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> "$LOG_FILE"
        ((FAILED++))
    fi
    
    # Clean up temp directory for this project
    echo -e "${BLUE}🧹 Cleaning up temp files...${NC}"
    rm -rf "$PROJECT_TMP"
    
    echo ""
    
    # Progress summary
    REMAINING=$((TOTAL_PROJECTS - PROCESSED - SKIPPED - FAILED))
    ELAPSED=$(($(date +%s) - START_TIME))
    if [ $PROCESSED -gt 0 ]; then
        AVG_TIME=$((ELAPSED / PROCESSED))
        EST_REMAINING=$((AVG_TIME * REMAINING))
        EST_HOURS=$((EST_REMAINING / 3600))
        EST_MINS=$(((EST_REMAINING % 3600) / 60))
        echo -e "${CYAN}📊 Progress: $PROCESSED/$TOTAL_PROJECTS processed | $SKIPPED skipped | $FAILED failed | $REMAINING remaining${NC}"
        echo -e "${CYAN}⏱  Estimated time remaining: ${EST_HOURS}h ${EST_MINS}m${NC}"
    else
        echo -e "${CYAN}📊 Progress: $PROCESSED/$TOTAL_PROJECTS processed | $SKIPPED skipped | $FAILED failed${NC}"
    fi
    echo ""
done

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                    Batch Processing Complete               ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✅ Successfully processed: $PROCESSED projects${NC}"
echo -e "${YELLOW}⏭  Skipped (already done):  $SKIPPED projects${NC}"
echo -e "${RED}❌ Failed:                 $FAILED projects${NC}"
echo -e "${BLUE}⏱  Total time:            ${HOURS}h ${MINUTES}m ${SECONDS}s${NC}"
echo ""
echo -e "${BLUE}📁 Databases saved to:${NC}     $DB_DIR"
echo -e "${BLUE}📋 Log file:${NC}              $LOG_FILE"
echo -e "${BLUE}📊 Progress file:${NC}         $PROGRESS_FILE"
echo ""

# Create summary report
SUMMARY_FILE="$WORKSPACE/data/ai_data/batch_predictions_summary_$(date -u +%Y%m%dT%H%M%SZ).json"
cat > "$SUMMARY_FILE" <<EOF
{
  "batch_id": "batch_$(date -u +%Y%m%dT%H%M%SZ)",
  "started_at": "$(date -d @$START_TIME -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -r $START_TIME -u +%Y-%m-%dT%H:%M:%SZ)",
  "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "total_duration_seconds": $TOTAL_DURATION,
  "models": {
    "ranker": "$RANKER_MODEL",
    "crop_proposer": "$CROP_MODEL"
  },
  "results": {
    "total_projects": $TOTAL_PROJECTS,
    "processed": $PROCESSED,
    "skipped": $SKIPPED,
    "failed": $FAILED
  },
  "output_directory": "$DB_DIR",
  "log_file": "$LOG_FILE"
}
EOF

echo -e "${GREEN}📄 Summary report saved to: $SUMMARY_FILE${NC}"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Some projects failed. Check the log file for details.${NC}"
    exit 1
else
    echo -e "${GREEN}🎉 All projects processed successfully!${NC}"
    exit 0
fi

