# #!/bin/bash

#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# ← adjust these if you ever move things
# MODEL_DIR="/Users/afsarabenazir/Llama-3.3-70B-Instruct"
# OUTPUT_DIR="/Users/afsarabenazir/xctrace_data"
# LLAMA_CLI="/Users/afsarabenazir/llama.cpp/build/bin/llama-cli"

MODEL_DIR="/Volumes/SSD/DeepSeek-Coder-V2-Instruct-236B"
OUTPUT_DIR="/Users/afsarabenazir/xctrace_data/236B"
LLAMA_CLI="/Users/afsarabenazir/llama.cpp/build/bin/llama-cli"

# # ← your constant parameters
# PROMPT_TEXT="About 80% of undergraduates and 20% of graduate students live on campus. The majority of the graduate students on campus live in one of four graduate housing complexes on campus, while all on-campus undergraduates live in one of the 29 residence halls. Because of the religious affiliation of the university, all residence halls are single-sex, with 15 male dorms and 14 female dorms. The university maintains a visiting policy (known as parietal hours) for those students who live in dormitories, specifying times when members of the opposite sex are allowed to visit other students' dorm rooms; however, all residence halls have 24-hour social spaces for students regardless of gender. Many residence halls have at least one nun and/or priest as a resident. There are no traditional social fraternities or sororities at the university, but a majority of students live in the same residence hall for all four years. Some intramural sports are based on residence hall teams, where the university offers the only non-military academy program of full-contact intramural American football. At the end of the intramural season, the championship game is played on the field in Notre Dame Stadium. What percentage of undergrads live on the Notre Dame campus?"
NGL=99
# C=512
N=10

# 128 tokens
# PROMPT_TEXT="About 80% of undergraduates and 20% of graduate students live on campus. The majority of the graduate students on campus live in one of four graduate housing complexes on campus, while all on-campus undergraduates live in one of the 29 residence halls. Because of the religious affiliation of the university, all residence halls are single-sex, with 15 male dorms and 14 female dorms. The university maintains a visiting policy (known as parietal hours) for those students who live in dormitories, specifying times when members of the opposite sex are allowed to visit other students' dorm rooms;" 
# 256 tokens
# C=128
C=6
# PROMPT_TEXT="About 80% of undergraduates and 20% of graduate students live on campus. The majority of the graduate students on campus live in one of four graduate housing complexes on campus, while all on-campus undergraduates live in one of the 29 residence halls. Because of the religious affiliation of the university, all residence halls are single-sex, with 15 male dorms and 14 female dorms. The university maintains a visiting policy (known as parietal hours) for those students who live in dormitories, specifying times when members of the opposite sex are allowed to visit other students' dorm rooms; About 80% of undergraduates and 20% of graduate students live on campus. The majority of the graduate students on campus live in one of four graduate housing complexes on campus, while all on-campus undergraduates live in one of the 29 residence halls. Because of the religious affiliation of the university, all residence halls are single-sex, with 15 male dorms and 14 female dorms. The university maintains a visiting policy (known as parietal hours) for those students who live in dormitories, specifying times when members of the opposite sex are allowed to visit other students' dorm rooms;" 
PROMPT_TEXT="explain paging in operating systems"
#128 tokens
# PROMPT_TEXT="About 80% of undergraduates and 20% of graduate students live on campus. The majority of the graduate students on campus live in one of four graduate housing complexes on campus, while all on-campus undergraduates live in one of the 29 residence halls. Because of the religious affiliation of the university, all residence halls are single-sex, with 15 male dorms and 14 female dorms. The university maintains a visiting policy (known as parietal hours) for those students who live in dormitories, specifying times when members of the opposite sex are allowed to visit other students' dorm rooms;" 

for MODEL_PATH in "$MODEL_DIR"/*.gguf; do
  MODEL_NAME=$(basename "$MODEL_PATH" .gguf)
  TIMESTAMP=$(date +'%Y_%m_%d_%H%M%S')
  OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_NAME}_${TIMESTAMP}_N${N}_C${C}.trace"

  echo ">>> Tracing model: $MODEL_NAME"
  xctrace record \
    --template gpu_counter4 \
    --output "$OUTPUT_FILE" \
    --launch -- "$LLAMA_CLI" \
      -m "$MODEL_PATH" \
      --prompt "$PROMPT_TEXT" \
      -no-cnv \
      -c "$C" \
      -n "$N" \
      -ngl "$NGL"
done

# # Define variables
# MODEL_NAME="Llama-3.3-70B-Instruct-Q4_0"
# MODEL_PATH="/Users/afsarabenazir/Llama-3.3-70B-Instruct/$MODEL_NAME.gguf"

# PROMPT_TEXT="About 80% of undergraduates and 20% of graduate students live on campus. The majority of the graduate students on campus live in one of four graduate housing complexes on campus, while all on-campus undergraduates live in one of the 29 residence halls. Because of the religious affiliation of the university, all residence halls are single-sex, with 15 male dorms and 14 female dorms. The university maintains a visiting policy (known as parietal hours) for those students who live in dormitories, specifying times when members of the opposite sex are allowed to visit other students' dorm rooms; however, all residence halls have 24-hour social spaces for students regardless of gender. Many residence halls have at least one nun and/or priest as a resident. There are no traditional social fraternities or sororities at the university, but a majority of students live in the same residence hall for all four years. Some intramural sports are based on residence hall teams, where the university offers the only non-military academy program of full-contact intramural American football. At the end of the intramural season, the championship game is played on the field in Notre Dame Stadium. What percentage of undergrads live on the Notre Dame campus?"
# NGL=99
# C=512
# N=1024
# # LOG_PATH="/Users/afsarabenazir/llama.cpp/dispatch.log"

# # Run the command
# # ./build/bin/llama-cli \
# #     -m "$MODEL_PATH" \
# #     -ngl "$NGL" \
# #     --prompt "$PROMPT_TEXT" \
# #     -no-cnv \
# #     -c "$C" \
# #     -n "$N" \
# #     2>&1 | tee "$LOG_PATH"

# xctrace record \
#     --template "shader_timeline" \
#     --output /Users/afsarabenazir/xctrace_data/Llama-3.3-70B-Instruct-Q4_0_$(date +%Y_%m_%d_%H%M%S)_N"$N"_C"$C".trace \
#     --launch -- /Users/afsarabenazir/llama.cpp/build/bin/llama-cli \
#     -m "$MODEL_PATH" \
#     --prompt "$PROMPT_TEXT" \
#     -no-cnv \
#     -c 512 \
#     -n 10 \
#     -ngl 99 \


# MODEL_NAME="DeepSeek-Coder-V2-Instruct.Q8_0"
# MODEL_PATH="/Volumes/SSD2/DeepSeek-Coder-V2-Instruct-236B/DeepSeek-Coder-V2-Instruct.Q8_0.gguf"

# xctrace record \
#     --template "gpu_counter4" \
#     --output /Users/afsarabenazir/xctrace_data/236B/${MODEL_NAME}_$(date +%Y_%m_%d_%H%M%S)_N"$N"_C"$C".trace \
#     --launch -- /Users/afsarabenazir/llama.cpp/build/bin/llama-cli \
#     -m "$MODEL_PATH" \
#     --prompt "$PROMPT_TEXT" \
#     -no-cnv \
#     -c "$C" \
#     -n "$N" \
#     -ngl "$NGL" \

# # xctrace export --input /Users/afsarabenazir/xctrace_data/Meta-Llama-3-8B-Instruct-IQ1_M_2025_05_06_141329_N100_C512.trace \
# # --xpath '/Users/afsarabenazir/xctrace_data/Meta-Llama-3-8B-Instruct-IQ1_M_2025_05_06_141329_N100_C512.trace/run/data/table[@schema="com.apple.dt.xctrace.schema.metal-kernel-resource-allocations"]' \
# #   --output shader_summary.xml