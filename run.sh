#!/bin/bash

# llama-cli
# Define variables
# MODEL_PATH="/Users/afsarabenazir/LLama-8b-models/Meta-Llama-3-8B-Instruct-IQ1_M.gguf"
# MODEL_PATH="/Users/afsarabenazir/llama.cpp/llama-8b-first-block-hf/llama-8b-first-block.gguf"

# PROMPT_TEXT="explain paging in operating systems"
# # 512 tokens
# # PROMPT_TEXT="About 80% of undergraduates and 20% of graduate students live on campus. The majority of the graduate students on campus live in one of four graduate housing complexes on campus, while all on-campus undergraduates live in one of the 29 residence halls. Because of the religious affiliation of the university, all residence halls are single-sex, with 15 male dorms and 14 female dorms. The university maintains a visiting policy (known as parietal hours) for those students who live in dormitories, specifying times when members of the opposite sex are allowed to visit other students' dorm rooms; however, all residence halls have 24-hour social spaces for students regardless of gender. Many residence halls have at least one nun and/or priest as a resident. There are no traditional social fraternities or sororities at the university, but a majority of students live in the same residence hall for all four years. Some intramural sports are based on residence hall teams, where the university offers the only non-military academy program of full-contact intramural American football. At the end of the intramural season, the championship game is played on the field in Notre Dame Stadium. What percentage of undergrads live on the Notre Dame campus? About 80% of undergraduates and 20% of graduate students live on campus. The majority of the graduate students on campus live in one of four graduate housing complexes on campus, while all on-campus undergraduates live in one of the 29 residence halls. Because of the religious affiliation of the university, all residence halls are single-sex, with 15 male dorms and 14 female dorms. The university maintains a visiting policy (known as parietal hours) for those students who live in dormitories, specifying times when members of the opposite sex are allowed to visit other students' dorm rooms; however, all residence halls have 24-hour social spaces for students regardless of gender. Many residence halls have at least one nun and/or priest as a resident. There are no traditional social fraternities or sororities at the university, but a majority of students live in the same residence hall for all four years. Some intramural sports are based on residence hall teams, where the university offers the only non-military academy program of full-contact intramural American football. At the end of the intramural season, the championship game is played on the field in Notre Dame Stadium. What percentage of undergrads live on the Notre Dame?"
# # 128 tokens
# PROMPT_TEXT="About 80% of undergraduates and 20% of graduate students live on campus. The majority of the graduate students on campus live in one of four graduate housing complexes on campus, while all on-campus undergraduates live in one of the 29 residence halls. Because of the religious affiliation of the university, all residence halls are single-sex, with 15 male dorms and 14 female dorms. The university maintains a visiting policy (known as parietal hours) for those students who live in dormitories, specifying times when members of the opposite sex are allowed to visit other students' dorm rooms;" 
# # 256 tokens
# # PROMPT_TEXT="About 80% of undergraduates and 20% of graduate students live on campus. The majority of the graduate students on campus live in one of four graduate housing complexes on campus, while all on-campus undergraduates live in one of the 29 residence halls. Because of the religious affiliation of the university, all residence halls are single-sex, with 15 male dorms and 14 female dorms. The university maintains a visiting policy (known as parietal hours) for those students who live in dormitories, specifying times when members of the opposite sex are allowed to visit other students' dorm rooms; About 80% of undergraduates and 20% of graduate students live on campus. The majority of the graduate students on campus live in one of four graduate housing complexes on campus, while all on-campus undergraduates live in one of the 29 residence halls. Because of the religious affiliation of the university, all residence halls are single-sex, with 15 male dorms and 14 female dorms. The university maintains a visiting policy (known as parietal hours) for those students who live in dormitories, specifying times when members of the opposite sex are allowed to visit other students' dorm rooms;" 
# NGL=99
# C=512
# N=5
# LOG_PATH="/Users/afsarabenazir/llama.cpp/dispatch.log"

# # # Run the command
# ./build/bin/llama-cli \
#     -m "$MODEL_PATH" \
#     -ngl "$NGL" \
#     --prompt "$PROMPT_TEXT" \
#     -no-cnv \
#     -c "$C" \
#     -n "$N" \
#     --verbose \
    # 2>&1 | tee "$LOG_PATH"

# # llama-bench
# NGL=99
# ./build/bin/llama-bench \
# 	-m /Users/afsarabenazir/Llama-3.3-70B-Instruct/Llama-3.3-70B-Instruct-Q4_K_M.gguf \
# -ngl "$NGL" \
# -r 1 \
# -p 0 \
# -n 1024,2048,4096 

NGL=30
LOG_PATH=/Users/afsarabenazir/llama.cpp/deepseek-bench-test-236b-$(date +%Y_%m_%d).log

# ./build/bin/llama-bench \
#   $(find /Users/afsarabenazir/LLama-8b-models -name "*.gguf" -print0 | xargs -0 -I{} echo -m "{}") \
#   -ngl "$NGL" \
#   -p 128,512,1024,2048 \
#   -n 128,256,512,1024,2048,4096 \
#   2>&1 | tee "$LOG_PATH"

# LOG_PATH=/Users/afsarabenazir/llama.cpp/llama-bench-70b-$(date +%Y_%m_%d).log
# NGL=99

# for file in $(find /Volumes/SSD/DeepSeek-Coder-V2-Instruct-236B -name "*.gguf"); do
#   echo "Running benchmark for: $file"
#   ./build/bin/llama-bench \
#     -m "$file" \
#     -ngl "$NGL" \
#     -p 128,512,1024,2048 \
#     -n 128,256,512,1024,2048,4096 \
#     2>&1 | tee -a "$LOG_PATH"
# done

for file in $(find /Volumes/SSD2/tmp -name "*.gguf"); do
  echo "Running benchmark for: $file"
  ./build/bin/llama-bench \
    -m "$file" \
    -ngl "$NGL" \
    -p 128,512,1024,2048 \
    -n 128,256,512,1024,2048,4096 \
    2>&1 | tee -a "$LOG_PATH"
done