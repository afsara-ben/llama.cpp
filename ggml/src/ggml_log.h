#ifndef GGML_LOG_H
#define GGML_LOG_H

#include "ggml.h"
#include <stdio.h>

static FILE* g_log_file = NULL;

static void ggml_file_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    
    if (g_log_file == NULL) {
        const char* log_path = getenv("GGML_LOG_FILE");
        if (log_path != NULL) {
            g_log_file = fopen(log_path, "a");
            if (g_log_file == NULL) {
                fprintf(stderr, "Failed to open log file %s\n", log_path);
                g_log_file = stderr;
            }
        } else {
            g_log_file = stderr;
        }
    }
    
    fputs(text, g_log_file);
    fflush(g_log_file);
}

static void init_file_logging(void) {
    ggml_log_set(ggml_file_log_callback, NULL);
}

#endif // GGML_LOG_H 