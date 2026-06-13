// Small Arduino-isms used across the app, expressed in ESP-IDF terms.
#pragma once
#include <stdint.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

static inline uint32_t millis() { return (uint32_t)(esp_timer_get_time() / 1000); }
static inline void delay_ms(uint32_t ms) { vTaskDelay(pdMS_TO_TICKS(ms)); }
