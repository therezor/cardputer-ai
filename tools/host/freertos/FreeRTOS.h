// Host-build stub: binary semaphores + pinned tasks via std::thread, so the
// dual-"core" matmul worker path runs the same code as on device.
#pragma once
#include <condition_variable>
#include <mutex>
#include <thread>

#define portMAX_DELAY 0
typedef int BaseType_t;
typedef void* TaskHandle_t;
inline BaseType_t xPortGetCoreID() { return 0; }

struct HostBinarySem {
  std::mutex m;
  std::condition_variable cv;
  bool flag = false;
  void give() { std::lock_guard<std::mutex> l(m); flag = true; cv.notify_one(); }
  void take() {
    std::unique_lock<std::mutex> l(m);
    cv.wait(l, [&]{ return flag; });
    flag = false;
  }
};
typedef HostBinarySem* SemaphoreHandle_t;
inline SemaphoreHandle_t xSemaphoreCreateBinary() { return new HostBinarySem(); }
inline void xSemaphoreGive(SemaphoreHandle_t s) { s->give(); }
inline void xSemaphoreTake(SemaphoreHandle_t s, int) { s->take(); }

typedef void (*TaskFunction_t)(void*);
inline BaseType_t xTaskCreatePinnedToCore(TaskFunction_t fn, const char*, int,
                                          void* arg, int, TaskHandle_t* handle,
                                          BaseType_t) {
  std::thread t(fn, arg);
  *handle = (TaskHandle_t)1;
  t.detach();
  return 1;
}
