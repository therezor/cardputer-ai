#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string>

class ChatUI {
public:
  void begin();
  void status(const char* s);
  void statusf(const char* fmt, ...);
  void ready();
  void fatal(const char* s) __attribute__((noreturn));

  // Selector screen
  void showSelector(const char* title, const std::string* items, const std::string* subs, int n, int selected);
  void showProgress(const char* label, size_t done, size_t total);

  // Settings screen: names left-aligned, values right-aligned.
  void showSettings(const char* title, const std::string* names, const std::string* values, int n, int selected);
  void repaint();                    // full redraw of the chat screen

  // Chat
  void onChar(char c);
  void onBackspace();
  std::string takeInput();
  void appendUser(const std::string& s);
  void beginBotReply();
  void appendBot(const char* piece);
  void endBotReply(int tokens, uint32_t ms);
  void tickGenerating(int tokens);   // call each loop() while generating
  void scrollChat(int lines);        // >0 scrolls up (older), <0 down

private:
  static constexpr int H = 135;
  static constexpr int W = 240;
  static constexpr int MARGIN = 3;
  // Chat + input use the vendored Unifont Cyrillic face (16 px tall, see
  // main/ukr_font.h) and word-wrap at spaces.
  static constexpr int LINE_H   = 16;
  static constexpr int STATUS_H = 12;              // top bar, Font0 size 1
  static constexpr int INPUT_H  = 20;
  static constexpr int INPUT_Y  = H - INPUT_H;
  static constexpr int CHAT_Y0  = STATUS_H + 3;
  static constexpr int VISIBLE  = (INPUT_Y - CHAT_Y0) / LINE_H;  // 6 lines
  static constexpr int MAX_LINES = 120;            // scrollback depth (ring)

  static constexpr uint16_t C_BG     = 0x0000;
  static constexpr uint16_t C_BAR    = 0x10A2;     // bar background
  static constexpr uint16_t C_DIV    = 0x4A49;     // divider lines
  static constexpr uint16_t C_USER   = 0x07E0;     // green prompt marker
  static constexpr uint16_t C_TEXT   = 0xFFFF;     // user text
  static constexpr uint16_t C_BOT    = 0x07FF;     // bot text (cyan)
  static constexpr uint16_t C_STATUS = 0xFFE0;     // yellow status text
  static constexpr uint16_t C_DIM    = 0xC618;     // grey

  // The keys the user actually pressed (ASCII) and the Cyrillic rendering of
  // them. latin_ is the source of truth: the whole buffer is re-transliterated
  // on every keystroke so multi-letter sequences ("s" then "h" -> "ш") work.
  std::string latin_;
  std::string input_;
  int      chat_y_   = CHAT_Y0;
  int      cursor_x_ = MARGIN;
  bool     in_bot_   = false;
  uint32_t spin_last_ = 0;
  int      spin_      = 0;

  // Scrollback. Committed lines live in a ring buffer; each stores its text
  // with inline color codes (chars 0x01..0x03, see colorFor). `cur_` is the
  // line still being written at the bottom; offset_ > 0 means the user has
  // scrolled into history (new output snaps back to live).
  std::string lines_[MAX_LINES];
  int      line_count_ = 0;          // total committed lines ever (ring-indexed)
  std::string cur_;
  uint8_t  cur_color_ = 0;
  int      offset_ = 0;

  void chatFont();
  void uiFont();
  static uint16_t colorFor(uint8_t code);
  static uint8_t  codeFor(uint16_t color);
  void drawStatusBar(const char* s, uint16_t color);
  void drawInputBox();
  void refreshInput();               // latin_ -> input_ (transliterate)

public:
  // Latin keycaps can't reach Cyrillic, so keystrokes are transliterated by
  // default. This toggles it off for typing Latin (names, URLs, English).
  void toggleTranslit();
  bool translitOn() const { return translit_; }

private:
  bool translit_ = true;
  void newline();
  void commitLine();
  void drawLine(const std::string& l, int y);
  void redrawChat();
  void snapToLive();
  void writeText(const char* s, uint16_t color);
};
