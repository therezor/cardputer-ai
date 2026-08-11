#include "ui.h"
#include "port.h"
#include "translit.h"
#include "ukr_font.h"
#include <M5Unified.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static auto& D() { return M5.Display; }

// Chat text is Ukrainian, so it needs a font with the full Cyrillic block —
// m5gfx's bundled efont has the Russian letters but not Ґ Є І Ї ґ є і ї.
// 16 px tall, matching LINE_H. m5gfx decodes UTF-8 by default, so print() and
// textWidth() take the same UTF-8 strings the model emits.
static const lgfx::U8g2font ukrFont(u8g2_font_unifont_t_cyrillic);

void ChatUI::chatFont() { D().setFont(&ukrFont); D().setTextSize(1); }
void ChatUI::uiFont()   { D().setFont(&fonts::Font0); D().setTextSize(1); }

// ---------- UTF-8 helpers ----------
// Cyrillic is two bytes per character in UTF-8, so every place that used to
// step or split by byte has to step by character instead; a split sequence
// renders as garbage and corrupts what gets fed back to the tokenizer.

static inline bool utf8IsCont(char c) { return ((uint8_t)c & 0xC0) == 0x80; }

// Bytes in the sequence that starts with `lead`.
static inline int utf8SeqLen(char lead) {
  uint8_t b = (uint8_t)lead;
  if (b < 0x80) return 1;
  if ((b & 0xE0) == 0xC0) return 2;
  if ((b & 0xF0) == 0xE0) return 3;
  if ((b & 0xF8) == 0xF0) return 4;
  return 1;                                  // stray continuation byte
}

// Largest index <= i that starts a character.
static size_t utf8Floor(const char* s, size_t i) {
  while (i > 0 && utf8IsCont(s[i])) i--;
  return i;
}

uint16_t ChatUI::colorFor(uint8_t code) {
  switch (code) {
    case 1:  return C_USER;
    case 3:  return C_BOT;
    default: return C_TEXT;
  }
}
uint8_t ChatUI::codeFor(uint16_t color) {
  if (color == C_USER) return 1;
  if (color == C_BOT)  return 3;
  return 2;
}

void ChatUI::begin() {
  D().setRotation(1);
  D().fillScreen(C_BG);
  D().setTextWrap(false);
  chat_y_     = CHAT_Y0;
  cursor_x_   = MARGIN;
  line_count_ = 0;
  cur_        = "";
  cur_color_  = 0;
  offset_     = 0;
  drawStatusBar("Cardputer AI", C_DIM);
  drawInputBox();
}

// ---------- Status bar (top) ----------

void ChatUI::drawStatusBar(const char* s, uint16_t color) {
  D().fillRect(0, 0, W, STATUS_H, C_BAR);
  D().drawFastHLine(0, STATUS_H, W, C_DIV);
  uiFont();
  D().setTextColor(color, C_BAR);
  D().setCursor(4, 2);
  D().print(s);
}

void ChatUI::status(const char* s) { drawStatusBar(s, C_STATUS); }

void ChatUI::statusf(const char* fmt, ...) {
  char buf[96]; va_list ap; va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap); va_end(ap);
  status(buf);
}

void ChatUI::tickGenerating(int tokens) {
  uint32_t now = millis();
  if (now - spin_last_ < 200) return;        // ~5 fps is enough for a spinner
  spin_last_ = now;
  spin_ = (spin_ + 1) & 3;
  static const char frames[] = "|/-\\";
  char buf[48];
  if (tokens > 0)
    snprintf(buf, sizeof(buf), "%c generating...  %d tok", frames[spin_], tokens);
  else
    snprintf(buf, sizeof(buf), "%c thinking...", frames[spin_]);
  drawStatusBar(buf, C_BOT);
}

// ---------- Input box (bottom) ----------

void ChatUI::drawInputBox() {
  D().fillRect(0, INPUT_Y, W, INPUT_H, C_BAR);
  D().drawFastHLine(0, INPUT_Y, W, C_DIV);
  chatFont();
  // Prompt marker doubles as the input-mode indicator, so the current mode is
  // always visible without opening settings.
  D().setTextColor(translit_ ? C_USER : C_DIM, C_BAR);
  D().setCursor(4, INPUT_Y + 2);
  D().print(translit_ ? "ua>" : "en>");
  int x0 = D().getCursorX();

  // Show the tail of the input if it is wider than the box, so the cursor
  // (= what the user is typing right now) is always visible.
  const char* s = input_.c_str();
  int start = 0;
  const int avail = W - x0 - 12;             // room for the caret block
  while (s[start] && D().textWidth(s + start) > avail)
    start += utf8SeqLen(s[start]);           // step by character, not by byte

  D().setCursor(x0, INPUT_Y + 2);
  if (start > 0) {                           // clipped on the left
    D().setTextColor(C_DIM, C_BAR);
    D().print("<");
    D().setTextColor(C_TEXT, C_BAR);
    // Drop one more character to make room for the '<' marker.
    int skip = s[start] ? utf8SeqLen(s[start]) : 0;
    D().print(s + start + skip);
  } else {
    D().setTextColor(C_TEXT, C_BAR);
    D().print(s);
  }
  D().fillRect(D().getCursorX() + 1, INPUT_Y + 4, 6, 12, C_USER);  // caret
}

// ---------- Chat area / scrollback ----------

void ChatUI::commitLine() {
  lines_[line_count_ % MAX_LINES] = cur_;
  line_count_++;
  cur_ = "";
  cur_color_ = 0;
}

// Draw one stored line (text with inline color codes) at pixel row y.
void ChatUI::drawLine(const std::string& l, int y) {
  int x = MARGIN;
  uint16_t col = C_TEXT;
  std::string seg;
  auto flush = [&]() {
    if (!seg.length()) return;
    D().setTextColor(col, C_BG);
    D().setCursor(x, y);
    D().print(seg.c_str());
    x += D().textWidth(seg.c_str());
    seg = "";
  };
  for (size_t i = 0; i < l.length(); i++) {
    char c = l[i];
    if ((uint8_t)c < 8) { flush(); col = colorFor((uint8_t)c); }
    else seg += c;
  }
  flush();
}

// Repaint the whole chat region from the scrollback buffer.
void ChatUI::redrawChat() {
  chatFont();
  D().fillRect(0, CHAT_Y0, W, INPUT_Y - CHAT_Y0, C_BG);
  int total  = line_count_ + 1;                       // +1 = live line (cur_)
  int oldest = line_count_ > MAX_LINES ? line_count_ - MAX_LINES : 0;
  int end    = total - 1 - offset_;
  int start  = end - VISIBLE + 1;
  if (start < oldest) start = oldest;
  int y = CHAT_Y0;
  for (int idx = start; idx <= end; idx++, y += LINE_H) {
    if (idx == line_count_) drawLine(cur_, y);
    else                    drawLine(lines_[idx % MAX_LINES], y);
  }
  if (offset_ == 0) {                                 // restore live cursor
    chat_y_ = y - LINE_H;
    int x = MARGIN;
    std::string seg;
    for (size_t i = 0; i < cur_.length(); i++) {
      char c = cur_[i];
      if ((uint8_t)c < 8) { x += D().textWidth(seg.c_str()); seg = ""; }
      else seg += c;
    }
    cursor_x_ = x + D().textWidth(seg.c_str());
  }
}

void ChatUI::snapToLive() {
  if (offset_ == 0) return;
  offset_ = 0;
  redrawChat();
}

void ChatUI::scrollChat(int lines) {
  int total   = line_count_ + 1;
  int oldest  = line_count_ > MAX_LINES ? line_count_ - MAX_LINES : 0;
  int max_off = total - VISIBLE - oldest;
  if (max_off < 0) max_off = 0;
  int no = offset_ + lines;
  if (no < 0) no = 0;
  if (no > max_off) no = max_off;
  if (no == offset_) return;
  offset_ = no;
  redrawChat();
  if (offset_ > 0) statusf("history -%d    fn+. = back to latest", offset_);
  else             drawStatusBar("Cardputer AI", C_DIM);
}

void ChatUI::newline() {
  snapToLive();
  commitLine();
  chat_y_  += LINE_H;
  cursor_x_ = MARGIN;
  const int region_h = INPUT_Y - CHAT_Y0;
  while (chat_y_ + LINE_H > INPUT_Y - 1) {   // scroll chat region up one line
    D().copyRect(0, CHAT_Y0, W, region_h - LINE_H, 0, CHAT_Y0 + LINE_H);
    D().fillRect(0, CHAT_Y0 + region_h - LINE_H, W, LINE_H, C_BG);
    chat_y_ -= LINE_H;
  }
}

void ChatUI::writeText(const char* s, uint16_t color) {
  snapToLive();
  chatFont();
  D().setTextColor(color, C_BG);
  const uint8_t code = codeFor(color);
  auto emit = [&](const char* t) {           // record into the live line
    if (cur_color_ != code) { cur_ += (char)code; cur_color_ = code; }
    cur_ += t;
  };
  const int maxw = W - MARGIN;
  const char* p = s;
  char buf[48];
  while (*p) {
    if (*p == '\n') { newline(); p++; continue; }
    // Take one word (or a single space) so we can wrap at word boundaries.
    const char* start = p;
    if (*p == ' ') p++;
    else while (*p && *p != ' ' && *p != '\n') p++;
    int len = p - start;
    if (len > (int)sizeof(buf) - 1) {
      // Back the cut off to a character boundary — Cyrillic is two bytes per
      // letter, so a blind cut at 47 bytes can land inside one.
      len = (int)utf8Floor(start, sizeof(buf) - 1);
      p = start + len;
    }
    memcpy(buf, start, len); buf[len] = 0;

    int wpx = D().textWidth(buf);
    if (cursor_x_ + wpx > maxw) {
      if (MARGIN + wpx > maxw) {             // word longer than a whole line
        for (int i = 0; i < len; ) {
          int cl = utf8SeqLen(buf[i]);
          if (i + cl > len) cl = len - i;
          char t[5];
          memcpy(t, buf + i, cl); t[cl] = 0;
          int cw = D().textWidth(t);
          if (cursor_x_ + cw > maxw) newline();
          D().setCursor(cursor_x_, chat_y_);
          D().print(t);
          emit(t);
          cursor_x_ += cw;
          i += cl;
        }
        continue;
      }
      newline();
      if (buf[0] == ' ') continue;           // swallow the space at a wrap
    }
    D().setCursor(cursor_x_, chat_y_);
    D().print(buf);
    emit(buf);
    cursor_x_ += wpx;
  }
}

void ChatUI::ready() {
  D().fillScreen(C_BG);
  chat_y_     = CHAT_Y0;
  cursor_x_   = MARGIN;
  input_      = "";
  line_count_ = 0;
  cur_        = "";
  cur_color_  = 0;
  offset_     = 0;
  drawStatusBar("Cardputer AI", C_DIM);
  drawInputBox();
}

void ChatUI::repaint() {
  D().fillScreen(C_BG);
  drawStatusBar("Cardputer AI", C_DIM);
  redrawChat();
  drawInputBox();
}

// ---------- Settings ----------

void ChatUI::showSettings(const char* title, const std::string* names, const std::string* values,
                          int n, int selected) {
  D().fillScreen(C_BG);
  chatFont();
  D().setTextColor(C_STATUS, C_BG);
  D().setCursor(4, 2);
  D().print(title);
  int header = LINE_H + 4;
  D().drawFastHLine(0, header, W, C_DIV);

  const int footer_h = 12;
  const int row_h = LINE_H + 2;

  for (int i = 0; i < n; i++) {
    int y = header + 4 + i * row_h;
    uint16_t bg = (i == selected) ? 0x2A4A : C_BG;
    D().fillRect(0, y, W, row_h, bg);
    D().setTextColor(i == selected ? C_TEXT : C_DIM, bg);
    D().setCursor(6, y + 1);
    D().print(names[i].c_str());
    D().setTextColor(i == selected ? C_STATUS : C_BOT, bg);
    D().setCursor(W - 8 - D().textWidth(values[i].c_str()), y + 1);
    D().print(values[i].c_str());
  }

  D().fillRect(0, H - footer_h, W, footer_h, C_BAR);
  D().drawFastHLine(0, H - footer_h, W, C_DIV);
  uiFont();
  D().setCursor(4, H - footer_h + 2);
  D().setTextColor(C_DIM, C_BAR);
  D().print("; .  move    , /  adjust    tab  back");
}

void ChatUI::fatal(const char* s) {
  D().fillScreen(0x8000);
  D().setTextWrap(true);
  chatFont();
  D().setCursor(4, 4);
  D().setTextColor(0xFFFF);
  D().print("FATAL: ");
  D().print(s);
  while (1) delay_ms(1000);
}

// ---------- Selector ----------

void ChatUI::showSelector(const char* title, const std::string* items, const std::string* subs,
                          int n, int selected) {
  D().fillScreen(C_BG);
  chatFont();
  D().setTextColor(C_STATUS, C_BG);
  D().setCursor(4, 2);
  D().print(title);
  int header = LINE_H + 4;
  D().drawFastHLine(0, header, W, C_DIV);

  const int footer_h = 12;
  const int row_h = LINE_H + 2;
  const int avail = H - header - footer_h - 2;
  const int rows = avail / row_h;

  int first = 0;
  if (selected >= rows) first = selected - rows + 1;

  for (int i = 0; i < rows && first + i < n; i++) {
    int idx = first + i;
    int y = header + 2 + i * row_h;
    uint16_t bg = (idx == selected) ? 0x2A4A : C_BG;
    D().fillRect(0, y, W, row_h, bg);
    D().setTextColor(idx == selected ? C_TEXT : C_DIM, bg);
    D().setCursor(6, y + 1);
    D().print(items[idx].c_str());
  }

  D().fillRect(0, H - footer_h, W, footer_h, C_BAR);
  uiFont();
  D().setCursor(4, H - footer_h + 2);
  D().setTextColor(C_DIM, C_BAR);
  D().print("; / .  move    enter  select");
}

void ChatUI::showProgress(const char* label, size_t done, size_t total) {
  static size_t last = SIZE_MAX;
  if (done != total && done - last < total / 50) return; // throttle
  last = done;
  D().fillRect(0, H/2 - 16, W, 32, C_BG);
  uiFont();
  D().setCursor(4, H/2 - 12);
  D().setTextColor(C_STATUS, C_BG);
  D().print(label);
  int pct = total ? (int)(done * 100 / total) : 0;
  D().drawRect(4, H/2 + 2, W - 8, 8, C_TEXT);
  D().fillRect(5, H/2 + 3, (W - 10) * pct / 100, 6, 0x07E0);
  D().setCursor(W - 36, H/2 - 12);
  char b[16]; snprintf(b, sizeof(b), "%d%%", pct);
  D().print(b);
}

// ---------- Input ----------

// Re-render input_ from the Latin keystrokes. Lines starting with '/' are
// commands ("/new") and stay Latin — transliterating them would turn /new into
// /неш and the command would never match.
void ChatUI::refreshInput() {
  bool raw = !translit_ || (!latin_.empty() && latin_[0] == '/');
  input_ = raw ? latin_ : translitToCyrillic(latin_);
}

void ChatUI::toggleTranslit() {
  translit_ = !translit_;
  refreshInput();
  drawInputBox();
  statusf("%s    fn+u switches", translit_ ? "UA - typing Ukrainian"
                                           : "EN - typing Latin");
}

void ChatUI::onChar(char c) {
  if (c < 32 || c > 126) return;
  if (latin_.length() > 200) return;
  latin_ += c;
  refreshInput();
  drawInputBox();
}
void ChatUI::onBackspace() {
  if (latin_.empty()) return;
  latin_.pop_back();                  // latin_ is ASCII, so one byte is one key
  refreshInput();
  drawInputBox();
}
std::string ChatUI::takeInput() {
  std::string s = input_;
  latin_.clear();
  input_.clear();
  drawInputBox();
  return s;
}

void ChatUI::appendUser(const std::string& s) {
  snapToLive();
  cursor_x_ = MARGIN;
  writeText("> ", C_USER);
  writeText(s.c_str(), C_TEXT);
  newline();
}
void ChatUI::beginBotReply() { in_bot_ = true; cursor_x_ = MARGIN; }
void ChatUI::appendBot(const char* piece) { writeText(piece, C_BOT); }
void ChatUI::endBotReply(int tokens, uint32_t ms) {
  newline();
  in_bot_ = false;
  if (ms > 0) statusf("%d tok / %.1fs (%.2f t/s)", tokens, ms/1000.0f, tokens*1000.0f/ms);
}
