// Ukrainian-capable Cyrillic bitmap font for the chat UI.
//
// GNU Unifont 8x16 via u8g2, covering U+0400-U+052F in full (including the
// Ukrainian-only Ґ Є І Ї ґ є і ї that m5gfx's bundled efont omits) plus ASCII.
// Rendered through lgfx::U8g2font — see ChatUI::chatFont() in ui.cpp.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern const uint8_t u8g2_font_unifont_t_cyrillic[];

#ifdef __cplusplus
}
#endif
