// Latin -> Ukrainian Cyrillic phonetic transliteration for the Cardputer.
//
// The keyboard is physically Latin: there is no way to press "ї". Rather than
// overlay a ЙЦУКЕН layout the user would have to memorise against unlabelled
// keycaps, we transliterate phonetically as they type — "pryvit" becomes
// "привіт".
//
// The raw Latin string stays the source of truth and the whole buffer is
// re-transliterated on every keystroke. That is what makes multi-letter
// sequences work: typing "s" shows "с" and typing "h" after it turns the pair
// into "ш", which incremental per-key mapping could not do.
#pragma once

#include <string>

// Transliterate `latin` (ASCII) into UTF-8 Ukrainian Cyrillic.
// Characters with no mapping (digits, punctuation, spaces) pass through.
std::string translitToCyrillic(const std::string& latin);
