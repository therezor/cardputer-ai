#include "translit.h"

#include <string.h>

namespace {

struct Rule { const char* latin; const char* cyr; };

// Longest match wins, so the table is ordered longest-first. The scheme is the
// usual Ukrainian phonetic one (sh/ch/zh/ya/yu/ye/yi), plus single-key
// shortcuts for letters that would otherwise need a digraph:
//   q -> ь (soft sign)   w -> ш   x -> х   c -> ц   y -> и
// Both "h" and "g" give г, which is what people actually type; the rare ґ is
// "gg". "ї" is "yi" or "ji", so Київ is typed "Kyjiv" — plain "Kyiv" reads its
// "yi" as the digraph and comes out "Кїв".
const Rule RULES[] = {
  {"shch", "щ"},
  {"zh", "ж"}, {"ch", "ч"}, {"sh", "ш"}, {"kh", "х"}, {"gg", "ґ"},
  {"ya", "я"}, {"yu", "ю"}, {"ye", "є"}, {"yi", "ї"}, {"yo", "йо"},
  {"ja", "я"}, {"ju", "ю"}, {"je", "є"}, {"ji", "ї"},
  {"a", "а"}, {"b", "б"}, {"c", "ц"}, {"d", "д"}, {"e", "е"}, {"f", "ф"},
  {"g", "г"}, {"h", "г"}, {"i", "і"}, {"j", "й"}, {"k", "к"}, {"l", "л"},
  {"m", "м"}, {"n", "н"}, {"o", "о"}, {"p", "п"}, {"q", "ь"}, {"r", "р"},
  {"s", "с"}, {"t", "т"}, {"u", "у"}, {"v", "в"}, {"w", "ш"}, {"x", "х"},
  {"y", "и"}, {"z", "з"},
};

inline char lower(char c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

// Uppercase a UTF-8 Cyrillic string produced by the table above. The table's
// outputs live in U+0430-U+044F (а-я, where upper = lower - 0x20) plus і ї є ґ
// from the U+0450-U+049F range (where upper = lower - 1).
void appendUpper(std::string& out, const char* s) {
  bool first = true;
  for (const unsigned char* p = (const unsigned char*)s; *p; ) {
    if (*p < 0x80) { out += (char)*p; p++; first = false; continue; }
    unsigned cp = ((*p & 0x1F) << 6) | (p[1] & 0x3F);
    if (first) {
      if (cp >= 0x430 && cp <= 0x44F)      cp -= 0x20;   // а-я -> А-Я
      else if (cp == 0x456 || cp == 0x457 ||
               cp == 0x454 || cp == 0x491) cp -= 1;      // і ї є ґ
    }
    out += (char)(0xC0 | (cp >> 6));
    out += (char)(0x80 | (cp & 0x3F));
    p += 2;
    first = false;
  }
}

}  // namespace

namespace {
// In Ukrainian, ї only ever follows a vowel (or starts a word) — Україна,
// твої, свої. So a bare "i" after a vowel is almost always ї, not і. Without
// this, "Ukraini" produced "Украіні", which is a misspelling the model does
// not recognise. ("yi"/"ji" still work and are matched earlier.)
inline bool isLatinVowel(char c) {
  switch (c | 0x20) {
    case 'a': case 'e': case 'o': case 'u': case 'y': return true;
    default: return false;
  }
}
}  // namespace

std::string translitToCyrillic(const std::string& latin) {
  std::string out;
  out.reserve(latin.size() * 2);

  size_t i = 0;
  while (i < latin.size()) {
    // Word-final "yi"/"yy" is the adjective ending -ий (Український,
    // наївний), not ї. Without this the ї digraph swallows it and every
    // adjective comes out misspelled.
    if ((latin[i] | 0x20) == 'y' && i + 1 < latin.size()) {
      char n1 = latin[i + 1] | 0x20;
      bool at_end = (i + 2 >= latin.size()) ||
                    !((latin[i + 2] | 0x20) >= 'a' && (latin[i + 2] | 0x20) <= 'z');
      if ((n1 == 'i' || n1 == 'y') && at_end) {
        if (latin[i] >= 'A' && latin[i] <= 'Z') appendUpper(out, "ий");
        else                                    out += "ий";
        i += 2;
        continue;
      }
    }
    if ((latin[i] == 'i' || latin[i] == 'I') && i > 0 && isLatinVowel(latin[i - 1])) {
      if (latin[i] == 'I') appendUpper(out, "ї");
      else                 out += "ї";
      i++;
      continue;
    }
    bool matched = false;
    for (const Rule& r : RULES) {
      size_t n = strlen(r.latin);
      if (i + n > latin.size()) continue;
      bool eq = true;
      for (size_t k = 0; k < n; k++) {
        if (lower(latin[i + k]) != r.latin[k]) { eq = false; break; }
      }
      if (!eq) continue;
      // Capitalise the output when the user capitalised the first letter of
      // the sequence ("Kyiv" -> "Київ", not "кИЇВ").
      if (latin[i] >= 'A' && latin[i] <= 'Z') appendUpper(out, r.cyr);
      else                                    out += r.cyr;
      i += n;
      matched = true;
      break;
    }
    if (!matched) out += latin[i++];   // digits, spaces, punctuation
  }
  return out;
}
