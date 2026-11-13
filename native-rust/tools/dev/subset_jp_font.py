"""Builds assets/fonts/NotoSansJP-Regular-subset.ttf: Noto Sans JP at weight 400 with
only the characters the Japanese translations use, plus all kana and common
punctuation. Run it again after changing src/i18n/table.rs.

    uv run --no-project --with fonttools python tools/dev/subset_jp_font.py NotoSansJP[wght].ttf

The source font is the variable Noto Sans JP from Google Fonts (SIL OFL 1.1).
"""
import re
import sys
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

here = Path(__file__).resolve().parents[2]
table = (here / 'src/i18n/table.rs').read_text(encoding='utf-8')
# The third string of every (English, Indonesian, Japanese) row.
japanese = re.findall(r'^\s*\("(?:[^"\\]|\\.)*", "(?:[^"\\]|\\.)*", "((?:[^"\\]|\\.)*)"\),$', table, re.M)
chars = set(''.join(japanese)) | set('日本語')
chars |= {chr(c) for c in range(0x3040, 0x3100)}  # hiragana and katakana
chars |= {chr(c) for c in range(0x3000, 0x3040)}  # CJK punctuation
chars |= {chr(c) for c in range(0xFF01, 0xFF5F)}  # full-width forms

font = TTFont(sys.argv[1])
if 'fvar' in font:
    font = instancer.instantiateVariableFont(font, {'wght': 400})
options = subset.Options()
options.layout_features = ['*']
options.name_IDs = ['*']
options.notdef_outline = True
subsetter = subset.Subsetter(options)
subsetter.populate(text=''.join(sorted(chars)))
subsetter.subset(font)
out = here / 'assets/fonts/NotoSansJP-Regular-subset.ttf'
font.save(out)
print(f'{len(japanese)} strings, {len(chars)} characters, {out.stat().st_size // 1024} KiB -> {out}')
