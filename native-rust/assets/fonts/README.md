# Bundled fonts

Subsets embedded into the binary by `src/m3.rs`.

- `Roboto-*-subset.ttf`: Roboto 3 (SIL Open Font License, `Roboto-OFL.txt`), Latin and common symbols only.
- `NotoSansJP-Regular-subset.ttf`: Noto Sans JP (SIL Open Font License, `NotoSansJP-OFL.txt`) at weight 400, with the characters of the Japanese translations, all kana and CJK punctuation. Rebuild it after changing `src/i18n/table.rs` with `uv run --no-project --with fonttools python tools/dev/subset_jp_font.py "NotoSansJP[wght].ttf"`.
- `MaterialSymbolsRounded*-subset.ttf`: Material Symbols Rounded (Apache 2.0, `MaterialSymbols-LICENSE.txt`), static instances at `opsz=24 wght=400 GRAD=0` with `FILL=0` and `FILL=1`, holding only the icons listed in `m3::icons`.

To add an icon, append its codepoint and regenerate with fontTools:

```sh
src="MaterialSymbolsRounded[FILL,GRAD,opsz,wght].ttf"
pyftsubset "$src" --unicodes="U+E1B8,U+E030,..." --layout-features='' --output-file=sym.ttf
fonttools varLib.instancer sym.ttf FILL=0 GRAD=0 opsz=24 wght=400 -o MaterialSymbolsRounded-subset.ttf
fonttools varLib.instancer sym.ttf FILL=1 GRAD=0 opsz=24 wght=400 -o MaterialSymbolsRounded-Filled-subset.ttf
```
