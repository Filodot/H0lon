--[==[
H0lon text-symbol filter (LaTeX output only).

The text font (Times New Roman by default) lacks many mathematical symbols that agents and
sources put into plain text: ⟨x, y⟩, ⇒, ∀, ℝ … XeLaTeX then silently drops them («Missing
character»). This filter sets such characters in math mode, where unicode-math uses the math
font (Cambria Math), and keeps the plain character for PDF bookmarks via \texorpdfstring.
Math, code and raw blocks are not touched (the filter only rewrites Str elements).
]==]

-- Characters routed to the math font. Keep the list to symbols that exist in common math
-- fonts; ordinary punctuation and letters stay in the text font.
local SYMBOLS = {}
for _, cp in ipairs({
  0x27E8, 0x27E9, 0x27EA, 0x27EB, 0x27E6, 0x27E7, -- ⟨ ⟩ ⟪ ⟫ ⟦ ⟧
  0x21A6, 0x21D2, 0x21D0, 0x21D4, 0x27F9, 0x27F8, 0x27FA, -- ↦ ⇒ ⇐ ⇔ ⟹ ⟸ ⟺
  0x2218, 0x2282, 0x2283, 0x2286, 0x2287, 0x222A, 0x2229, 0x2205, -- ∘ ⊂ ⊃ ⊆ ⊇ ∪ ∩ ∅
  0x2200, 0x2203, 0x2204, 0x2208, 0x2209, 0x220B, -- ∀ ∃ ∄ ∈ ∉ ∋
  0x2261, 0x2262, 0x221D, 0x2220, 0x22A5, 0x2225, -- ≡ ≢ ∝ ∠ ⊥ ∥
  0x2295, 0x2297, 0x2299, 0x2227, 0x2228, 0x22A4, 0x22A2, 0x22A8, -- ⊕ ⊗ ⊙ ∧ ∨ ⊤ ⊢ ⊨
  0x220E, 0x22C5, 0x22EF, 0x22EE, 0x22F1, -- ∎ ⋅ ⋯ ⋮ ⋱
  0x2115, 0x2124, 0x211A, 0x211D, 0x2102, 0x2119, -- ℕ ℤ ℚ ℝ ℂ ℙ
  0x2A7D, 0x2A7E, 0x226A, 0x226B, 0x227A, 0x227B, -- ⩽ ⩾ ≪ ≫ ≺ ≻
}) do
  SYMBOLS[utf8.char(cp)] = true
end

-- Cyrillic inside formulas ($x_{ср}$, $\mathrm{знач}$): math fonts have no Cyrillic, XeLaTeX
-- drops the letters. Runs of Cyrillic (with inner spaces, hyphens, dots) go into \text{…},
-- which uses the text font; nesting in \text, \mathrm or \operatorname is harmless.
local function is_cyrillic(cp)
  return cp >= 0x0400 and cp <= 0x04FF
end

function Math(el)
  if not FORMAT:match('latex') then return nil end
  local has = false
  for _, cp in utf8.codes(el.text) do
    if is_cyrillic(cp) then
      has = true
      break
    end
  end
  if not has then return nil end
  local out, run, pending = {}, {}, {}
  local function flush()
    if #run > 0 then
      out[#out + 1] = '\\text{' .. table.concat(run) .. '}'
      run = {}
    end
    for _, c in ipairs(pending) do out[#out + 1] = c end
    pending = {}
  end
  for _, cp in utf8.codes(el.text) do
    local ch = utf8.char(cp)
    if is_cyrillic(cp) then
      for _, c in ipairs(pending) do
        if #run > 0 then run[#run + 1] = c else out[#out + 1] = c end
      end
      pending = {}
      run[#run + 1] = ch
    elseif #run > 0 and (ch == ' ' or ch == '-' or ch == '.') then
      pending[#pending + 1] = ch
    else
      flush()
      out[#out + 1] = ch
    end
  end
  flush()
  el.text = table.concat(out)
  return el
end

local function math_inline(ch)
  return pandoc.RawInline('latex', '\\texorpdfstring{\\ensuremath{' .. ch .. '}}{' .. ch .. '}')
end

function Str(el)
  if not FORMAT:match('latex') then return nil end
  local text = el.text
  local found = false
  for _, cp in utf8.codes(text) do
    if SYMBOLS[utf8.char(cp)] then
      found = true
      break
    end
  end
  if not found then return nil end
  local out, buf = {}, {}
  for _, cp in utf8.codes(text) do
    local ch = utf8.char(cp)
    if SYMBOLS[ch] then
      if #buf > 0 then
        out[#out + 1] = pandoc.Str(table.concat(buf))
        buf = {}
      end
      out[#out + 1] = math_inline(ch)
    else
      buf[#buf + 1] = ch
    end
  end
  if #buf > 0 then out[#out + 1] = pandoc.Str(table.concat(buf)) end
  return out
end
