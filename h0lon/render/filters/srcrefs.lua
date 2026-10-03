--[==[
H0lon source-anchor filter.

* Anchors [[H1:p3]], [[S1:s12]], [[V1:12:34]], [[P2:p45-47]] may be glued to punctuation,
  follow each other or share one Str. Consecutive anchors (optionally separated by spaces)
  are merged into one compact superscript: LaTeX \hzsrc{H1:p3, S1:s12},
  HTML <sup class="h0src">H1:p3, S1:s12</sup>. The space before an anchor is dropped so
  the mark sticks to the preceding word, like a footnote mark.
* Metadata `h0lon-clean: true` removes the anchors together with the preceding space.
* HTML comments (<!-- ... -->, e.g. <!-- src: H1.b007 -->) never reach the output.
* "[неразборчиво]" is set in gray.
]==]

local IS_LATEX = FORMAT:match('latex') ~= nil or FORMAT:match('beamer') ~= nil
local IS_HTML = FORMAT:match('html') ~= nil

-- Source id (letter + alnum) ':' locator without spaces or brackets.
-- Explicit ASCII sets: %s/%w/%u depend on LC_CTYPE, which on Windows is the ANSI code page
-- (cp1251 treats 0xA0, a byte of UTF-8 «Р», as a space).
local ANCHOR_PATTERN = '%[%[([A-Z][A-Za-z0-9]*:[^%[%] \t\r\n]+)%]%]'
local UNREADABLE = '[неразборчиво]'

local clean = false
local in_header = false -- set while the inlines of a Header are processed

local function meta_truthy(v)
  if v == nil then return false end
  if type(v) == 'boolean' then return v end
  local s = pandoc.utils.stringify(v):lower()
  return s == 'true' or s == 'yes' or s == '1'
end

local function latex_escape(s)
  return (s:gsub('[\\{}$&#^_%%~]', function(c)
    if c == '\\' then return '\\textbackslash{}' end
    if c == '^' then return '\\textasciicircum{}' end
    if c == '~' then return '\\textasciitilde{}' end
    return '\\' .. c
  end))
end

local function html_escape(s)
  return (s:gsub('[&<>"]', { ['&'] = '&amp;', ['<'] = '&lt;', ['>'] = '&gt;', ['"'] = '&quot;' }))
end

local function is_comment(raw)
  return (raw.format == 'html' or raw.format == 'html5')
    and raw.text:match('^[ \t\r\n]*<!%-%-.*%-%->[ \t\r\n]*$') ~= nil
end

local function is_space(el)
  return el ~= nil and (el.t == 'Space' or el.t == 'SoftBreak')
end

-- `inline`: the anchors open the inline list (a table cell, a Source Doc heading), so they
-- are set on the baseline as a small gray locator instead of a superscript.
local function anchor_mark(ids, inline)
  local text = table.concat(ids, ', ')
  if IS_LATEX then
    local macro = inline and '\\hzsrcinline{' or '\\hzsrc{'
    return pandoc.RawInline('latex', macro .. latex_escape(text) .. '}')
  elseif IS_HTML then
    -- In headings the mark is hidden from the accessibility tree, which Chromium uses for
    -- the PDF outline (LaTeX drops it from bookmarks via \pdfstringdefDisableCommands).
    local hidden = in_header and ' aria-hidden="true"' or ''
    local html
    if inline then
      html = '<span class="h0src h0src-inline"' .. hidden .. '>' .. html_escape(text) .. '</span>'
    else
      -- U+2060 WORD JOINER: no line break between the word (or formula) and its mark.
      html = '<sup class="h0src"' .. hidden .. '>\u{2060}' .. html_escape(text) .. '</sup>'
    end
    return pandoc.RawInline('html', html)
  end
  if inline then return pandoc.Str(text) end
  return pandoc.Superscript({ pandoc.Str(text) })
end

local function unreadable_mark()
  if IS_LATEX then
    return pandoc.RawInline('latex', '\\hzunreadable{' .. UNREADABLE .. '}')
  elseif IS_HTML then
    return pandoc.RawInline('html', '<span class="unreadable">' .. UNREADABLE .. '</span>')
  end
  return pandoc.Str(UNREADABLE)
end

-- The anchors open the line: nothing before them, or (HTML) only the heading number that
-- blocks.lua has put in front of the heading text.
local function opens_line(out)
  if #out == 0 then return true end
  return in_header and #out == 1 and out[1].t == 'Span'
    and out[1].classes:includes('header-section-number')
end

-- Split a Str into pieces: plain strings and {anchor = id} / {unreadable = true} markers.
local function split_str(text, out)
  local pos = 1
  while pos <= #text do
    local s, e, id = text:find(ANCHOR_PATTERN, pos)
    local us, ue = text:find(UNREADABLE, pos, true)
    if us and (not s or us < s) then
      if us > pos then out[#out + 1] = pandoc.Str(text:sub(pos, us - 1)) end
      out[#out + 1] = { unreadable = true }
      pos = ue + 1
    elseif s then
      if s > pos then out[#out + 1] = pandoc.Str(text:sub(pos, s - 1)) end
      out[#out + 1] = { anchor = id }
      pos = e + 1
    else
      out[#out + 1] = pandoc.Str(text:sub(pos))
      break
    end
  end
end

local function process_inlines(inlines)
  -- 1. Merge adjacent Str (an anchor may be split by the reader) and drop comments.
  local merged = {}
  local dropped = false
  for _, el in ipairs(inlines) do
    local last = merged[#merged]
    if el.t == 'RawInline' and is_comment(el) then
      dropped = true -- spaces around the removed comment collapse below
    elseif dropped and is_space(el) and (last == nil or is_space(last)) then
      -- skip: double space (or leading space) left by a removed comment
    elseif el.t == 'Str' and last and last.t == 'Str' then
      merged[#merged] = pandoc.Str(last.text .. el.text)
    else
      merged[#merged + 1] = el
    end
  end
  if dropped and is_space(merged[#merged]) then table.remove(merged) end

  -- 2. Split Str into text and markers; quick exit when nothing to do.
  local pieces = {}
  local found = false
  for _, el in ipairs(merged) do
    if el.t == 'Str' and (el.text:find('[[', 1, true) or el.text:find(UNREADABLE, 1, true)) then
      split_str(el.text, pieces)
      found = true
    else
      pieces[#pieces + 1] = el
    end
  end
  if not found then
    if #merged == #inlines then return nil end
    return pandoc.Inlines(merged)
  end

  -- 3. Group consecutive anchors (separated only by spaces) and emit marks.
  local out = pandoc.List({})
  local i = 1
  while i <= #pieces do
    local p = pieces[i]
    if p.unreadable then
      out:insert(unreadable_mark())
      i = i + 1
    elseif p.anchor then
      local ids = { p.anchor }
      local j = i + 1
      while true do
        local k = j
        while is_space(pieces[k]) do k = k + 1 end
        if pieces[k] and pieces[k].anchor then
          ids[#ids + 1] = pieces[k].anchor
          j = k + 1
        else
          break
        end
      end
      local had_space_before = is_space(out[#out])
      local space = had_space_before and out:remove(#out) or nil
      if not clean then
        local opens = opens_line(out)
        if opens and #out > 0 and space then out:insert(space) end -- "1.1. H1:p3 Страница"
        out:insert(anchor_mark(ids, opens))
      elseif not had_space_before and #out == 0 then
        -- Anchor opens the line ("[[H1:p1]] Страница 1"): drop the space after it too.
        while is_space(pieces[j]) do j = j + 1 end
      end
      i = j
    else
      out:insert(p)
      i = i + 1
    end
  end
  return pandoc.Inlines(out)
end

local function process_rawblock(raw)
  if is_comment(raw) then return {} end
end

function Pandoc(doc)
  clean = meta_truthy(doc.meta['h0lon-clean'])
  -- Headings first (marks hidden from the PDF outline), then everything else; the second
  -- pass finds nothing left to do in headings.
  doc = doc:walk({
    Header = function(h)
      in_header = true
      local out = h:walk({ Inlines = process_inlines })
      in_header = false
      return out
    end,
  })
  return doc:walk({
    Inlines = process_inlines,
    RawBlock = process_rawblock,
  })
end
