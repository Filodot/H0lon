--[==[
H0lon blocks filter.

* Semantic fenced divs (::: {.definition #def:x title="..."}) become hz* environments of
  the LaTeX template (title -> optional argument, id -> \label) or, for HTML, divs with a
  numbered label ("Определение 2.1.") that the template CSS styles.
* A header with class .appendix switches to appendices (\hzappendix in LaTeX, letter
  numbering А, Б, В ... in HTML).
* HTML: headers are numbered here (1., 1.1., Приложение А.) so that numbers match LaTeX.
* Russian typography: straight double quotes become «ёлочки».
* Metadata: `sources` is normalised into `h0lon-sources` (id, title, details) for the
  title page; for HTML the running-head strings for CSS are computed and a missing `date`
  becomes today's date (the LaTeX title page prints \today).

Patterns: on Windows Pandoc's Lua runs with the ANSI code page as LC_CTYPE (for example
Russian_Russia.1251), where %s, %w, %a, %u ... also match bytes of UTF-8 Cyrillic (0xA0 in
«Р»). Only explicit ASCII sets ([ \t\r\n], [A-Za-z0-9]) are used on text.
]==]

local stringify = pandoc.utils.stringify

local IS_LATEX = FORMAT:match('latex') ~= nil or FORMAT:match('beamer') ~= nil
local IS_HTML = FORMAT:match('html') ~= nil

-- class -> {env, label, numbered, style}
-- style: 'runin' (label + text on one line), 'bar' (label on its own line, gray bar),
--        'note' (service block), 'theorem' (label column)
local BLOCKS = {
  definition = { env = 'hzdefinition', label = 'Определение', numbered = true, style = 'runin' },
  theorem = { env = 'hztheorem', label = 'Теорема', numbered = true, style = 'theorem' },
  lemma = { env = 'hzlemma', label = 'Лемма', numbered = true, style = 'statement' },
  proposition = { env = 'hzproposition', label = 'Предложение', numbered = true, style = 'statement' },
  corollary = { env = 'hzcorollary', label = 'Следствие', numbered = true, style = 'statement' },
  proof = { env = 'hzproof', label = 'Доказательство', numbered = false, style = 'bar' },
  example = { env = 'hzexample', label = 'Пример', numbered = true, style = 'runin' },
  problem = { env = 'hzproblem', label = 'Задача', numbered = true, style = 'runin' },
  solution = { env = 'hzsolution', label = 'Решение', numbered = false, style = 'bar' },
  remark = { env = 'hzremark', label = 'Замечание', numbered = false, style = 'runin' },
  editorial = { env = 'hzeditorial', label = 'Редакторское дополнение', numbered = false, style = 'note' },
  conflict = { env = 'hzconflict', label = 'Расхождение источников', numbered = false, style = 'note' },
  uncertain = { env = 'hzuncertain', label = 'Требует проверки', numbered = false, style = 'note' },
  ['author-question'] = { env = 'hzauthorquestion', label = 'Вопрос автора', numbered = false, style = 'note' },
}
-- Lookup order when a div has several classes.
local BLOCK_ORDER = {
  'definition', 'theorem', 'lemma', 'proposition', 'corollary', 'proof', 'example',
  'problem', 'solution', 'remark', 'editorial', 'conflict', 'uncertain', 'author-question',
}

-- Same sequence as polyglossia's \Asbuk.
local ASBUK = {
  'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'К', 'Л', 'М', 'Н', 'О', 'П',
  'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Э', 'Ю', 'Я',
}

local KIND_LABELS = {
  handwritten = 'рукопись',
  ['pdf-text'] = 'PDF',
  ['pdf-scan'] = 'скан PDF',
  pdf = 'PDF',
  slides = 'слайды',
  pptx = 'слайды',
  video = 'видео',
  audio = 'аудио',
  web = 'веб-страница',
  docx = 'DOCX',
  md = 'Markdown',
  tex = 'LaTeX',
  book = 'книга',
  textbook = 'учебник',
}

local UNIT_LABELS = {
  { 'pages', 'стр.' },
  { 'slides', 'сл.' },
  { 'minutes', 'мин' },
  { 'frames', 'кадр.' },
}

---------------------------------------------------------------- helpers

local function meta_truthy(v)
  if v == nil then return false end
  if type(v) == 'boolean' then return v end
  local s = stringify(v):lower()
  return s == 'true' or s == 'yes' or s == '1'
end

local function block_kind(div)
  for _, cls in ipairs(BLOCK_ORDER) do
    if div.classes:includes(cls) then return cls end
  end
  return nil
end

local function parse_inlines(text)
  if text == nil or text == '' then return nil end
  local doc = pandoc.read(text, 'markdown')
  local first = doc.blocks[1]
  if first and (first.t == 'Para' or first.t == 'Plain') then
    return first.content
  end
  return pandoc.Inlines({ pandoc.Str(text) })
end

local function concat_inlines(...)
  local out = pandoc.Inlines({})
  for _, part in ipairs({ ... }) do
    if part ~= nil then out:extend(part) end
  end
  return out
end

local function latex(s) return pandoc.RawInline('latex', s) end

local function is_html_comment(el)
  return (el.t == 'RawBlock' or el.t == 'RawInline')
    and (el.format == 'html' or el.format == 'html5')
    and el.text:match('^[ \t\r\n]*<!%-%-.*%-%->[ \t\r\n]*$') ~= nil
end

-- A comment (<!-- src: H1.b002 -->) opening a block's first paragraph must not separate
-- the block label from the text; srcrefs.lua removes the remaining comments later.
local function strip_leading_comments(inlines)
  local i = 1
  local found = false
  while inlines[i] ~= nil do
    local el = inlines[i]
    if is_html_comment(el) then
      found = true
    elseif not (found and (el.t == 'Space' or el.t == 'SoftBreak')) then
      break
    end
    i = i + 1
  end
  if i == 1 then return inlines end
  local out = pandoc.Inlines({})
  for k = i, #inlines do out:insert(inlines[k]) end
  return out
end

-- Pandoc writes an internal link [x](#id) as \hyperref[toLabel(id)]{x}, and toLabel escapes
-- every non-ASCII character (опр -> ux43eux43fux440). The LaTeX writer itself is asked for
-- the label so that \label{} in the environment always matches the links.
local label_cache = {}

local function fallback_label(id)
  local out = {}
  for _, code in utf8.codes(id) do
    local ch = utf8.char(code)
    if ch:match('^[A-Za-z0-9_+=:;.-]$') then
      out[#out + 1] = ch
    else
      out[#out + 1] = string.format('ux%x', code)
    end
  end
  return table.concat(out)
end

local function latex_label(id)
  if label_cache[id] == nil then
    local ok, text = pcall(function()
      local probe = pandoc.Div({ pandoc.Plain({ pandoc.Str('x') }) }, { id = id })
      return pandoc.write(pandoc.Pandoc({ probe }), 'latex')
    end)
    local label = ok and text:match('\\label{(.-)}') or nil
    if label == nil or label == '' then label = fallback_label(id) end
    label_cache[id] = label
  end
  return label_cache[id]
end

local function strip_trailing_period(inlines)
  -- Titles get their own period in the label ("Название."), avoid "Название..".
  local last = inlines[#inlines]
  if last and last.t == 'Str' and last.text:match('[%.!?]$') then
    local copy = inlines:clone()
    local text = last.text:gsub('%.$', '')
    if text == '' then copy:remove(#copy) else copy[#copy] = pandoc.Str(text) end
    return copy, last.text:match('[!?]$') ~= nil
  end
  return inlines, false
end

---------------------------------------------------------------- numbering state (HTML)

local state = {
  appendix = false,
  numbered = true,
  secnumdepth = 3,
  counters = { 0, 0, 0, 0, 0, 0 },
  blocks = {},
}

local function section_prefix()
  local top = state.counters[1]
  if top == 0 then return nil end
  if state.appendix then return ASBUK[top] or tostring(top) end
  return tostring(top)
end

local function header_number(level)
  local parts = {}
  for i = 1, level do
    local n = state.counters[i]
    if i == 1 and state.appendix then
      parts[#parts + 1] = ASBUK[n] or tostring(n)
    else
      parts[#parts + 1] = tostring(n)
    end
  end
  return table.concat(parts, '.')
end

---------------------------------------------------------------- headers

local function handle_header(h)
  local out = {}
  local is_appendix = h.classes:includes('appendix')
  if is_appendix and not state.appendix then
    state.appendix = true
    state.counters = { 0, 0, 0, 0, 0, 0 }
    if IS_LATEX then
      out[#out + 1] = pandoc.RawBlock('latex', '\\hzappendix')
    end
  end
  if IS_HTML then
    if state.appendix then h.classes:insert('h0-appendix') end
    local numbered = state.numbered and not h.classes:includes('unnumbered')
      and h.level <= state.secnumdepth
    if numbered then
      state.counters[h.level] = state.counters[h.level] + 1
      for i = h.level + 1, #state.counters do state.counters[i] = 0 end
      if h.level == 1 then state.blocks = {} end
      local num = header_number(h.level)
      local label = num .. '.'
      if h.level == 1 and state.appendix then
        label = 'Приложение ' .. num .. '.'
      end
      h.attributes['data-number'] = num
      local content = pandoc.Inlines({
        pandoc.Span({ pandoc.Str(label) }, { class = 'header-section-number' }),
        pandoc.Space(),
      })
      content:extend(h.content)
      h.content = content
    end
  end
  out[#out + 1] = h
  return out
end

---------------------------------------------------------------- semantic divs

local process_blocks -- forward declaration

local function latex_block(div, kind, spec)
  local title = parse_inlines(div.attributes['title'])
  local head = pandoc.Inlines({ latex('\\begin{' .. spec.env .. '}') })
  if title then
    title = strip_trailing_period(title)
    head:insert(latex('[{'))
    head:extend(title)
    head:insert(latex('}]'))
  end
  if div.identifier ~= '' then
    head:insert(latex('\\label{' .. latex_label(div.identifier) .. '}'))
  end
  local content = process_blocks(div.content)
  local out = pandoc.Blocks({})
  local first = content[1]
  if first and (first.t == 'Para' or first.t == 'Plain') then
    head:extend(strip_leading_comments(first.content))
    out:insert(pandoc.Para(head))
    for i = 2, #content do out:insert(content[i]) end
  else
    out:insert(pandoc.Plain(head))
    out:extend(content)
  end
  out:insert(pandoc.RawBlock('latex', '\\end{' .. spec.env .. '}'))
  return out
end

local function html_block(div, kind, spec)
  local label = spec.label
  if spec.numbered then
    state.blocks[kind] = (state.blocks[kind] or 0) + 1
    local prefix = section_prefix()
    local num = tostring(state.blocks[kind])
    if prefix then num = prefix .. '.' .. num end
    label = label .. ' ' .. num
  end
  local title = parse_inlines(div.attributes['title'])
  div.attributes['title'] = nil
  div.classes:insert('h0-block')
  div.content = process_blocks(div.content)

  local head = pandoc.Inlines({
    pandoc.Span({ pandoc.Str(spec.style == 'bar' and label or (label .. '.')) }, { class = 'h0-label' }),
  })
  if title then
    local bare, has_mark = strip_trailing_period(title)
    local t = bare:clone()
    if spec.style == 'theorem' or spec.style == 'statement' or spec.style == 'bar' then
      t:insert(1, pandoc.Str('('))
      t:insert(pandoc.Str(')'))
    elseif not has_mark then
      t:insert(pandoc.Str('.'))
    end
    head:insert(pandoc.Space())
    head:insert(pandoc.Span(t, { class = 'h0-block-title' }))
  end

  if spec.style == 'bar' then
    div.content:insert(1, pandoc.Div({ pandoc.Plain(head) }, { class = 'h0-head' }))
  else
    local first = div.content[1]
    head:insert(pandoc.Space())
    if first and (first.t == 'Para' or first.t == 'Plain') then
      local merged = concat_inlines(head, strip_leading_comments(first.content))
      div.content[1] = pandoc.Para(merged)
    else
      div.content:insert(1, pandoc.Para(head))
    end
  end
  return div
end

local function handle_div(div)
  local kind = block_kind(div)
  if kind == nil then
    div.content = process_blocks(div.content)
    return { div }
  end
  local spec = BLOCKS[kind]
  if IS_LATEX then
    return latex_block(div, kind, spec)
  elseif IS_HTML then
    return { html_block(div, kind, spec) }
  end
  div.content = process_blocks(div.content)
  return { div }
end

---------------------------------------------------------------- raw TeX math (HTML)

local MATH_ENVS = {
  'align', 'align%*', 'aligned', 'equation', 'equation%*', 'gather', 'gather%*',
  'multline', 'multline%*', 'flalign', 'flalign%*', 'alignat', 'alignat%*',
}

local function raw_tex_math(text)
  local trimmed = text:match('^[ \t\r\n]*(.-)[ \t\r\n]*$')
  for _, env in ipairs(MATH_ENVS) do
    if trimmed:match('^\\begin{' .. env .. '}') then
      local body = trimmed
      -- equation-like envs are single displays; drop the wrapper for MathML.
      local inner = trimmed:match('^\\begin{equation%*?}(.*)\\end{equation%*?}$')
      if inner then body = inner end
      return body
    end
  end
  return nil
end

---------------------------------------------------------------- recursive walk

---------------------------------------------------------------- long code lines (LaTeX)

-- fancyvrb cannot break lines (fvextra is not always installed), so lines longer than
-- CODE_WIDTH characters are wrapped here: at the last space if possible, the continuation
-- indented like the original line plus four spaces.
local CODE_WIDTH = 90

local function wrap_code_line(line, width)
  local len = utf8.len(line)
  if len == nil or len <= width then return { line } end
  local indent = line:match('^[ \t]*')
  local cont = (#indent + 4 < width / 2) and (indent .. '    ') or ''
  local out = {}
  local rest = line
  while (utf8.len(rest) or 0) > width do
    local cut = utf8.offset(rest, width + 1) -- byte index of the first char that does not fit
    local head = rest:sub(1, cut - 1)
    local sp = head:match('.*()[ \t]')
    local min_break = (#out == 0) and #indent or #cont
    if sp and sp > min_break + 1 then
      out[#out + 1] = (rest:sub(1, sp - 1):gsub('[ \t]+$', ''))
      rest = cont .. rest:sub(sp + 1)
    else
      out[#out + 1] = head
      rest = cont .. rest:sub(cut)
    end
  end
  out[#out + 1] = rest
  return out
end

local function wrap_code(block, width)
  local lines = {}
  local changed = false
  for line in (block.text .. '\n'):gmatch('(.-)\r?\n') do
    local parts = wrap_code_line(line, width)
    if #parts > 1 then changed = true end
    for _, p in ipairs(parts) do lines[#lines + 1] = p end
  end
  if changed then block.text = table.concat(lines, '\n') end
  return block
end

-- HTML: "Рис. 1." / "Таблица 1." like the LaTeX captions (global numbering, captioned only).
local caption_counters = { figures = 0, tables = 0 }

local function number_caption(caption, counter, word)
  local long = caption and caption.long
  if long == nil or #long == 0 then return end
  local first = long[1]
  if not (first.t == 'Plain' or first.t == 'Para') or #first.content == 0 then return end
  caption_counters[counter] = caption_counters[counter] + 1
  local label = pandoc.Span(
    { pandoc.Str(word .. ' ' .. caption_counters[counter] .. '.') },
    { class = 'h0-caption-label' }
  )
  first.content:insert(1, pandoc.Space())
  first.content:insert(1, label)
  long[1] = first
  caption.long = long
end

local function process_list_items(items)
  local out = {}
  for i, item in ipairs(items) do out[i] = process_blocks(item) end
  return out
end

process_blocks = function(blocks)
  local out = pandoc.Blocks({})
  for _, b in ipairs(blocks) do
    local t = b.t
    if t == 'Header' then
      out:extend(handle_header(b))
    elseif t == 'Div' then
      out:extend(handle_div(b))
    elseif t == 'BlockQuote' then
      b.content = process_blocks(b.content)
      out:insert(b)
    elseif t == 'BulletList' or t == 'OrderedList' then
      b.content = process_list_items(b.content)
      out:insert(b)
    elseif t == 'DefinitionList' then
      for _, item in ipairs(b.content) do
        item[2] = process_list_items(item[2])
      end
      out:insert(b)
    elseif t == 'Figure' then
      b.content = process_blocks(b.content)
      if IS_HTML then number_caption(b.caption, 'figures', 'Рис.') end
      out:insert(b)
    elseif t == 'Table' then
      if IS_HTML then number_caption(b.caption, 'tables', 'Таблица') end
      out:insert(b)
    elseif is_html_comment(b) then
      -- dropped here already, so it can never become a block's first "paragraph"
    elseif t == 'CodeBlock' and IS_LATEX then
      out:insert(wrap_code(b, CODE_WIDTH))
    elseif t == 'RawBlock' and IS_HTML and (b.format == 'tex' or b.format == 'latex') then
      local math = raw_tex_math(b.text)
      if math then
        out:insert(pandoc.Para({ pandoc.Math('DisplayMath', math) }))
      else
        out:insert(b)
      end
    else
      out:insert(b)
    end
  end
  return out
end

---------------------------------------------------------------- metadata

local function units_label(units)
  if units == nil then return nil end
  if pandoc.utils.type(units) == 'table' then -- MetaMap, e.g. {pages: 2}
    local parts = {}
    for _, pair in ipairs(UNIT_LABELS) do
      local v = units[pair[1]]
      if v ~= nil then parts[#parts + 1] = stringify(v) .. ' ' .. pair[2] end
    end
    if #parts > 0 then return table.concat(parts, ', ') end
    return nil
  end
  local s = stringify(units)
  if s == '' then return nil end
  return s
end

local function normalise_sources(meta)
  local sources = meta['sources']
  if sources == nil then return end
  if pandoc.utils.type(sources) ~= 'List' then sources = { sources } end
  local out = pandoc.List({})
  for _, src in ipairs(sources) do
    local entry = {}
    if pandoc.utils.type(src) == 'table' then -- MetaMap
      if src.id then entry.id = pandoc.Inlines({ pandoc.Str(stringify(src.id)) }) end
      local title = src.title or src.name or src.origin
      if title then entry.title = title end
      local details = {}
      if src.kind then
        local k = stringify(src.kind)
        details[#details + 1] = KIND_LABELS[k] or k
      end
      local u = units_label(src.units)
      if u then details[#details + 1] = u end
      if #details > 0 then
        entry.details = pandoc.Inlines({ pandoc.Str(table.concat(details, ', ')) })
      end
    else
      entry.title = src
    end
    if entry.title == nil then entry.title = pandoc.Inlines({}) end
    if entry.id == nil then entry.id = pandoc.Inlines({}) end
    out:insert(entry)
  end
  meta['h0lon-sources'] = out
end

local function css_string(s)
  s = s:gsub('\\', '\\\\'):gsub('"', '\\"'):gsub('<', '\\3C '):gsub('\n', ' ')
  return '"' .. s .. '"'
end

local function html_running_heads(meta)
  local course = meta['course'] and stringify(meta['course']) or ''
  local title = meta['short-title'] or meta['title']
  title = title and stringify(title) or ''
  title = title:gsub('%[%[[^%]]-%]%]', ''):gsub('[ \t\r\n]+', ' ')
  local left = course ~= '' and course or title
  meta['h0lon-head-left'] = pandoc.RawInline('html', css_string(left))
  meta['h0lon-head-right'] = pandoc.RawInline('html', css_string(title))
end

-- Without a date the LaTeX title page prints \today (polyglossia, Russian).
local MONTHS_GENITIVE = {
  'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
  'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря',
}

local function today_ru()
  local t = os.date('*t')
  return string.format('%d %s %d г.', t.day, MONTHS_GENITIVE[t.month], t.year)
end

---------------------------------------------------------------- entry point

local function is_english(meta)
  local lang = meta['lang'] and stringify(meta['lang']) or 'ru'
  return lang:match('^en') ~= nil
end

function Pandoc(doc)
  local meta = doc.meta
  if meta['numbersections'] ~= nil and not meta_truthy(meta['numbersections']) then
    state.numbered = false
  end
  if meta['secnumdepth'] ~= nil then
    state.secnumdepth = tonumber(stringify(meta['secnumdepth'])) or state.secnumdepth
  end
  normalise_sources(meta)
  if IS_HTML then
    html_running_heads(meta)
    if meta['date'] == nil then meta['date'] = pandoc.MetaInlines({ pandoc.Str(today_ru()) }) end
  end

  doc.blocks = process_blocks(doc.blocks)

  if IS_HTML then
    doc = doc:walk({
      RawInline = function(r)
        if r.format == 'tex' or r.format == 'latex' then
          local math = raw_tex_math(r.text)
          if math then return pandoc.Math('DisplayMath', math) end
        end
      end,
    })
  end
  if not is_english(meta) then
    doc = doc:walk({
      Quoted = function(q)
        local open, close = '«', '»'
        if q.quotetype == 'SingleQuote' then open, close = '„', '“' end
        local out = pandoc.Inlines({ pandoc.Str(open) })
        out:extend(q.content)
        out:insert(pandoc.Str(close))
        return out
      end,
    })
  end
  doc.meta = meta
  return doc
end
