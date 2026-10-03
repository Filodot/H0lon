--[==[
H0lon HTML sanitiser: runs first, only for HTML output.

The master is compiled from source materials, and their content is data. The HTML engine
calls Pandoc with --embed-resources, which reads every src= it meets (raw <iframe>/<img>,
images, data-src attributes): local files anywhere on disk end up in the page and so in the
PDF, remote URLs are fetched over the network during the render. Therefore:

* raw HTML is dropped, except comments (srcrefs.lua removes them) and a few harmless tags
  without attributes (<br>, <sub>, <b>, ...). The LaTeX writer drops raw HTML anyway, so
  both engines print the same text;
* element attributes are cut down to a short whitelist: Pandoc writes unknown keys as
  data-*, and data-src would be embedded as well;
* an image must be a file inside the source directory (metadata `h0lon-resource-dir`) or a
  data:image URI; any other image is replaced by its description;
* metadata `css` is ignored (the template's stylesheet is passed on the command line).

Patterns use explicit ASCII sets (see blocks.lua).
]==]

if not FORMAT:match('html') then return {} end

local SAFE_TAGS = {
  br = true, wbr = true, sub = true, sup = true, b = true, i = true, em = true,
  strong = true, u = true, s = true, small = true, mark = true, kbd = true, var = true,
}
local SAFE_ATTRIBUTES = {
  title = true, width = true, height = true, lang = true, dir = true, startFrom = true,
}

local base = nil -- source directory, forward slashes
local dropped_raw = 0

local function warn(msg)
  if pandoc.log and pandoc.log.warn then
    pandoc.log.warn(msg)
  else
    io.stderr:write('[WARNING] ' .. msg .. '\n')
  end
end

local function ascii_lower(s)
  return (s:gsub('[A-Z]', function(c) return string.char(c:byte() + 32) end))
end

local function is_comment(text)
  return text:match('^[ \t\r\n]*<!%-%-.*%-%->[ \t\r\n]*$') ~= nil
end

local function is_safe_tag(text)
  local name = text:match('^<(/?[A-Za-z]+)[ \t]*/?>$')
  if name == nil then return false end
  return SAFE_TAGS[ascii_lower(name:gsub('^/', ''))] == true
end

local function is_html(format)
  return format == 'html' or format == 'html5' or format == 'html4'
end

local function raw(el)
  if not is_html(el.format) or is_comment(el.text) or is_safe_tag(el.text) then return nil end
  dropped_raw = dropped_raw + 1
  return {}
end

local function clean_attributes(el)
  local attrs = el.attributes
  if attrs == nil then return nil end
  local changed = false
  for key, _ in pairs(attrs) do
    if not SAFE_ATTRIBUTES[key] then
      attrs[key] = nil
      changed = true
    end
  end
  if changed then
    el.attributes = attrs
    return el
  end
  return nil
end

---------------------------------------------------------------- image sources

local function percent_decode(s)
  return (s:gsub('%%([0-9A-Fa-f][0-9A-Fa-f])', function(h) return string.char(tonumber(h, 16)) end))
end

-- Path segments with "." and ".." resolved; nil if ".." climbs above the start.
local function segments(path)
  local out = {}
  for seg in path:gmatch('[^/]+') do
    if seg == '..' then
      if #out == 0 then return nil end
      table.remove(out)
    elseif seg ~= '.' then
      out[#out + 1] = seg
    end
  end
  return out
end

local function inside(path_segs, base_segs)
  if #path_segs <= #base_segs then return false end
  for i, seg in ipairs(base_segs) do
    if ascii_lower(path_segs[i]) ~= ascii_lower(seg) then return false end
  end
  return true
end

local function image_allowed(src)
  if src:match('^data:image/') then return true end
  local path = percent_decode(src):gsub('\\', '/')
  if path:match('^[A-Za-z]:/') or path:match('^/') then
    if base == nil then return false end
    local p, b = segments(path), segments(base)
    return p ~= nil and b ~= nil and inside(p, b)
  end
  if path:match('^[A-Za-z][A-Za-z0-9+.-]*:') then return false end -- http:, file:, data:...
  return path ~= '' and segments(path) ~= nil
end

local function image(img)
  if image_allowed(img.src) then return clean_attributes(img) end
  warn('HTML-рендер: изображение «' .. img.src .. '» пропущено — допускаются только файлы '
    .. 'из каталога исходника')
  local alt = img.caption
  if alt == nil or #alt == 0 then alt = { pandoc.Str('[изображение пропущено]') } end
  return pandoc.Span(alt, { class = 'h0-image-blocked' })
end

---------------------------------------------------------------- entry point

function Pandoc(doc)
  local dir = doc.meta['h0lon-resource-dir']
  if dir ~= nil then
    base = pandoc.utils.stringify(dir):gsub('\\', '/')
    if base == '' then base = nil end
  end
  doc.meta['css'] = nil
  doc = doc:walk({
    RawBlock = raw,
    RawInline = raw,
    Image = image,
    Div = clean_attributes,
    Span = clean_attributes,
    Header = clean_attributes,
    CodeBlock = clean_attributes,
    Code = clean_attributes,
    Link = clean_attributes,
    Table = clean_attributes,
    Figure = clean_attributes,
  })
  if dropped_raw > 0 then
    warn('HTML-рендер: пропущено фрагментов сырого HTML из мастер-конспекта: '
      .. dropped_raw .. ' (в PDF они не выводятся ни одним движком)')
  end
  return doc
end
