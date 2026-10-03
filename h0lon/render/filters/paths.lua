--[==[
H0lon image-path filter.

Relative image paths are resolved against the directory of the source document
(metadata `h0lon-resource-dir`, set by the render pipeline). The result is absolute and
uses forward slashes, which is what XeLaTeX on Windows expects. URLs, data: URIs and
absolute paths are left untouched. Patterns use explicit ASCII sets (see blocks.lua).
]==]

local base = nil

local function is_url(src)
  return src:match('^[A-Za-z][A-Za-z0-9+.-]*://') ~= nil or src:match('^data:') ~= nil
end

local function is_absolute(src)
  return src:match('^[A-Za-z]:[/\\]') ~= nil or src:match('^[/\\]') ~= nil
end

local function resolve(src)
  if base == nil or src == '' or is_url(src) or is_absolute(src) then return nil end
  local path = pandoc.path.normalize(pandoc.path.join({ base, src }))
  return (path:gsub('\\', '/'))
end

function Pandoc(doc)
  local dir = doc.meta['h0lon-resource-dir']
  if dir == nil then return nil end
  base = pandoc.utils.stringify(dir)
  if base == '' then return nil end
  return doc:walk({
    Image = function(img)
      local resolved = resolve(img.src)
      if resolved then
        img.src = resolved
        return img
      end
    end,
  })
end
