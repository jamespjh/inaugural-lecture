-- image-credit: Quarto shortcode for embedding an image with alt-text and optional image credit.
--
-- Usage:
--   {{< image-credit src="assets/photo.png" alt="Alt text" >}}
--   {{< image-credit src="https://example.com/photo.jpg"
--                    alt="A BBC Micro"
--                    credit="The National Museum of Computing"
--                    credit-url="https://www.tnmoc.org/" >}}
--
-- Parameters (all keyword):
--   src        (required) – path or URL to the image file
--   alt        (optional) – accessible description / alt text
--   credit     (optional) – human-readable credit string (e.g. author or organisation)
--   credit-url (optional) – URL to link the credit text to
--   width      (optional) – CSS width value (e.g. "50%", "300px")

-- Return the string value of a shortcode kwarg, or a default when absent/empty.
local function kwarg(kwargs, key, default)
  local v = kwargs[key]
  if v == nil then return default end
  local s = pandoc.utils.stringify(v)
  return s ~= "" and s or default
end

-- Escape special HTML characters to prevent injection in attribute values and text.
local function html_escape(s)
  s = s:gsub("&", "&amp;")
  s = s:gsub("<", "&lt;")
  s = s:gsub(">", "&gt;")
  s = s:gsub('"', "&quot;")
  return s
end

return {
  ["image-credit"] = function(args, kwargs, meta, env)
    local src        = kwarg(kwargs, "src",        "")
    local alt        = kwarg(kwargs, "alt",        "")
    local credit     = kwarg(kwargs, "credit",     nil)
    local credit_url = kwarg(kwargs, "credit-url", nil)
    local width      = kwarg(kwargs, "width",      nil)

    if src == "" then
      quarto.log.warning("image-credit: 'src' parameter is required but was not provided.")
    end

    -- Build the image element
    local width_attr = width and (' style="width:' .. html_escape(width) .. '"') or ""
    local img_html = '<img src="' .. html_escape(src) .. '" alt="' .. html_escape(alt) .. '"' .. width_attr .. '>'

    -- Build optional credit line
    local credit_html = ""
    if credit then
      if credit_url then
        credit_html = '\n<span class="image-credit">Image credit: <a href="'
          .. html_escape(credit_url) .. '">' .. html_escape(credit) .. '</a>.</span>'
      else
        credit_html = '\n<span class="image-credit">Image credit: ' .. html_escape(credit) .. '.</span>'
      end
    end

    return pandoc.RawBlock("html", img_html .. credit_html)
  end
}
