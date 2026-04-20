-- video-credit: Quarto shortcode for embedding a video with accessible alt-text.
--
-- Usage:
--   {{< video-credit src="assets/my-video.mp4" alt="Description of the video" >}}
--   {{< video-credit src="assets/my-video.mp4" alt="Description" width="80%" >}}
--
-- Parameters (all keyword):
--   src    (required) – path or URL to the .mp4 file
--   alt    (optional) – accessible description used as aria-label and fallback text
--   width  (optional, default "100%") – CSS width value for the <video> element

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
  ["video-credit"] = function(args, kwargs, meta, env)
    local src   = kwarg(kwargs, "src",   "")
    local alt   = kwarg(kwargs, "alt",   "")
    local width = kwarg(kwargs, "width", "100%")

    if src == "" then
      quarto.log.warning("video-credit: 'src' parameter is required but was not provided.")
    end

    local html = '<video controls autoplay muted loop playsinline'
      .. ' width="' .. html_escape(width) .. '"'
      .. ' aria-label="' .. html_escape(alt) .. '">\n'
      .. '  <source src="' .. html_escape(src) .. '" type="video/mp4">\n'
      .. '  ' .. html_escape(alt) .. '\n'
      .. '</video>'

    return pandoc.RawBlock("html", html)
  end
}
