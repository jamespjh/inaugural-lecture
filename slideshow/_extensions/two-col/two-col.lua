-- two-col: Quarto filter for a shorthand two-column layout using the {.two-col} class.
--
-- Usage:
--   :::: {.two-col}
--   ::: {}
--   Left column content here.
--   :::
--   ::: {}
--   Right column content here.
--   :::
--   ::::
--
-- By default the left column is 55 % wide and the right column 45 %.
-- Override with left-width and right-width attributes:
--
--   :::: {.two-col left-width="60%" right-width="40%"}
--   ...
--   ::::
--
-- The filter transforms the {.two-col} div into Quarto's native {.columns} /
-- {.column} structure that reveal.js renders as side-by-side columns.

function Div(div)
  if not div.classes:includes("two-col") then
    return nil
  end

  -- Collect the immediate child Div blocks (skip bare paragraphs / spaces).
  local col_blocks = {}
  for _, block in ipairs(div.content) do
    if block.tag == "Div" then
      table.insert(col_blocks, block)
    end
  end

  if #col_blocks ~= 2 then
    quarto.log.warning(
      "two-col: expected exactly 2 child divs, found " .. #col_blocks .. ". Leaving unchanged."
    )
    return nil
  end

  local left_width  = div.attributes["left-width"]  or "55%"
  local right_width = div.attributes["right-width"] or "45%"

  local left  = col_blocks[1]
  local right = col_blocks[2]

  left.classes:insert("column")
  left.attributes["width"] = left_width

  right.classes:insert("column")
  right.attributes["width"] = right_width

  return pandoc.Div(
    pandoc.Blocks({ left, right }),
    pandoc.Attr("", { "columns" }, {})
  )
end
