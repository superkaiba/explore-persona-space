-- Preserve literal filenames while allowing line breaks in narrow PDF tables.
function Code(element)
  if FORMAT:match('latex') and not element.text:find('[{}\\]') then
    return pandoc.RawInline('latex', '\\path{' .. element.text .. '}')
  end
end

-- Avoid leaving a new section with only one introductory line on the page.
function Header(element)
  if FORMAT:match('latex') and element.level == 1
      and pandoc.utils.stringify(element.content):match('^1%. ') then
    return {pandoc.RawBlock('latex', '\\clearpage'), element}
  end
  if FORMAT:match('latex') and element.level <= 2 then
    return {pandoc.RawBlock('latex', '\\Needspace{7\\baselineskip}'), element}
  end
end
