We are comparing two systems that each try to segment historical newspaper pages into components.

One component is one printed item: an article, a notice, an advertisement, a poem, a list, and so on. A story continued down or across columns is still one component. The text comes from OCR, so it may contain misread words; ignore them.

One system marked the text below as a region of the page; the other left this part of the page out. You will be given the newspaper, the date and the page, then the text.

What kind of text is it? Answer "advertisement" if it is an advertisement (or several), "terms" if it is subscription terms or a colophon, "furniture" if it is page furniture such as a masthead or section heading, or "other" if it is any other kind of printed item (a notice is not an advertisement). Give a short reason, one or two sentences.

Answer with JSON only, matching this schema:
{"type": "object", "properties": {"kind": {"type": "string", "enum": ["advertisement", "terms", "furniture", "other"]}, "reason": {"type": "string"}}, "required": ["kind", "reason"], "additionalProperties": false}
