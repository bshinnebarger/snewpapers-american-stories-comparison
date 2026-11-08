We are comparing two systems that each try to segment historical newspaper pages into components.

One component is one printed item: an article, a notice, an advertisement, a poem, a list, and so on. A story continued down or across columns is still one component. The text comes from OCR, so it may contain misread words; ignore them.

You will be given the newspaper, the date and the page, then the text of a component, then a text printed next to it on the page, which one system includes in the component and the other does not. Text printed next to a component may be part of it, or may end the item before it or begin the item after it.

Segmentation systems occasionally leave out what they judge to be page furniture: a newspaper's title or dateline, a tagline, section titles printed over several items, a jump line such as "(Continued on page 3)", and the like. Judge by the overall content of the component, not by tiny fragments like these. If the text in question is only such a line or two, answer "furniture": it should not decide which system is right.

Does the text belong to this component? Answer "belongs" if it is part of the same printed item as the component, "separate" if it is part of another item or an item of its own, or "furniture" as above. Give a short reason, one or two sentences.

Answer with JSON only, matching this schema:
{"type": "object", "properties": {"verdict": {"type": "string", "enum": ["belongs", "separate", "furniture"]}, "reason": {"type": "string"}}, "required": ["verdict", "reason"], "additionalProperties": false}
