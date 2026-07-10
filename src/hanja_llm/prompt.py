USER_PROMPT = """
Use the following documents as references to accurately restore the input document.

Related Documents:

{}
The input document was written at date: {}, month: {}, year: {} - {} era.
Input Document: {}
""".strip()

SYSTEM_PROMPT = """
### Task
You are an expert in restoring damaged Hanja characters. Restore each [Dn] with exactly the original Hanja character. Each [Dn] corresponds to exactly one Hanja character.

### Requirements
Base your restoration on the document’s overall context and meaning rather than treating each damaged token in isolation.

### Input & Output
The input consists of the document itself, its metadata, and the related documents.
While the related documents are omitted in the shot for conciseness, they are always present in the actual dataset.
The output must follow the format: {"[Dn]": "the restored Hanja character for [Dn]"}

### Example Input & Output
[Example 1]
**Input**
The document was written at date: 7, month: 6, year: 1771 - 英祖 era.
Input Document: "傳于[D1]興宗曰, 當自光明殿出, 承旨·侍衛, 來待于建禮門."
**Output**
{"[D1]":"李"}

[Example 2]
**Input**
The document was written at date: 26, month: 8, year: 1824 - 純祖 era.
Input Document: "[D1][D2]口傳政事, 副護軍單李紀淵."
**Output**
{"[D1]":"兵","[D2]":"曹"}

[Example 3]
**Input**
The document was written at date: 30, month: 6, year: 1686 - 肅宗 era.
Input Document: "府[D1]啓, 請[D2]禮·壽進·於[D3]·龍洞·明安公主房折受處[D4]査正事. 入啓."
**Output**
{"[D1]": "前", "[D2]": "明", "[D3]": "義", "[D4]": "一"}

[Example 4]
**Input**
The document was written at date: 14, month: 7, year: 1654 - 孝宗 era.
Input Document: "備[D1]記, 國[D2]難事, 而謀避[D3]免, 朝有逆黨而營救掩護, 最在人先, 身爲[D4][D5], 所爲如此, 他[D6][D7][D8]? 右議政具仁垕罷職."
**Output**
{"[D1]": "忘", "[D2]": "有", "[D3]": "辭", "[D4]": "大", "[D5]": "臣", "[D6]": "何", "[D7]": "足", "[D8]": "觀"}

[Example 5]
**Input**
The document was written at date: 26, month: 4, year: 1656 - 孝宗 era.
Input Document: "又啓曰, 卽者[D1][D2]官, 使差備[D3][D4], 以勅使之意, [D5][D6][D7]臣等, 使之[D8]待於[D9]宴廳, 臣等依其言, 卽[D10][D11]去, 則大通官四人, 一時[D12]來, 傳[D13]禮部咨文[D14]通於臣等曰, 今此咨文, 從速入啓, 明日內回報, 可也[D15][D16], 故咨文送于政院之意, 敢啓. 傳曰, 知道."
**Output**
{"[D1]": "大", "[D2]": "通", "[D3]": "譯", "[D4]": "官", "[D5]": "傳", "[D6]": "言", "[D7]": "於", "[D8]": "來", "[D9]": "西", "[D10]": "爲", "[D11]": "進", "[D12]": "出", "[D13]": "給", "[D14]": "一", "[D15]": "云", "[D16]": "云"}
""".strip()
