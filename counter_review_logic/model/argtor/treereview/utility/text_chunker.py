import re
from nltk.tokenize import sent_tokenize
from ..utility.utils import count_token

class TextChunker:
    def __init__(self, model_name="gpt-4o", encoding_name="o200k_base"):
        self.model_name = model_name
        self.encoding_name = encoding_name

    def add_section_info_to_paragraphs(self, full_text):
        section_pattern = re.compile(r'^(#{1,6})\s+(.*)', re.MULTILINE)
        paragraphs = full_text.split('\n\n')
        current_section = []
        para_idx = 0
        for i, paragraph in enumerate(paragraphs):
            match = section_pattern.match(paragraph)
            if match:
                level = len(match.group(1))
                if level == 1:
                    continue
                title = match.group(2)
                while current_section and current_section[-1][0] >= level:
                    current_section.pop()
                current_section.append((level, title))
                para_idx = 0
                continue
            para_idx += 1
            if current_section:
                section_info = "Section: " + " > ".join([title for _, title in current_section])
                paragraph_info = f"Paragraph: {para_idx}"
                paragraphs[i] = f"{section_info}, {paragraph_info}\n{paragraph}"
        new_markdown_text = '\n\n'.join(paragraphs)
        return new_markdown_text

    def chunk(self, full_text: str, chunk_size: int=1024, keep_paragraph: bool=True,
                       add_section_info: bool=True, allow_overlap: bool=False, overlap_sent_count: int=1) -> list[str]:
        if add_section_info:
            full_text = self.add_section_info_to_paragraphs(full_text)
        chunks = []
        paragraphs = full_text.split("\n\n")
        cur_chunk = []
        cur_token_count = 0

        if keep_paragraph:
            for para in paragraphs:
                para_tokens = count_token(para) + 2
                if cur_token_count + para_tokens > chunk_size:
                    if cur_chunk:
                        chunks.append("\n\n".join(cur_chunk))
                        cur_chunk = []
                        cur_token_count = 0
                cur_chunk.append(para)
                cur_token_count += para_tokens
            if cur_chunk:
                chunks.append("\n\n".join(cur_chunk))
        else:
            for para in paragraphs:
                sentences = sent_tokenize(para)
                for sent in sentences:
                    sent_tokens = count_token(sent) + 1
                    if cur_token_count + sent_tokens > chunk_size:
                        if cur_chunk:
                            chunks.append(" ".join(cur_chunk))
                            if allow_overlap:
                                cur_chunk = cur_chunk[-overlap_sent_count:]
                                cur_token_count = sum(count_token(s) for s in cur_chunk) + len(cur_chunk)
                            else:
                                cur_chunk = []
                                cur_token_count = 0
                    cur_chunk.append(sent)
                    cur_token_count += sent_tokens
                if cur_chunk:
                    cur_chunk[-1] += "\n\n"
                    cur_token_count += 2
            if cur_chunk:
                chunks.append(" ".join(cur_chunk))
        return chunks

