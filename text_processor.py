import re
import unicodedata

class TextProcessor:
    def remove_markdown(self, text):
        '''
        :param text:
        :return: text without markdown elements
        mục đích: tránh lỗi hiển thị khi viết từng phần tử của đoạn văn
        '''
        if not isinstance(text, str):
            return ''
        # bold và italic
        text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)
        # strikethrough
        text = re.sub(r'~~(.*?)~~', r'\1', text)
        # headers
        text = re.sub(r'^#{1,6}\s', '', text, flags=re.MULTILINE)
        # links
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        # blockquotes
        text = re.sub(r'^\s*>\s', '', text, flags=re.MULTILINE)
        # horizontal rules
        text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        # list markers
        text = re.sub(r'^\s*[-*+]\s', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s', '', text, flags=re.MULTILINE)
        return text.strip()

    def get_end_tokens(self):
        try:
            with open('end_tokens.txt', 'r', encoding='utf-8') as file:
                end_tokens = set(token.strip() for token in file if token.strip())
            return end_tokens
        except FileNotFoundError:
            print(
                f"Không tìm thấy file.")
            return set()

    def remove_stopwords(self, query):
        '''
        :param query:
        :return: query sau khi đã loại bỏ các từ để hỏi như là gì, thế nào...
        mục đích: Dễ tìm kiếm tương đồng hơn cho RAG
        '''
        return query

    def remove_accents(self, input_str):
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

    def format_context(self, context):
        parts = context.split("SEPARATED")
        if len(parts) != 2:
            return context

        formatted_parts = []
        for part in parts:
            formatted_part = re.sub(r'(\d+\.\d*(?:\.\d+)*\s+[A-Z\s]+)', r'<br><br>\1', part)
            formatted_parts.append(formatted_part.strip())

        return "<br><br>---------------------------------Context khác---------------------------------<br><br>".join(formatted_parts)

