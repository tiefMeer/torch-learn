
import jieba

# 文本预处理, 将输入一段文本分词后清除异常字符, 返回词iter
def pretreat(text):
    pattern = re.compile("[\u4e00-\u9fff0-9a-zA-Z]+")
    for word in jieba.cut(text):
        if pattern.fullmatch(word):
            yield word
            
