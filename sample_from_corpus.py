import json
import os.path
import random
import re
from tqdm import tqdm
from collections import defaultdict

class CorpusManager:
    def __init__(self, library_path_list):
        self.root = os.path.dirname(__file__)
        """
        :library_path_list: 语料库名称列表
        """
        self.library_lang_map = {
            'THUCNews': 'zh',
            'chinese-table-level0-800': 'zh',
            'chinese-table-level1-3500': 'zh',
            'chinese-table-level2-3000': 'zh',
            'chinese-table-level3-1605': 'zh',
            'english-table-level1-3870': 'en',
            'english-table-level2-4M': 'en',
            'chinese-chatgpt': 'zh',
            'english-wiki': 'en',
            'english-wiki_part_1': 'en',
            'english-wiki_part_2': 'en',
            'english-wiki_part_3': 'en',
            'english-wiki_part_4': 'en',
            'english-wiki_part_5': 'en',
            'english-wiki_part_6': 'en',
            'english-wiki_part_7': 'en',
            'english-wiki_part_8': 'en',
            'english-wiki_part_9': 'en',
            'english-wiki_part_10': 'en',

            'chinese-news': "zh",
            'chinese-laws': "zh",
            'chinese-novel': "zh",

            'vocab_zh': 'zh',
            'vocab_en': 'en',
            'chinese-xinhua': 'zh',
            'vocab_zh_added_8105': 'zh',

            "emoji-seguiemj-1.45-3d": "emoji"
            
        }
        self.whitespaces = [
            '\\t',
            '\\n',
            # 普通空格及衍生空格
            '\\u0020',  # 标准半角空格
            '\\u00A0',  # 不间断空格
            '\\u2002',  # En空格
            '\\u2003',  # Em空格
            '\\u2004',  # 三分之一Em空格
            '\\u2005',  # 四分之一Em空格
            '\\u2006',  # 六分之一Em空格
            '\\u2007',  # 数字空格
            '\\u2008',  # 标点空格
            '\\u2009',  # 窄空格
            '\\u200A',  # 头发空格
            '\\u3000',  # 全角空格（中文宽空格）
            
            # 零宽空白字符
            '\\u200B',  # 零宽空格
            '\\u200C',  # 零宽不连字
            '\\u200D',  # 零宽连字
            
            # 行/段落分隔符
            '\\u2028',  # 行分隔符
            '\\u2029',  # 段落分隔符
            
            # 控制字符类空白
            '\\u0009',  # 水平制表符（\t）
            '\\u000A',  # 换行符（\n）
            '\\u000D'   # 回车符（\r）
        ]

        self.corpora = {}  # 存储所有语料库的数据

        self.traversal_state = defaultdict(lambda: {
            'current_idx': 0,    # 当前处理到的行索引
            'is_completed': False # 是否遍历完该库
        })
        self.global_completed = False
        
        # 加载常用中文字符
        # with open(os.path.join(self.root, 'source/chinese-table-level0-800.jsonl'), 'r', encoding='utf-8') as f:
        #     self.Commonly_Used_Chinese_Characters = list(json.load(f).get('content', ''))

        # with open(os.path.join(self.root, 'source/chinese-table-level1-3500.jsonl'), 'r', encoding='utf-8') as f:
        #     # 逐行读取JSONL文件，解析每个对象的content字段，收集所有字符
        #     self.Commonly_Used_Chinese_Characters = []
        #     for line in f:
        #         line = line.strip()
        #         if not line:  # 跳过空行（如果有）
        #             continue
        #         data = json.loads(line)
        #         char = data.get('content', '').strip()
        #         if char:  # 确保只添加非空字符
        #             self.Commonly_Used_Chinese_Characters.append(char)

        for library_path in library_path_list:
            library_name = library_path.split("/")[-1].split(".")[0]
            if library_name not in self.library_lang_map:
                raise ValueError(f"未登记的语料库: {library_name}")

            self.filepath = os.path.join(self.root, f'source/{library_name}.jsonl')
            if not os.path.exists(self.filepath):
                raise ValueError(f"文件不存在: {self.filepath}")

            self.language_type = self.library_lang_map[library_name]
            self.corpus = self._load_corpus()
            self.corpora[library_name] = self.corpus  # 将当前语料库的数据存储到字典中

    def _load_corpus(self):
        """加载语料"""
        corpus = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            # for i, line in enumerate(tqdm(f, desc=self.filepath)):
            lines = f.read().splitlines()
            for i, line in enumerate(tqdm(lines, desc=self.filepath)):
                try:
                    data = json.loads(line)
                    if "content" in data:
                        corpus.append(data["content"])
                except json.JSONDecodeError:
                    continue
        return corpus

    def _is_chinese(self, text):
        """判断是否为中文文本（包含中文字符则认为是中文）"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))


    def reset_traversal(self, library_name=None):
        """
        重置遍历状态（支持单库/全库重置）
        :param library_name: 可选，指定重置的库名，默认重置所有
        """
        # 重置全局完成标记
        self.global_completed = False
        if library_name:
            if library_name in self.traversal_state:
                self.traversal_state[library_name] = {
                    'current_idx': 0,
                    'is_completed': False
                }
        else:
            self.traversal_state.clear()

    def sample_traverse(self, length):
        """
        有序遍历语料库，每次返回完整的一行语料（忽略length参数）
        遍历完所有语料库一次后永久结束，不再自动重置
        :param length: 兼容外部调用，实际无作用
        :return: 语料库中的完整一行文本
        """

        # 1. 如果全局已遍历完成，直接抛出异常（不再重置）
        if self.global_completed:
            raise StopIteration("所有语料库已遍历完毕，遍历结束（如需重新遍历请调用 reset_traversal() 方法）")

        # 2. 筛选未遍历完成的语料库
        available_libraries = [
            name for name in self.corpora.keys()
            if not self.traversal_state[name]['is_completed']
        ]

        # 3. 所有库都遍历完，标记全局完成并抛出异常
        if not available_libraries:
            self.global_completed = True
            raise StopIteration("所有语料库已遍历完毕，遍历结束（如需重新遍历请调用 reset_traversal() 方法）")

        # 4. 选择第一个未完成的库（保持选库优先级）
        library_name = available_libraries[0]
        corpus = self.corpora[library_name]
        state = self.traversal_state[library_name]

        if not corpus:
            raise ValueError(f"语料库 {library_name} 为空")

        # 5. 获取当前行并更新遍历状态
        current_idx = state['current_idx']
        # 返回当前行的完整文本
        result = corpus[current_idx]
        
        # 6. 更新索引，判断是否遍历完该库
        state['current_idx'] += 1
        if state['current_idx'] >= len(corpus):
            state['is_completed'] = True

        return result

    def advance_current_library_index(self):
        """
        手动前移当前正在遍历的库的索引，跳过当前无效文本
        适配原有 traversal_state 结构，保持遍历逻辑一致性
        """
        # 1. 若全局已完成，直接返回（无操作）
        if self.global_completed:
            return
        
        # 2. 筛选未遍历完成的语料库（和 sample_traverse 逻辑一致）
        available_libraries = [
            name for name in self.corpora.keys()
            if not self.traversal_state[name]['is_completed']
        ]
        
        if not available_libraries:
            self.global_completed = True
            return
        
        # 3. 选择当前正在遍历的库（和 sample_traverse 选库优先级一致）
        library_name = available_libraries[0]
        corpus = self.corpora[library_name]
        state = self.traversal_state[library_name]

        if not corpus:
            return
        
        # 4. 手动前移索引，更新库完成状态（核心：跳过当前无效文本）
        state['current_idx'] += 1
        if state['current_idx'] >= len(corpus):
            state['is_completed'] = True

    def sample(self, length):
        """
        随机选择一个语料库，并随机截取一段语料
        中文 -> 按字数
        英文 -> 按单词数
        """
        if length == 0:
            return ""
        library_name = random.choice(list(self.corpora.keys()))
        corpus = self.corpora[library_name]
        language_type = self.library_lang_map[library_name]

        if not corpus:
            raise ValueError("语料库为空")

        def get_one():
            return random.choice(corpus)

        if language_type == 'zh':
            result = ""
            while len(result) < length:
                text = get_one()
                text = self.correct_text(text)
                remain = length - len(result)
                if len(text) <= remain:
                    if self.is_valid_start(text[0]):
                        result += text
                else:
                    start = random.randint(0, len(text) - remain)
                    retry_num = 0
                    while not self.is_valid_start(text[start]) and retry_num < 10:
                        start = random.randint(0, len(text) - remain)
                        retry_num += 1
                    result += text[start:start + remain]
            return result

        elif language_type == 'en':
            result_words = []
            while (len(" ".join(result_words)) < length):
                text = get_one()
                text = self.correct_text(text).strip()
                if self.is_valid_start(text[0]):
                    while '  ' in text:
                        text = text.replace('  ',' ')
                    words = text.split()
                    result_words.extend(words)
            return " ".join(result_words)[:length].strip()

            # while len(result_words) < length:
            #     text = get_one()
            #     words = text.split()
            #     remain = length - len(result_words)
            #     if len(words) <= remain:
            #         result_words.extend(words)
            #     else:
            #         start = random.randint(0, len(words) - remain)
            #         result_words.extend(words[start:start + remain])
            # return " ".join(result_words)

    def correct_text(self, text):
        for w in self.whitespaces:
            text = text.replace(w, ' ')
        return text

    def is_valid_start(self, char):
        return bool(re.match(r'[\u4e00-\u9fffA-Za-z ]', char))

    # def sample_simple_chinese(self, length):
    #     """
    #     随机选择一个中文语料库，并随机截取一段语料
    #     中文 -> 按字数
    #     """

    #     library_name = random.choice(list(self.corpora.keys()))
    #     corpus = self.corpora[library_name]

    #     if not corpus:
    #         raise ValueError("语料库为空")

    #     result = []
    #     remaining_length = length


    #     while remaining_length > 0:
    #         text = random.choice(corpus)
    #         text = self.correct_text(text)

    #         # 如果里面有不在 Commonly_Used_Chinese_Characters 的字符，就跳过重新采样
    #         text = ''.join([char for char in text if (not self._is_chinese(char) or char in self.Commonly_Used_Chinese_Characters)])

    #         if len(text) <= remaining_length:
    #             if self.is_valid_start(text[0]):
    #                 result.append(text)
    #                 remaining_length -= len(text)
    #         else:
    #             start = random.randint(0, len(text) - remaining_length)
    #             # 保证截取的起点是合法的
    #             while not self.is_valid_start(text[start]):
    #                 start = random.randint(0, len(text) - remaining_length)
    #             result.append(text[start:start + remaining_length])
    #             remaining_length = 0

    #     return ''.join(result).strip()


if __name__ == "__main__":
    # 设置一个使用全部语料库
    # 设置一个使用全部中文语料库
    # cm_zh = CorpusManager(["chinese-table-level1-3500", "chinese-table-level2-3000", "chinese-table-level3-1605"])
    # for i in range(100):
    #     print(f'length={i} | {cm_zh.sample(i)}')
    # print('=====================================================')
    # cm_zh = CorpusManager(["THUCNews", "chinese-chatgpt"])
    cm_zh = CorpusManager(["vocab_zh"])
    for i in range(100):
        print(f'length={i} | {cm_zh.sample(i)}')
    print('=====================================================')
    for i in range(100):
        print(f'length={i} | {cm_zh.sample_simple_chinese(i)}')
    print('=====================================================')

    # # 设置一个使用全部英文语料库
    # cm_en = CorpusManager(["english-table-level1-3870", "english-table-level2-4M"])
    # for i in range(100):
    #     print(f'length={i} | {cm_en.sample(i)}')
    # cm_en = CorpusManager(["english-wiki"])
    # for i in range(100):
    #     print(f'length={i} | {cm_en.sample(i)}')