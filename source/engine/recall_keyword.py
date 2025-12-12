import re
from loguru import logger
from typing import Tuple, List, Dict
import jieba
from rank_bm25 import BM25Okapi  # 导入BM25工具

BAD_WORDS = [
    "的", "地", "得", "着", "了", "过", "吧", "吗", "呢", "啊", "呀", "嘛", "呗", "哦", "呵", "哈",
    "在", "于", "为", "对", "对于", "关于", "和", "与", "或", "及", "而", "且", "但", "却", "因", "因为",
    "所以", "如果", "假如", "要是", "即使", "虽然", "尽管", "既然", "只要", "只有", "除非", "无论",
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们", "咱", "咱们", "自己", "自个儿",
    "这", "那", "此", "彼", "其", "该", "每", "各", "某", "谁", "什么", "哪", "哪些", "哪儿", "哪里",
    "怎么", "怎么样", "怎样", "多少", "几", "数", "许多", "很多", "一些", "任何", "所有", "全部",
    "很", "非常", "特别", "太", "更", "最", "还", "也", "都", "只", "就", "才", "又", "再", "已", "曾",
    "将", "要", "会", "能", "可以", "可", "能够", "应", "应该", "需", "需要", "得", "必须", "务必",
    "个", "只", "条", "本", "把", "张", "件", "位", "名", "群", "批", "类", "种", "样", "点", "些",
    "是", "有", "存在", "无", "没有", "出现", "发生", "进行", "做", "搞", "弄", "办", "干", "实施",
    "执行", "开展", "采取", "通过", "利用", "使用", "根据", "依据", "按照", "遵循", "依照", "基于",
    "当", "每当", "随着", "由于", "由", "被", "给", "让", "使", "叫", "令", "派", "请", "帮", "替",
    "向", "往", "朝", "沿", "顺", "从", "自从", "打", "到", "至", "直到", "为止", "比", "跟", "同",
    "与", "及", "以及", "并", "并且", "而", "而且", "或", "或者", "要么", "不但", "不仅", "不光",
    "就是", "而是", "然而", "否则", "何况", "况且", "甚至", "乃至", "尤其", "特别", "格外", "更加",
    "略微", "稍微", "几乎", "差不多", "将近", "大约", "大概", "总共", "共计", "累计", "合计",
    "之一", "之二", "例如", "比如", "譬如", "像", "好像", "仿佛", "似乎", "如同", "好比", "不如",
    "不及", "胜过", "超过", "低于", "高于", "等于", "等于", "属于", "包含", "包括", "含有", "具有",
    "具备", "拥有", "缺乏", "缺少", "不足", "够", "足够", "满足", "达到", "实现", "完成", "结束",
    "开始", "起始", "起初", "终于", "最终", "最后", "首先", "其次", "然后", "接着", "后来", "之后",
    "以前", "以后", "目前", "当前", "现在", "如今", "过去", "将来", "未来", "曾经", "一度", "屡次",
    "再三", "多次", "偶尔", "偶然", "经常", "时常", "往往", "通常", "一般", "总是", "始终", "一直",
    "仍然", "依然", "依旧", "照旧", "照样", "果然", "果真", "居然", "竟然", "幸亏", "幸好", "幸而",
    "偏偏", "反倒", "反而", "反正", "横竖", "好歹", "毕竟", "到底", "究竟", "居然", "简直", "几乎",
    "索性", "干脆", "特意", "故意", "有意", "无意", "真心", "诚心", "特意", "特地", "专门", "亲自",
    "一起", "一同", "共同", "互相", "相互", "彼此", "分别", "各自", "独自", "单独", "私下", "暗中",
    "公开", "公然", "明显", "显然", "分明", "明明", "实在", "确实", "的确", "真的", "果真", "果然",
    # 标点及特殊符号
    "，", "。", "！", "？", "；", "：", "”", "“", "’", "‘", "（", "）", "【", "】", "{", "}", "《", "》",
    "、", "…", "—", "-", "=", "+", "*", "&", "^", "#", "@", "!", "~", "`", "|", "\\", "/",
]

def keyword_recall_relevant_paragraphs(
    document: str,
    query: str,
    config: dict,
) -> Tuple[List[Dict[str, str | float]], Dict]:
    """Recall top-k relevant paragraphs"""
    # 1. 段落分割
    logger.info("Splitting document with regex: \\n\\s*\\n")
    paragraphs = re.split(r"\n\s*\n", document)
    paragraphs = [para.strip() for para in paragraphs if para.strip()]

    if not paragraphs:
        logger.warning("No valid paragraphs found in document")
        return [], {"num_paragraphs": 0, "paragraphs": []}
    logger.info(f"Split document into {len(paragraphs)} valid paragraphs")

    # 2. 配置加载
    badwords = BAD_WORDS
    model = config.get("model", "bm25")
    topk = config.get("topk", 5)
    logger.info(f"Using {model} model, top-k={topk}")

    # 3. 分词函数（中文分词+停用词过滤）
    def tokenize(text: str) -> List[str]:
        tokens = jieba.lcut(text)
        return [
            token for token in tokens
            if token.strip() and token not in badwords and any(char.isalnum() for char in token)
        ]

    # 4. Query分词
    query_tokens = tokenize(query)
    query_tokens.extend(['<table>', '$', '</table>', '%', '.'])  # 表格和公式提升权重
    if not query_tokens:
        final_scores = [0.0] * len(paragraphs)
    else:
        # 5. 段落分词（为rank_bm25准备输入）
        para_tokenized = [tokenize(para) for para in paragraphs]
        
        # 6. 直接调用rank_bm25计算得分（无需手写BM25逻辑！）
        if model == "bm25":
            # 初始化BM25模型（支持k1、b参数配置）
            bm25 = BM25Okapi(
                para_tokenized,
                k1=config.get("bm25_k1", 1.5),
                b=config.get("bm25_b", 0.5)
            )
            # 计算Query与所有段落的相似度得分
            final_scores = bm25.get_scores(query_tokens)
        
        elif model == "jaccard":
            # 保留Jaccard作为备选
            query_set = set(query_tokens)
            final_scores = []
            for tokens in para_tokenized:
                para_set = set(tokens)
                union = len(query_set | para_set)
                final_scores.append(len(query_set & para_set) / union if union else 0.0)
        
        else:
            raise ValueError(f"Unsupported model: {model} (choose 'bm25'/'jaccard')")

    # 7. 排序返回结果
    scored_paragraphs = list(enumerate(final_scores))
    scored_paragraphs.sort(key=lambda x: x[1], reverse=True)

    top_results = [
        {"paragraph": paragraphs[idx], "score": float(score)}
        for idx, score in scored_paragraphs[:topk]
    ]
    
    all_paras = [
        {
            "paragraph": paragraphs[idx],
            "score": float(score),
            "length": len(paragraphs[idx]),
            "token_count": len(tokenize(paragraphs[idx]))
        }
        for idx, score in scored_paragraphs
    ]
    all_results = {
        "num_paragraphs": len(paragraphs),
        "paragraphs": all_paras,
        "model": model,
    }

    logger.success(f"Top-{topk} paragraphs (highest score: {scored_paragraphs[0][1]:.4f})")
    return top_results, all_results
