import math
import collections


# N-grams
def get_ngrams(tokens, n):
    """
    从给定的token列表中提取n-gram。
    :param tokens: list, token列表
    :param n: int, n-gram的n值
    :return: collections.Counter, n-gram及其计数的Counter对象
    """
    if len(tokens) < n:
        return collections.Counter()

    # slide window to calculate n-grams
    ngrams = collections.Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])  # n-gram must be hashable, thus using tuple
        ngrams[ngram] += 1
    return ngrams


# P-n of each n-grams
def calculate_pn_sentence(candidate_tokens, references_tokens_list, n):
    """
    计算单个候选句相对于参考句列表的n-gram精确率 p_n。
    :param candidate_tokens: list, 候选句的token列表
    :param references_tokens_list: list of lists, 每个元素是一个参考句的token列表
    :param n: int, n-gram的n值
    :return: float, n-gram精确率 p_n
    """
    # count n-grams
    candidate_ngrams = get_ngrams(candidate_tokens, n)
    if not candidate_ngrams:
        return None

    # denominator of P-n, n-gram count of the candidate sentence
    candidate_ngram_count = sum(candidate_ngrams.values())

    # loop each n-gram word
    sum_clip_count = 0  # molecular of P-n, count of clipped n-gram
    for ngram, cand_count in candidate_ngrams.items():
        # max n-gram count in reference sentences
        max_ref_count = 0
        for ref_tokens in references_tokens_list:
            ref_ngrams = get_ngrams(ref_tokens, n)
            max_ref_count = max(max_ref_count, ref_ngrams.get(ngram, 0))

        # clipped count: n-gram count of the candidate should be less than max n-gram count of all refs
        clip_count = min(cand_count, max_ref_count)
        sum_clip_count += clip_count

    # calculate P-n
    pn = sum_clip_count / candidate_ngram_count
    return pn


# BLEU
def calculate_bleu_sentence(candidate_sentence, reference_sentences, max_n=4, epsilon=1e-12):
    """
    计算单个候选句相对于一组参考句的BLEU得分。
    :param candidate_sentence: str, 候选句子 (假设已分词，token以空格分隔)
    :param reference_sentences: list of str, 参考句子列表 (每个句子token以空格分隔)
    :param max_n: int, 计算BLEU时考虑的最大n-gram长度 (通常为4)
    :param epsilon: float, 用于平滑log(0)的极小值，避免数学错误
    :return: float, BLEU得分 (0到1之间)
    """
    # to tokens
    candidate_tokens = candidate_sentence.split()
    references_tokens_list = [ref.split() for ref in reference_sentences]

    # brevity penalty (bp), get candidate length
    candidate_len = len(candidate_tokens)
    ref_lens = [len(ref_tokens) for ref_tokens in references_tokens_list]

    # get effective reference length
    # effective ref length is the min length of all effective ref sentences (which have the closest length diff with cand)
    effective_r = 0
    closest_diff = float('inf')
    possible_r_values = []
    for r_len in ref_lens:
        diff = abs(candidate_len - r_len)  # note absolute sentence length difference
        if diff < closest_diff:
            closest_diff = diff
            possible_r_values = [r_len]
        elif diff == closest_diff:
            possible_r_values.append(r_len)
    effective_r = min(possible_r_values)  # if there are more than one effective reference sentences

    # calculate BP
    if candidate_len > effective_r:
        bp = 1.0
    else:
        bp = math.exp(1 - effective_r / candidate_len)

    # n-gram
    log_pn_sum = 0.0
    weights = [1 / max_n] * max_n  # 对于BLEU-N，权重通常均匀分配为1/N
    for n_val in range(1, max_n + 1):
        pn = calculate_pn_sentence(candidate_tokens, references_tokens_list, n_val)
        log_pn_sum += weights[n_val - 1] * math.log(pn + epsilon)  # epsilon is for avoiding log(0) case

    # sentence BLEU score
    bleu_score = bp * math.exp(log_pn_sum)
    return bleu_score


# self-BLEU
def calculate_self_bleu(corpus_sentences, max_n=4, epsilon=1e-12):
    """
    计算给定语料库的Self-BLEU得分。
    Self-BLEU衡量语料库中句子之间的多样性。
    每个句子轮流作为候选句，其余句子作为参考句。
    :param corpus_sentences: list of str, 句子列表 (语料库), 每个句子token以空格分隔
    :param max_n: int, 计算BLEU时考虑的最大n-gram长度
    :param epsilon: float, 用于平滑log(0)的极小值
    :return: float, Self-BLEU得分
    """
    # check the corpus length
    num_sentences = len(corpus_sentences)
    if num_sentences < 2:
        print("Sentences in corpus is less than 2, cannot calculate Self-BLEU.")
        return None

    # BLEU for each candidate sentence and rest ref sentences
    total_bleu_score = 0.0
    processed_sentences = 0
    for i in range(num_sentences):
        candidate = corpus_sentences[i]
        references = corpus_sentences[:i] + corpus_sentences[i + 1:]

        # BLEU score between candidate and reference sentences
        bleu_for_sentence = calculate_bleu_sentence(candidate, references, max_n, epsilon)
        total_bleu_score += bleu_for_sentence
        processed_sentences += 1

    # take the avg and get self-BLEU score
    self_bleu = total_bleu_score / processed_sentences if processed_sentences > 0 else 0.0
    return self_bleu
