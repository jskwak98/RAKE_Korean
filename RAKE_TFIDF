from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple
from math import log


Phrase = str
Document_ID = int


class TF_IDF_UPON_RAKE:
    """
    For Multiple Documents, you may want to consider TF-IDF score of keywords
    combined with the RAKE score itself.

    On Development
    """
    def __init__(
        self,
        idx_kwd: Dict[Document_ID, Dict],
        idf: Dict[Phrase, int]
    ):
        self.idx_kwd = idx_kwd
        self.idf = idf
        self.d = len(idx_kwd)
        
        # {int : [(KWD, rakescore, tfidfscore)]}
        self.metric: DefaultDict[Document_ID, List[Tuple[str, float, float]]] = defaultdict(list)
        
    def compute_tfidf(self) -> None:
        for idx in self.idx_kwd:
            # compute tfidf
            if self.idx_kwd[idx]['tf']:
                max_tf = max(self.idx_kwd[idx]['tf'].values())
                for rake_score, kwd in self.idx_kwd[idx]['kwds']:
                    tf = self.idx_kwd[idx]['tf'][kwd]

                    term_freq = 0.5 + (0.5 * tf) / max_tf  # tf
                    inverse_doc_freq = log(self.d / self.idf[kwd])  # idf

                    tfidf = term_freq * inverse_doc_freq

                    self.metric[idx].append((kwd, rake_score, tfidf))

    def rank_with_tfidf(self) -> DefaultDict[Document_ID, List[Tuple[float, Phrase]]]:
        rank = defaultdict(list)
        for idx in self.metric:
            for kwd, rake, tfidf in self.metric[idx]:
                rank[idx].append((tfidf, kwd))
            rank[idx] = list(sorted(rank[idx], reverse=True))
        return rank
                
    def compute_metric(self, method='multiply', alpha=0.5, top=10) -> DefaultDict[Document_ID, List[Tuple[float, Phrase]]]:
        """
        On Development
        
        Currently implemented:
        1. Multiply scaled_rake score with TFIDF value, scaling is conducted using log function and dividing phrase with the number of words
        2. Weighted average of rake score and the TFIDF value
        
        TODO
        1. To support Top-K result cutoff
        2. To support Top-percentile result cutoff(top 30%, while keeping it not less than 10)
        """
        rank = defaultdict(list)
        for idx in self.metric:
            for kwd, rake, tfidf in self.metric[idx]:
                if method == 'multiply':
                    rake += 1.8
                    scaled_rake = log(rake) / len(kwd.split(" "))
                    rank[idx].append((scaled_rake * tfidf, kwd))
                else:
                    rank[idx].append((rake * alpha + tfidf * (1 - alpha), kwd))
            rank[idx] = list(sorted(rank[idx], reverse=True))
        return rank
