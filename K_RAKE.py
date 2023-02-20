from kiwipiepy import Kiwi, Token
from collections import Counter, defaultdict
from itertools import chain, groupby, product
from typing import DefaultDict, Dict, List, Set, Tuple


Word = str
Sentence = str
Phrase = Tuple[str, ...]


class K_RAKE:
    def __init__(
        self,
        tokenizer: Kiwi,
        stopwords: Set[str] = set(),
    ):
        self.stopwords = stopwords
        self.tokenizer = tokenizer
        
        self.allowed = {'NNG', 'NNP', 'SL'}
        self.recover_list: List[Token] = []
        self.phrase_list: List[Phrase] = []
        
        self.frequency_dist: Dict[Word, int]
        self.degree: Dict[Word, int]
        self.rank_list: List[Tuple[float, Sentence]]
        self.phrase_frequency: Dict[Word, int]
    
    def extract_keywords(self, document: List[Sentence]) -> List[Tuple[float, Sentence]]:
        self._build(document)
        return self.rank_list
    
    def _recover(self, tokens: Tuple[Token]) -> str:
        recovered = tokens[0][0]
        for i in range(1, len(tokens)):
            if tokens[i - 1][2] + tokens[i - 1][3] == tokens[i][2]:
                recovered += tokens[i][0]
            else:
                recovered += ' ' + tokens[i][0]
        return recovered
    
    def _build(self, document: List[Sentence]) -> None:
        self._build_phrase_list(document)
        self._build_frequency_dist()
        self._build_co_occurance_graph()
        self._build_rank_list()
         
    def _build_frequency_dist(self):
        self.frequency_dist = Counter(chain.from_iterable(self.phrase_list))
         
    def _build_co_occurance_graph(self):
        co_occurance_graph: DefaultDict[Word, DefaultDict[Word, int]] = defaultdict(lambda: defaultdict(lambda: 0))
        for phrase in self.phrase_list:
            for (word, coword) in product(phrase, phrase):
                co_occurance_graph[word][coword] += 1
        self.degree = defaultdict(lambda: 0)
        for key in co_occurance_graph:
            self.degree[key] = sum(co_occurance_graph[key].values())
        
    def _build_rank_list(self):
        self.rank_list = []
        self.phrase_frequency = defaultdict(int)
        rank_set = set()
        for i, phrase in enumerate(self.phrase_list):
            rank = 0.0
            for word in phrase:
                rank += 1.0 * self.degree[word] / self.frequency_dist[word]
            
            recovered_phrase = self._recover(self.recover_list[i])
            self.phrase_frequency[recovered_phrase] += 1
            rank_set.add((rank, recovered_phrase))
            
        #  if a phrase is nested in the other phrase, it's deleted
        delset = set()
        
        for score, word in rank_set:
            for _, compare in rank_set:
                if word == compare:
                    continue
                if word in compare:
                    delset.add((score, word))
        
        rank_set -= delset
        
        self.rank_list = list(sorted(list(rank_set), reverse=True))
        self.ranked_phrases = [ph[1] for ph in self.rank_list]
             
    def _build_phrase_list(self, document: List[Sentence]) -> None:
        for sentence in document:
            token_list: List[Token] = self.tokenizer.tokenize(sentence)
            groups = groupby(token_list, lambda x : x[0] not in self.stopwords and x[1] in self.allowed)
            
            token_phrases = [tuple(group[1]) for group in groups if group[0]]
            self.recover_list.extend(token_phrases)
            self.phrase_list.extend([tuple(x[0] for x in tup) for tup in token_phrases]) 
