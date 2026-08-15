"""File 9/14: Neuro-symbolic Panini interface.

Combines a Kāraka dependency graph with an additive symbolic attention-bias
matrix. PyTorch is optional; the core reference implementation uses stdlib.
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from karaka_dependency import KarakaDependencyEngine, KarakaGraph
except ImportError:
    KarakaDependencyEngine = None
    KarakaGraph = Any

DEFAULT_RELATION_BIAS = {
    "kartā": 5.0, "karma": 5.0, "karaṇa": 5.0,
    "sampradāna": 5.0, "apādāna": 5.0, "adhikaraṇa": 5.0,
    "sambandha": 3.0, "adhikāra": 2.0, "unknown": 0.0,
}

@dataclass
class BiasConfig:
    relation_bias: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RELATION_BIAS))
    reverse_scale: float = 0.5
    confidence_floor: float = 0.2
    distance_decay: float = 0.0
    diagonal_bias: float = 0.0

    def to_dict(self): return asdict(self)

@dataclass
class SymbolicRelation:
    source: int
    target: int
    relation: str
    score: float
    bias: float
    confidence: str
    evidence: List[str] = field(default_factory=list)

    def to_dict(self): return asdict(self)

@dataclass
class SymbolicRepresentation:
    sentence_id: str
    text: str
    token_count: int
    root_verbs: List[int]
    relations: List[SymbolicRelation]
    metadata: Dict[str, Any]

    def to_dict(self):
        return {"record_type":"panini_neuro_symbolic_representation",
                "sentence_id":self.sentence_id,"text":self.text,
                "token_count":self.token_count,"root_verbs":self.root_verbs,
                "relations":[r.to_dict() for r in self.relations],
                "metadata":self.metadata}

@dataclass
class PaniniNeuralLayerSpec:
    symbolic_bias_enabled: bool = True
    learned_symbolic_gate: bool = True
    relation_embedding_enabled: bool = True
    karaka_embedding_enabled: bool = True
    preserve_raw_attention: bool = True
    symbolic_bias_scale: float = 1.0
    def to_dict(self): return asdict(self)

def confidence_multiplier(confidence: str) -> float:
    return {"high":1.0,"medium":0.8,"low":0.55,"very_low":0.25}.get(confidence,0.2)

def softmax(values: Sequence[float]) -> List[float]:
    if not values: return []
    m=max(values); ex=[math.exp(x-m) for x in values]; s=sum(ex)
    return [x/s for x in ex]

def add_bias_to_logits(logits, bias):
    if len(logits)!=len(bias) or any(len(a)!=len(b) for a,b in zip(logits,bias)):
        raise ValueError("Logit and symbolic-bias matrices must have identical dimensions.")
    return [[float(logits[i][j])+float(bias[i][j]) for j in range(len(logits[i]))] for i in range(len(logits))]

def row_softmax_attention(logits): return [softmax(row) for row in logits]

def symbolic_attention(logits,bias): return row_softmax_attention(add_bias_to_logits(logits,bias))

def attention_entropy(probabilities):
    return -sum(p*math.log(p,2) for p in probabilities if p>0)

class SymbolicRepresentationBuilder:
    def __init__(self, config: Optional[BiasConfig]=None): self.config=config or BiasConfig()
    def _distance_factor(self,s,t):
        return math.exp(-self.config.distance_decay*abs(s-t)) if self.config.distance_decay>0 else 1.0
    def build(self, graph: KarakaGraph):
        relations=[]
        for edge in graph.edges:
            score=float(edge.score)
            conf="high" if score>=.85 else "medium" if score>=.65 else "low" if score>=.40 else "very_low"
            effective=max(score,self.config.confidence_floor)
            base=self.config.relation_bias.get(edge.relation,self.config.relation_bias.get("unknown",0.0))
            bias=base*effective*self._distance_factor(edge.source,edge.target)
            relations.append(SymbolicRelation(edge.source,edge.target,edge.relation,score,round(bias,6),conf,list(edge.evidence)))
        return SymbolicRepresentation(graph.sentence_id,graph.text,len(graph.nodes),list(graph.root_verbs),relations,
            {"bias_config":self.config.to_dict(),"symbolic_source":"KarakaGraph","uncertainty_preserved":True,
             "representation_type":"pairwise_symbolic_bias"})

class SymbolicAttentionBias:
    def __init__(self, reverse_scale=.5, diagonal_bias=0.0): self.reverse_scale=reverse_scale; self.diagonal_bias=diagonal_bias
    def build(self, representation):
        n=representation.token_count; m=[[0.0]*n for _ in range(n)]
        for i in range(n): m[i][i]=self.diagonal_bias
        for r in representation.relations:
            if 0<=r.source<n and 0<=r.target<n:
                m[r.source][r.target]+=r.bias
                m[r.target][r.source]+=r.bias*self.reverse_scale
        return m

def torch_symbolic_attention(logits, symbolic_bias):
    try: import torch
    except ImportError as exc: raise RuntimeError("PyTorch is not installed.") from exc
    if not isinstance(logits,torch.Tensor): raise TypeError("logits must be a torch.Tensor")
    if not isinstance(symbolic_bias,torch.Tensor): symbolic_bias=torch.tensor(symbolic_bias,dtype=logits.dtype,device=logits.device)
    symbolic_bias=symbolic_bias.to(device=logits.device,dtype=logits.dtype)
    if symbolic_bias.ndim==2: symbolic_bias=symbolic_bias[None,None,:,:]
    elif symbolic_bias.ndim==3: symbolic_bias=symbolic_bias[:,None,:,:]
    if symbolic_bias.ndim!=4: raise ValueError("symbolic_bias must be [seq,seq], [batch,seq,seq], or [batch,heads,seq,seq]")
    return logits+symbolic_bias

class PaniniNeuroSymbolicController:
    def __init__(self,bias_config=None,layer_spec=None):
        self.bias_config=bias_config or BiasConfig(); self.layer_spec=layer_spec or PaniniNeuralLayerSpec()
        self.builder=SymbolicRepresentationBuilder(self.bias_config)
    def represent(self,graph): return self.builder.build(graph)
    def attention_bias(self,representation):
        m=SymbolicAttentionBias(self.bias_config.reverse_scale,self.bias_config.diagonal_bias).build(representation)
        s=self.layer_spec.symbolic_bias_scale
        return [[v*s for v in row] for row in m]
    def prepare(self,graph):
        r=self.represent(graph); return {"representation":r.to_dict(),"attention_bias":self.attention_bias(r),"layer_spec":self.layer_spec.to_dict()}

def build_neuro_symbolic_representation(text,bias_scale=1.0,reverse_scale=.5,distance_decay=0.0):
    if KarakaDependencyEngine is None: raise RuntimeError("karaka_dependency.py is required")
    graph=KarakaDependencyEngine().analyze(text)
    return PaniniNeuroSymbolicController(BiasConfig(reverse_scale=reverse_scale,distance_decay=distance_decay),PaniniNeuralLayerSpec(symbolic_bias_scale=bias_scale)).prepare(graph)

def write_representation(payload,path):
    p=Path(path); p.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8"); return p

def render(graph,prepared):
    lines=["="*96,"PANINI NEURO-SYMBOLIC REPRESENTATION","="*96,f"Sentence: {graph.text}","","SYMBOLIC RELATIONS","-"*96]
    for r in prepared["representation"]["relations"]:
        lines.append(f"{r['source']} -> {r['target']} {r['relation']} score={r['score']:.3f} bias={r['bias']:.3f} {r['confidence']}")
    lines += ["","ADDITIVE SYMBOLIC ATTENTION BIAS","-"*96]
    for i,row in enumerate(prepared["attention_bias"]): lines.append(f"{i}: "+" ".join(f"{v:7.3f}" for v in row))
    lines += ["","NEURAL CONTRACT: attention_logits + symbolic_bias -> softmax"]
    return "\n".join(lines)

def self_test():
    if KarakaDependencyEngine is None: raise AssertionError("karaka_dependency.py unavailable")
    graph=KarakaDependencyEngine().analyze("राम पुस्तक घेऊन गेला.")
    c=PaniniNeuroSymbolicController(); p=c.prepare(graph); b=p["attention_bias"]
    assert len(b)==len(graph.nodes) and all(len(r)==len(graph.nodes) for r in b)
    a=symbolic_attention([[0.,0.],[0.,0.]],[[0.,5.],[2.5,0.]])
    assert a[0][1]>a[0][0] and a[1][0]>a[1][1]
    assert attention_entropy(a[0])>=0
    assert "attention_bias" in json.dumps(p,ensure_ascii=False)

def main(argv=None):
    ap=argparse.ArgumentParser(description="Paninian neuro-symbolic attention interface")
    ap.add_argument("text",nargs="?"); ap.add_argument("--bias-scale",type=float,default=1.0)
    ap.add_argument("--reverse-scale",type=float,default=.5); ap.add_argument("--distance-decay",type=float,default=0.0)
    ap.add_argument("--json"); ap.add_argument("--self-test",action="store_true"); args=ap.parse_args(argv)
    if args.self_test: self_test(); print("SELF_TEST_PASS"); return 0
    if not args.text: print("Provide a Marathi sentence or use --self-test."); return 2
    if KarakaDependencyEngine is None: print("karaka_dependency.py is required."); return 1
    graph=KarakaDependencyEngine().analyze(args.text)
    c=PaniniNeuroSymbolicController(BiasConfig(reverse_scale=args.reverse_scale,distance_decay=args.distance_decay),PaniniNeuralLayerSpec(symbolic_bias_scale=args.bias_scale))
    p=c.prepare(graph); print(render(graph,p))
    if args.json: write_representation(p,args.json); print(f"Representation written to: {args.json}")
    return 0

if __name__=="__main__": raise SystemExit(main())
