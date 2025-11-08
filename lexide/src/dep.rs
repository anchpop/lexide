use std::fmt;

/// Dependency relation types (Universal Dependencies)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash, Ord, PartialOrd, Copy)]
pub enum DependencyRelation {
    #[serde(rename = "acl")]
    Acl,
    #[serde(rename = "acl:relcl")]
    AclRelcl,
    #[serde(rename = "advcl")]
    Advcl,
    #[serde(rename = "advcl:relcl")]
    AdvclRelcl,
    #[serde(rename = "advmod")]
    Advmod,
    #[serde(rename = "advmod:emph")]
    AdvmodEmph,
    #[serde(rename = "advmod:lmod")]
    AdvmodLmod,
    #[serde(rename = "amod")]
    Amod,
    #[serde(rename = "appos")]
    Appos,
    #[serde(rename = "aux")]
    Aux,
    #[serde(rename = "aux:pass")]
    AuxPass,
    #[serde(rename = "case")]
    Case,
    #[serde(rename = "cc")]
    Cc,
    #[serde(rename = "cc:preconj")]
    CcPreconj,
    #[serde(rename = "ccomp")]
    Ccomp,
    #[serde(rename = "clf")]
    Clf,
    #[serde(rename = "compound")]
    Compound,
    #[serde(rename = "compound:lvc")]
    CompoundLvc,
    #[serde(rename = "compound:prt")]
    CompoundPrt,
    #[serde(rename = "compound:redup")]
    CompoundRedup,
    #[serde(rename = "compound:svc")]
    CompoundSvc,
    #[serde(rename = "conj")]
    Conj,
    #[serde(rename = "cop")]
    Cop,
    #[serde(rename = "csubj")]
    Csubj,
    #[serde(rename = "csubj:outer")]
    CsubjOuter,
    #[serde(rename = "csubj:pass")]
    CsubjPass,
    #[serde(rename = "dep")]
    Dep,
    #[serde(rename = "det")]
    Det,
    #[serde(rename = "det:numgov")]
    DetNumgov,
    #[serde(rename = "det:nummod")]
    DetNummod,
    #[serde(rename = "det:poss")]
    DetPoss,
    #[serde(rename = "discourse")]
    Discourse,
    #[serde(rename = "dislocated")]
    Dislocated,
    #[serde(rename = "expl")]
    Expl,
    #[serde(rename = "expl:impers")]
    ExplImpers,
    #[serde(rename = "expl:pass")]
    ExplPass,
    #[serde(rename = "expl:pv")]
    ExplPv,
    #[serde(rename = "fixed")]
    Fixed,
    #[serde(rename = "flat")]
    Flat,
    #[serde(rename = "flat:foreign")]
    FlatForeign,
    #[serde(rename = "flat:name")]
    FlatName,
    #[serde(rename = "goeswith")]
    Goeswith,
    #[serde(rename = "iobj")]
    Iobj,
    #[serde(rename = "list")]
    List,
    #[serde(rename = "mark")]
    Mark,
    #[serde(rename = "nmod")]
    Nmod,
    #[serde(rename = "nmod:poss")]
    NmodPoss,
    #[serde(rename = "nmod:tmod")]
    NmodTmod,
    #[serde(rename = "nsubj")]
    Nsubj,
    #[serde(rename = "nsubj:outer")]
    NsubjOuter,
    #[serde(rename = "nsubj:pass")]
    NsubjPass,
    #[serde(rename = "nummod")]
    Nummod,
    #[serde(rename = "nummod:gov")]
    NummodGov,
    #[serde(rename = "obj")]
    Obj,
    #[serde(rename = "obl")]
    Obl,
    #[serde(rename = "obl:agent")]
    OblAgent,
    #[serde(rename = "obl:arg")]
    OblArg,
    #[serde(rename = "obl:lmod")]
    OblLmod,
    #[serde(rename = "obl:tmod")]
    OblTmod,
    #[serde(rename = "orphan")]
    Orphan,
    #[serde(rename = "parataxis")]
    Parataxis,
    #[serde(rename = "punct")]
    Punct,
    #[serde(rename = "reparandum")]
    Reparandum,
    #[serde(rename = "root")]
    Root,
    #[serde(rename = "vocative")]
    Vocative,
    #[serde(rename = "xcomp")]
    Xcomp,
}

impl fmt::Display for DependencyRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", serde_plain::to_string(self).unwrap())
    }
}
