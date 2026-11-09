"""Parameters and system definitions. Defaults follow the research plan."""
from dataclasses import dataclass, replace, asdict

SYSTEMS = ("R0", "R1", "R2", "R2_nobond", "R2_nodecl")
ATTACKS = ("honest", "A1", "A1R", "A2", "A2v", "A3", "A3b", "A5", "A1S", "A3d")
E1_ATTACKS = ("A1", "A1R", "A2", "A3", "A3b", "A5")   # A2v is replaced by the reset diagnostic (E1b)


def uses_audits(system: str) -> bool:
    """R0 is peer rating only; every other system is scored from audits."""
    return system != "R0"


def terms_bind(system: str) -> bool:
    """Posted terms bind (undeclared price is a violation) in R2 and R2 - bond."""
    return system in ("R2", "R2_nobond")


def has_bond(system: str) -> bool:
    """A bond is posted in R2 and R2 - declaration."""
    return system in ("R2", "R2_nodecl")


@dataclass(frozen=True)
class Params:
    system: str = "R2"
    attack: str = "A1"          # coalition behaviour; "honest" = paired baseline
    seed: int = 0
    # market size
    T: int = 40
    n_slots: int = 8
    K: int = 2                   # coalition seller identities
    Q: int = 4                   # coalition buyer accounts
    n_buyers: int = 16           # honest buyers per round
    n_reserved: int = 3          # honest buyers reserved for newcomers
    capacity: int = 3
    # goods
    h: float = 0.5
    cost_L: float = 2.0
    cost_H: float = 6.0
    V: float = 10.0
    outside: float = 1.6
    markup: float = 1.0
    max_bill: float = 11.0
    # reputation and matching
    alpha: float = 3.0
    beta: float = 1.0
    tau: float = 1.0
    buyer_model: str = "mu"      # "mu" (default) or "split" robustness model
    bill_prior_weight: float = 4.0   # posted menu counts as this many observed bills
    newcomer_age: int = 5
    newcomer_every: int = 10
    # audits, bonds, identity, money
    p_audit: float = 0.05
    bond: float = 100.0
    M: int = 2                   # replacement identities beyond the initial ones
    endow_coalition: float = 3000.0
    endow_honest: float = 500.0
    linked: bool = False
    # attacks
    a1_start: int = 10
    a1_r: float = 0.3
    endgame: int = 3
    warmup: int = 10
    # reputation-sensitive slice (H1)
    fake_selfdeal: bool = False      # without audits, self-dealt transactions are pure fakes (no treatment cost)
    fake_selfdeal_all: bool = False  # diagnostic: free fakes even where audits exist (separates cost from evidence)
    incumbent_obs: int = 0           # initial honest sellers start with this many clean observations
    # noisy regime (E2)
    noisy: bool = False
    fn: float = 0.0
    fp: float = 0.0
    theta: float = 2.0

    def with_(self, **kw):
        return replace(self, **kw)

    def as_dict(self):
        return asdict(self)
