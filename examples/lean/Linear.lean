-- Generated from weightlang.linear-proof.v1; formula SHA256: 26aad89a61229dd5cb9274c66f8a7e3f6032a639e0330790092e5fbfd94223d1
namespace WeightLang
abbrev InputWidth := 3
abbrev OutputWidth := 2
-- Fin indexes carry the exact input/output widths. Weight layout is [out,in].
def dot {S : Type} [Zero S] [Add S] [Mul S]
    (x : Fin InputWidth → S) (w : Fin InputWidth → S) : S :=
  (List.finRange InputWidth).foldl (fun acc i => acc + x i * w i) 0

def linear {S R : Type} [Zero S] [Add S] [Mul S]
    (x : R → Fin InputWidth → S)
    (w : Fin OutputWidth → Fin InputWidth → S) (b : Fin OutputWidth → S)
    : R → Fin OutputWidth → S :=
  fun r o => dot (x r) (w o) + b o

-- Sum A B indexes concatenated rows without changing feature axes.
def rows {A B C : Type} (x : A → C) (y : B → C) : Sum A B → C :=
  Sum.elim x y

theorem linear_rows {S A B : Type} [Zero S] [Add S] [Mul S]
    (x : A → Fin InputWidth → S) (y : B → Fin InputWidth → S)
    (w : Fin OutputWidth → Fin InputWidth → S) (b : Fin OutputWidth → S) :
    linear (rows x y) w b = rows (linear x w b) (linear y w b) := by
  funext r o
  cases r <;> rfl

#print axioms linear_rows
end WeightLang
