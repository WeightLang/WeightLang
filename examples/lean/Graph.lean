-- Generated from weightlang.formula-graph.v1; formula SHA256: 48f3efdbf8c6bbd0bd2b49f0c480ed87374dd8127af5ceeb77f8c9760620f5da
namespace WeightLang
-- The generic scalar structure also admits real arithmetic instances.
def dot {S : Type} {n : Nat} [Zero S] [Add S] [Mul S]
    (x w : Fin n → S) : S :=
  (List.finRange n).foldl (fun acc i => acc + x i * w i) 0

def graph {S R : Type} [Zero S] [Add S] [Mul S]
    (x : R → Fin 3 → S)
    (w0 : Fin 4 → Fin 3 → S)
    (b0 : Fin 4 → S)
    (w2 : Fin 3 → Fin 4 → S) : R → Fin 3 → S :=
  fun r =>
    let n0 : Fin 4 → S := fun o => dot (x r) (w0 o) + b0 o
    let n1 : Fin 4 → S := fun o => (n0) o * (n0) o
    let n2 : Fin 3 → S := fun o => dot (n1) (w2 o)
    let n3 : Fin 3 → S := fun o => (n2) o + (x r) o
    n3

def rows {A B C : Type} (x : A → C) (y : B → C) : Sum A B → C := Sum.elim x y

theorem graph_rows {S A B : Type} [Zero S] [Add S] [Mul S]
    (x : A → Fin 3 → S) (y : B → Fin 3 → S)
    (w0 : Fin 4 → Fin 3 → S)
    (b0 : Fin 4 → S)
    (w2 : Fin 3 → Fin 4 → S) :
    graph (rows x y) w0 b0 w2 = rows (graph x w0 b0 w2) (graph y w0 b0 w2) := by
  funext r o
  cases r <;> rfl

#print axioms graph_rows
end WeightLang
