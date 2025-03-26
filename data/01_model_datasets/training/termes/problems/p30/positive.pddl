(define (problem termes-0072-4x3x6-random_towers_4x3_6_2_153)
    (:domain termes)
    (:objects n0 n1 n2 n3 n4 n5 n6 - numb pos-0-0 pos-0-1 pos-0-2 pos-1-0 pos-1-1 pos-1-2 pos-2-0 pos-2-1 pos-2-2 pos-3-0 pos-3-1 pos-3-2 - position)
    (:init (IS-DEPOT pos-2-0) (NEIGHBOR pos-0-0 pos-0-1) (NEIGHBOR pos-0-0 pos-1-0) (NEIGHBOR pos-0-1 pos-0-0) (NEIGHBOR pos-0-1 pos-0-2) (NEIGHBOR pos-0-1 pos-1-1) (NEIGHBOR pos-0-2 pos-0-1) (NEIGHBOR pos-0-2 pos-1-2) (NEIGHBOR pos-1-0 pos-0-0) (NEIGHBOR pos-1-0 pos-1-1) (NEIGHBOR pos-1-0 pos-2-0) (NEIGHBOR pos-1-1 pos-0-1) (NEIGHBOR pos-1-1 pos-1-0) (NEIGHBOR pos-1-1 pos-1-2) (NEIGHBOR pos-1-1 pos-2-1) (NEIGHBOR pos-1-2 pos-0-2) (NEIGHBOR pos-1-2 pos-1-1) (NEIGHBOR pos-1-2 pos-2-2) (NEIGHBOR pos-2-0 pos-1-0) (NEIGHBOR pos-2-0 pos-2-1) (NEIGHBOR pos-2-0 pos-3-0) (NEIGHBOR pos-2-1 pos-1-1) (NEIGHBOR pos-2-1 pos-2-0) (NEIGHBOR pos-2-1 pos-2-2) (NEIGHBOR pos-2-1 pos-3-1) (NEIGHBOR pos-2-2 pos-1-2) (NEIGHBOR pos-2-2 pos-2-1) (NEIGHBOR pos-2-2 pos-3-2) (NEIGHBOR pos-3-0 pos-2-0) (NEIGHBOR pos-3-0 pos-3-1) (NEIGHBOR pos-3-1 pos-2-1) (NEIGHBOR pos-3-1 pos-3-0) (NEIGHBOR pos-3-1 pos-3-2) (NEIGHBOR pos-3-2 pos-2-2) (NEIGHBOR pos-3-2 pos-3-1) (SUCC n1 n0) (SUCC n2 n1) (SUCC n3 n2) (SUCC n4 n3) (SUCC n5 n4) (SUCC n6 n5) (at pos-2-0) (height pos-0-0 n0) (height pos-0-1 n0) (height pos-0-2 n0) (height pos-1-0 n0) (height pos-1-1 n0) (height pos-1-2 n0) (height pos-2-0 n0) (height pos-2-1 n0) (height pos-2-2 n0) (height pos-3-0 n0) (height pos-3-1 n0) (height pos-3-2 n0))
    (:goal (and (height pos-0-0 n0) (height pos-0-1 n0) (height pos-0-2 n0) (height pos-1-0 n0) (height pos-1-1 n6) (height pos-1-2 n0) (height pos-2-0 n0) (height pos-2-1 n0) (height pos-2-2 n0) (height pos-3-0 n0) (height pos-3-1 n0) (height pos-3-2 n2) (not (has-block))))
)