(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b3) (clear b9) (on b1 b2) (on b10 b7) (on b3 b6) (on b4 b10) (on b5 b1) (on b6 b4) (on b7 b5) (on b9 b8) (on-table b2) (on-table b8))
    (:goal (and (on b4 b5) (on b5 b9) (on b6 b8) (on b7 b2) (on b8 b1) (on b9 b6) (on b10 b3)))
)