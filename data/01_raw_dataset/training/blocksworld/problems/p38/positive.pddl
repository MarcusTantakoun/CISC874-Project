(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b2) (on b1 b4) (on b10 b6) (on b2 b5) (on b3 b1) (on b4 b8) (on b5 b3) (on b6 b7) (on b7 b9) (on b8 b10) (on-table b9))
    (:goal (and (on b1 b10) (on b4 b3) (on b6 b2) (on b7 b1) (on b8 b7) (on b9 b4) (on b10 b5)))
)