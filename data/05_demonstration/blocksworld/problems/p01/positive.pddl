(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b1) (clear b2) (on b1 b8) (on b10 b5) (on b2 b6) (on b3 b4) (on b6 b7) (on b7 b9) (on b8 b10) (on b9 b3) (on-table b4) (on-table b5))
    (:goal (and (on b1 b6) (on b2 b1) (on b3 b8) (on b6 b4) (on b8 b10) (on b9 b3) (on b10 b5)))
)