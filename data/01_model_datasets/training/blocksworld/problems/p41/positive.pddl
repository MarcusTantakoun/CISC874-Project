(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b5) (clear b6) (on b1 b10) (on b10 b9) (on b2 b4) (on b4 b8) (on b5 b3) (on b6 b1) (on b8 b7) (on b9 b2) (on-table b3) (on-table b7))
    (:goal (and (on b1 b2) (on b3 b8) (on b4 b1) (on b5 b4) (on b7 b6) (on b9 b10) (on b10 b3)))
)