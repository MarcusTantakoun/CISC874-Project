(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b10) (clear b3) (clear b9) (on b10 b6) (on b2 b4) (on b3 b5) (on b4 b8) (on b5 b1) (on b8 b7) (on b9 b2) (on-table b1) (on-table b6) (on-table b7))
    (:goal (and (on b1 b4) (on b2 b7) (on b4 b5) (on b7 b1) (on b8 b2) (on b9 b3)))
)