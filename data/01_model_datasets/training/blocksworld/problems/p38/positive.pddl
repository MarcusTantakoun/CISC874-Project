(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b1) (clear b4) (clear b5) (on b1 b7) (on b2 b8) (on b3 b2) (on b5 b3) (on b7 b9) (on b8 b6) (on b9 b10) (on-table b10) (on-table b4) (on-table b6))
    (:goal (and (on b1 b2) (on b2 b8) (on b3 b5) (on b4 b7) (on b6 b4) (on b7 b3) (on b9 b10) (on b10 b1)))
)