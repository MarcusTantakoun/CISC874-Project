(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b10) (clear b4) (on b1 b3) (on b10 b6) (on b2 b8) (on b4 b1) (on b5 b2) (on b6 b9) (on b7 b5) (on b9 b7) (on-table b3) (on-table b8))
    (:goal (and (on b2 b5) (on b3 b8) (on b4 b9) (on b5 b4) (on b6 b2) (on b8 b6) (on b9 b7) (on b10 b1)))
)