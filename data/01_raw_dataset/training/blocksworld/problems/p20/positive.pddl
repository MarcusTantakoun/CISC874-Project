(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b4) (clear b5) (clear b8) (on b1 b2) (on b2 b10) (on b3 b9) (on b4 b6) (on b5 b1) (on b6 b3) (on b9 b7) (on-table b10) (on-table b7) (on-table b8))
    (:goal (and (on b1 b7) (on b2 b5) (on b5 b8) (on b6 b3) (on b7 b4) (on b8 b1) (on b10 b2)))
)