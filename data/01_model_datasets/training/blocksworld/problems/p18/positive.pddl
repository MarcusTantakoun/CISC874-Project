(define (problem BW-rand-10)
    (:domain blocksworld-4ops)
    (:objects b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (arm-empty) (clear b10) (clear b8) (on b10 b4) (on b3 b9) (on b4 b6) (on b5 b1) (on b6 b3) (on b7 b2) (on b8 b7) (on b9 b5) (on-table b1) (on-table b2))
    (:goal (and (on b2 b10) (on b3 b7) (on b4 b3) (on b5 b8) (on b6 b1) (on b8 b2) (on b9 b6) (on b10 b4)))
)